"""RAG Vector Store, hybrid BM25 search, MMR diversification, and in-memory thread cache.

Provides normalized cosine similarity, reciprocal rank fusion (RRF) with BM25,
MMR diversity ranking, per-thread in-memory index caching with invalidation,
strict model filtering, and offline reindexing helpers.
"""

import os
import re
import uuid
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

from core.db.repositories import (
    DocumentRepo,
    ChunkRepo,
    DocumentRecord,
    ChunkRecord,
)
from core.state import RetrievedChunk
from core.rag.embed import Embedder
from core.rag.chunk import chunk_text, compute_content_hash

logger = logging.getLogger(__name__)

RAG_MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.35"))
RAG_MAX_CONTEXT_CHARS = int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "24000"))


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize string into lowercase alphanumeric words for BM25 indexing."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in cleaned.split() if len(w) > 1]


@dataclass
class ThreadIndexCache:
    """In-memory cache of chunk embeddings and BM25 index for a thread."""
    thread_id: str
    model_name: str
    chunks: List[ChunkRecord]
    embedding_matrix: np.ndarray  # shape: (N, D)
    bm25: Optional[BM25Okapi]


# In-memory index cache keyed by thread_id
_thread_cache: Dict[str, ThreadIndexCache] = {}


def invalidate_thread_cache(thread_id: str) -> None:
    """Invalidate cached embeddings matrix and BM25 index for a thread."""
    if thread_id in _thread_cache:
        del _thread_cache[thread_id]


def clear_all_rag_caches() -> None:
    """Clear all cached thread vector indices."""
    _thread_cache.clear()


async def get_or_build_thread_index(
    thread_id: str,
    embedder: Embedder,
    chunk_repo: ChunkRepo,
    force_refresh: bool = False
) -> Optional[ThreadIndexCache]:
    """Retrieve or build in-memory embedding matrix and BM25 index for a thread.

    Strictly filters chunks by embedder.get_model_name(): chunks embedded with another
    model are excluded from search results.
    """
    model_name = embedder.get_model_name()
    if not force_refresh and thread_id in _thread_cache:
        cached = _thread_cache[thread_id]
        if cached.model_name == model_name:
            return cached

    # Fetch chunks matching current embedding model
    chunks = await chunk_repo.list_chunks_by_thread(thread_id, embedding_model=model_name)
    if not chunks:
        return None

    # Filter chunks with valid embeddings
    valid_chunks: List[ChunkRecord] = []
    vector_list: List[np.ndarray] = []

    for c in chunks:
        if c.embedding:
            try:
                arr = np.frombuffer(c.embedding, dtype=np.float32)
                # Ensure vector is normalized
                norm = np.linalg.norm(arr)
                if norm > 1e-9:
                    arr = arr / norm
                vector_list.append(arr)
                valid_chunks.append(c)
            except Exception:
                continue

    if not valid_chunks or not vector_list:
        return None

    matrix = np.vstack(vector_list)

    # Build BM25 index
    tokenized_corpus = [tokenize_for_bm25(c.text) for c in valid_chunks]
    bm25_index = BM25Okapi(tokenized_corpus) if any(tokenized_corpus) else None

    cache_obj = ThreadIndexCache(
        thread_id=thread_id,
        model_name=model_name,
        chunks=valid_chunks,
        embedding_matrix=matrix,
        bm25=bm25_index
    )
    _thread_cache[thread_id] = cache_obj
    return cache_obj


def compute_mmr(
    query_vec: np.ndarray,
    candidate_indices: List[int],
    embedding_matrix: np.ndarray,
    relevance_scores: Dict[int, float],
    top_k: int,
    lambda_mult: float = 0.7
) -> List[int]:
    """Maximal Marginal Relevance (MMR) ranking for diverse chunk retrieval.

    Balances relevance to query against redundancy with already selected chunks.
    """
    if not candidate_indices or top_k <= 0:
        return []

    selected: List[int] = []
    remaining = list(candidate_indices)

    # First chunk is highest relevance score
    first_choice = max(remaining, key=lambda idx: relevance_scores.get(idx, 0.0))
    selected.append(first_choice)
    remaining.remove(first_choice)

    while remaining and len(selected) < top_k:
        best_idx = None
        best_score = -float("inf")

        for idx in remaining:
            rel = relevance_scores.get(idx, 0.0)
            candidate_vec = embedding_matrix[idx]

            # Max similarity to any already selected chunk
            max_sim = max(float(np.dot(candidate_vec, embedding_matrix[sel_idx])) for sel_idx in selected)

            mmr_score = lambda_mult * rel - (1.0 - lambda_mult) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
        else:
            break

    return selected


async def query_hybrid_rag(
    thread_id: str,
    query: str,
    embedder: Embedder,
    chunk_repo: ChunkRepo,
    document_repo: Optional[DocumentRepo] = None,
    top_k: int = 5,
    min_score: float = RAG_MIN_SCORE,
    subtask_id: Optional[str] = None
) -> Tuple[List[RetrievedChunk], bool]:
    """Perform hybrid vector + BM25 search with MMR diversification.

    Returns:
        (retrieved_chunks, evidence_gap)
        evidence_gap is True if no retrieved chunk scores >= min_score.
    """
    index_cache = await get_or_build_thread_index(thread_id, embedder, chunk_repo)
    if index_cache is None or len(index_cache.chunks) == 0:
        return [], True

    # 1. Vector similarity
    q_vec_list = embedder.embed_texts([query])
    if not q_vec_list:
        return [], True
    q_vec = np.asarray(q_vec_list[0], dtype=np.float32)
    q_norm = np.linalg.norm(q_vec)
    if q_norm > 1e-9:
        q_vec = q_vec / q_norm

    # Dot product for cosine similarity with pre-normalized embeddings
    cos_sims = np.dot(index_cache.embedding_matrix, q_vec)

    # 2. BM25 scoring
    q_tokens = tokenize_for_bm25(query)
    bm25_scores = np.zeros(len(index_cache.chunks), dtype=np.float32)
    if index_cache.bm25 and q_tokens:
        bm25_raw = index_cache.bm25.get_scores(q_tokens)
        bm25_scores = np.asarray(bm25_raw, dtype=np.float32)

    # 3. Reciprocal Rank Fusion (RRF)
    # Sort by vector similarity descending
    vec_rank_order = np.argsort(-cos_sims)
    bm25_rank_order = np.argsort(-bm25_scores)

    rrf_scores: Dict[int, float] = {}
    k_rrf = 60.0

    for rank, idx in enumerate(vec_rank_order):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_rrf + rank + 1))

    for rank, idx in enumerate(bm25_rank_order):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k_rrf + rank + 1))

    # Candidate filtering by cosine threshold or top candidates
    candidate_indices = [idx for idx in range(len(index_cache.chunks))]
    # MMR diversification over candidate pool
    diverse_indices = compute_mmr(
        query_vec=q_vec,
        candidate_indices=candidate_indices,
        embedding_matrix=index_cache.embedding_matrix,
        relevance_scores=rrf_scores,
        top_k=top_k * 2
    )

    # Fetch document metadata if document_repo is available
    doc_map: Dict[str, DocumentRecord] = {}
    if document_repo:
        doc_records = await document_repo.list_documents_by_thread(thread_id)
        doc_map = {d.id: d for d in doc_records}

    retrieved: List[RetrievedChunk] = []
    max_score_seen = 0.0

    for idx in diverse_indices[:top_k]:
        chunk = index_cache.chunks[idx]
        sim = float(cos_sims[idx])
        if sim > max_score_seen:
            max_score_seen = sim

        doc = doc_map.get(chunk.document_id)
        source_url = doc.url_or_name if doc else chunk.meta.get("url", "")
        title = doc.title if doc else chunk.meta.get("title", "")

        retrieved.append(
            RetrievedChunk(
                id=chunk.id,
                document_id=chunk.document_id,
                thread_id=chunk.thread_id,
                text=chunk.text,
                score=round(sim, 4),
                source_url=source_url,
                title=title,
                subtask_id=subtask_id,
                meta=chunk.meta
            )
        )

    # Evidence gap: True if no chunks retrieved or highest score < min_score
    evidence_gap = (len(retrieved) == 0) or (max_score_seen < min_score)
    return retrieved, evidence_gap


async def ingest_document_text(
    thread_id: str,
    text: str,
    title: str,
    url_or_name: str,
    kind: str,
    embedder: Embedder,
    document_repo: DocumentRepo,
    chunk_repo: ChunkRepo,
    run_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    doc_id: Optional[str] = None
) -> Tuple[DocumentRecord, bool, List[ChunkRecord]]:
    """Ingest, chunk, embed, and store a document for a thread.

    Handles (thread_id, content_hash) deduplication cleanly.
    Returns (document_record, is_duplicate, chunk_records).
    """
    content_hash = compute_content_hash(text)
    doc_record, is_duplicate = await document_repo.create_document(
        thread_id=thread_id,
        kind=kind,
        title=title,
        url_or_name=url_or_name,
        content_hash=content_hash,
        run_id=run_id,
        status="indexed",
        meta=meta,
        doc_id=doc_id
    )

    if is_duplicate:
        # Document already indexed; fetch existing chunks
        existing_chunks = await chunk_repo.list_chunks_by_thread(
            thread_id,
            embedding_model=embedder.get_model_name()
        )
        doc_chunks = [c for c in existing_chunks if c.document_id == doc_record.id]
        return doc_record, True, doc_chunks

    # Chunk the text
    chunks_meta = chunk_text(text)
    if not chunks_meta:
        return doc_record, False, []

    chunk_texts = [c["text"] for c in chunks_meta]
    vectors = embedder.embed_texts(chunk_texts)
    model_name = embedder.get_model_name()
    dim = embedder.get_dimension()

    chunk_records: List[ChunkRecord] = []
    for item, vec in zip(chunks_meta, vectors):
        arr = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 1e-9:
            arr = arr / norm
        blob = arr.tobytes()

        chunk_id = str(uuid.uuid4())
        chunk_records.append(
            ChunkRecord(
                id=chunk_id,
                document_id=doc_record.id,
                thread_id=thread_id,
                idx=item["idx"],
                text=item["text"],
                token_count=item["token_count"],
                embedding=blob,
                embedding_model=model_name,
                embedding_dim=dim,
                meta={
                    **(meta or {}),
                    "url": url_or_name,
                    "title": title,
                    "char_count": item["char_count"]
                }
            )
        )

    # Insert chunks with quota validation
    await chunk_repo.insert_chunks(thread_id, chunk_records)
    # Invalidate in-memory cache
    invalidate_thread_cache(thread_id)
    return doc_record, False, chunk_records


async def reindex_thread_chunks(
    thread_id: str,
    embedder: Embedder,
    chunk_repo: ChunkRepo,
    batch_size: int = 32
) -> int:
    """Reindex any chunks in a thread that lack embeddings or use a different model.

    Returns the number of chunks successfully re-embedded.
    """
    target_model = embedder.get_model_name()
    chunks_to_reindex = await chunk_repo.list_chunks_needing_reindex(thread_id, target_model)
    if not chunks_to_reindex:
        return 0

    dim = embedder.get_dimension()
    reindexed_count = 0

    for i in range(0, len(chunks_to_reindex), batch_size):
        batch = chunks_to_reindex[i:i + batch_size]
        texts = [c.text for c in batch]
        vectors = embedder.embed_texts(texts)

        for chunk, vec in zip(batch, vectors):
            arr = np.asarray(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 1e-9:
                arr = arr / norm
            blob = arr.tobytes()

            await chunk_repo.update_chunk_embedding(
                chunk_id=chunk.id,
                embedding=blob,
                embedding_model=target_model,
                embedding_dim=dim
            )
            reindexed_count += 1

    invalidate_thread_cache(thread_id)
    return reindexed_count
