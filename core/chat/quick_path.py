"""Quick-answer workflow for short factual queries.

Executes:
1. Exactly ONE search via the configured search tool.
2. SSRF-safe fetching of the top 1-2 pages.
3. Ingestion into the thread's document and chunk store.
4. Hybrid RAG retrieval.
5. Returns chunks and confidence indicators for a single streamed LLM answer.
"""

import os
import json
import uuid
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

from mcp_tools.client import MCPClientManager
from core.rag import (
    fetch_and_extract_page,
    ingest_document_text,
    query_hybrid_rag,
    get_embedder,
    chunk_text,
    RetrievedChunk
)
from core.db.repositories import DocumentRepo, ChunkRepo, ChunkQuotaExceededError

logger = logging.getLogger(__name__)

QUICK_SEARCH_BUDGET = int(os.environ.get("QUICK_SEARCH_BUDGET", "1"))
CHAT_MIN_SCORE = float(os.environ.get("CHAT_MIN_SCORE", "0.35"))
MAX_CHUNKS_PER_THREAD = int(os.environ.get("MAX_CHUNKS_PER_THREAD", "5000"))


async def execute_quick_path_retrieval(
    thread_id: str,
    query: str,
    db_path: str = None,
    max_pages: int = 2
) -> Tuple[List[RetrievedChunk], bool, List[Dict[str, str]]]:
    """Execute quick search, page fetch, thread indexing, and retrieval.

    Returns:
        (retrieved_chunks, low_confidence, sources_list)
    """
    mcp_client = MCPClientManager()
    search_payload = await mcp_client.call_tool(
        "search_web",
        {"query": query, "max_results": 3}
    )

    results: List[Dict[str, Any]] = []
    if isinstance(search_payload, dict):
        output = search_payload.get("output")
        if isinstance(output, dict):
            results = output.get("results", [])
        elif isinstance(output, str):
            try:
                results = json.loads(output).get("results", [])
            except Exception:
                pass
        if not results:
            results = search_payload.get("results", [])
    elif isinstance(search_payload, str):
        try:
            parsed = json.loads(search_payload)
            if isinstance(parsed, dict):
                output = parsed.get("output")
                if isinstance(output, dict):
                    results = output.get("results", [])
                else:
                    results = parsed.get("results", [])
        except Exception:
            results = []

    if not results:
        return [], False, []

    doc_repo = DocumentRepo(db_path=db_path)
    chunk_repo = ChunkRepo(db_path=db_path)
    embedder = get_embedder()

    all_snippet_only = True
    sources: List[Dict[str, str]] = []
    in_memory_chunks: List[RetrievedChunk] = []

    current_chunks = await chunk_repo.count_chunks_by_thread(thread_id)
    max_chunks = int(os.environ.get("MAX_CHUNKS_PER_THREAD", str(MAX_CHUNKS_PER_THREAD)))

    # Fetch top 1-2 pages
    for r in results[:max_pages]:
        url = r.get("url", "")
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        if not url:
            continue

        sources.append({"url": url, "title": title, "snippet": snippet})

        try:
            page = await fetch_and_extract_page(url=url, title=title, snippet=snippet)
            if not page.snippet_only:
                all_snippet_only = False

            chunks_meta = chunk_text(page.text)
            if not chunks_meta:
                continue

            # Check if ingesting would exceed thread chunk quota
            if current_chunks + len(chunks_meta) > max_chunks:
                logger.warning(
                    "Thread '%s' chunk count (%d) plus new chunks (%d) exceeds MAX_CHUNKS_PER_THREAD (%d). "
                    "Answering from in-memory chunks without persisting.",
                    thread_id, current_chunks, len(chunks_meta), max_chunks
                )
                # Build in-memory chunks without writing to SQLite
                q_vecs = embedder.embed_texts([query])
                q_vec = np.asarray(q_vecs[0], dtype=np.float32) if q_vecs else np.zeros(embedder.get_dimension(), dtype=np.float32)
                q_norm = np.linalg.norm(q_vec)
                c_texts = [c["text"] for c in chunks_meta]
                c_vecs = [np.asarray(v, dtype=np.float32) for v in embedder.embed_texts(c_texts)]
                for c_meta, c_v in zip(chunks_meta, c_vecs):
                    sim = float(np.dot(q_vec, c_v) / (q_norm * np.linalg.norm(c_v) + 1e-9))
                    if sim >= CHAT_MIN_SCORE:
                        in_memory_chunks.append(
                            RetrievedChunk(
                                id=f"mem-{uuid.uuid4().hex[:6]}",
                                document_id="in-memory-quick",
                                thread_id=thread_id,
                                text=c_meta["text"],
                                score=round(sim, 4),
                                source_url=url,
                                title=title or url,
                                meta={"snippet_only": page.snippet_only, "quick_path": True, "in_memory": True}
                            )
                        )
            else:
                try:
                    await ingest_document_text(
                        thread_id=thread_id,
                        text=page.text,
                        title=title or url,
                        url_or_name=url,
                        kind="web",
                        embedder=embedder,
                        document_repo=doc_repo,
                        chunk_repo=chunk_repo,
                        meta={"snippet_only": page.snippet_only, "quick_path": True}
                    )
                    current_chunks += len(chunks_meta)
                except ChunkQuotaExceededError as q_err:
                    logger.warning("Chunk quota exceeded during quick path ingestion: %s. Answering in-memory.", q_err)
                    q_vecs = embedder.embed_texts([query])
                    q_vec = np.asarray(q_vecs[0], dtype=np.float32) if q_vecs else np.zeros(embedder.get_dimension(), dtype=np.float32)
                    q_norm = np.linalg.norm(q_vec)
                    c_texts = [c["text"] for c in chunks_meta]
                    c_vecs = [np.asarray(v, dtype=np.float32) for v in embedder.embed_texts(c_texts)]
                    for c_meta, c_v in zip(chunks_meta, c_vecs):
                        sim = float(np.dot(q_vec, c_v) / (q_norm * np.linalg.norm(c_v) + 1e-9))
                        if sim >= CHAT_MIN_SCORE:
                            in_memory_chunks.append(
                                RetrievedChunk(
                                    id=f"mem-{uuid.uuid4().hex[:6]}",
                                    document_id="in-memory-quick",
                                    thread_id=thread_id,
                                    text=c_meta["text"],
                                    score=round(sim, 4),
                                    source_url=url,
                                    title=title or url,
                                    meta={"snippet_only": page.snippet_only, "quick_path": True, "in_memory": True}
                                )
                            )
        except Exception as exc:
            logger.debug("Failed to process page in quick path: %s", exc)

    # Retrieve relevant chunks from thread
    retrieved, gap = await query_hybrid_rag(
        thread_id=thread_id,
        query=query,
        embedder=embedder,
        chunk_repo=chunk_repo,
        document_repo=doc_repo,
        top_k=5,
        min_score=CHAT_MIN_SCORE
    )

    combined_chunks = in_memory_chunks + retrieved
    combined_chunks.sort(key=lambda c: c.score, reverse=True)

    low_confidence = all_snippet_only and len(sources) > 0
    return combined_chunks[:5], low_confidence, sources
