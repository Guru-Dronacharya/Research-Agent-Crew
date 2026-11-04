"""RAG Ingestion, Vector Persistence, and Retrieval layer.

Provides SSRF-safe page fetching, PDF/text document parsing, recursive text chunking,
ONNX-based and fake embedders, hybrid vector + BM25 search with MMR diversification,
and thread document repositories.
"""

from core.rag.parse import (
    parse_document_bytes,
    stream_and_validate_upload_bytes,
    sanitize_filename,
    detect_file_type,
    MAX_UPLOAD_MB,
)
from core.rag.chunk import chunk_text, compute_content_hash
from core.rag.embed import (
    Embedder,
    FastEmbedEmbedder,
    FakeDeterministicEmbedder,
    get_embedder,
    reset_embedder_cache,
)
from core.rag.fetch import (
    fetch_and_extract_page,
    fetch_raw_body_safely,
    validate_and_resolve_host,
    is_ip_forbidden,
    parse_numeric_or_obfuscated_ip,
    is_robots_allowed,
    FetchedPage,
    SSRFValidationError,
    FetchQuotaExceededError,
)
from core.rag.store import (
    query_hybrid_rag,
    ingest_document_text,
    reindex_thread_chunks,
    invalidate_thread_cache,
    clear_all_rag_caches,
    RAG_MIN_SCORE,
    RAG_MAX_CONTEXT_CHARS,
)

__all__ = [
    "parse_document_bytes",
    "stream_and_validate_upload_bytes",
    "sanitize_filename",
    "detect_file_type",
    "MAX_UPLOAD_MB",
    "chunk_text",
    "compute_content_hash",
    "Embedder",
    "FastEmbedEmbedder",
    "FakeDeterministicEmbedder",
    "get_embedder",
    "reset_embedder_cache",
    "fetch_and_extract_page",
    "fetch_raw_body_safely",
    "validate_and_resolve_host",
    "is_ip_forbidden",
    "parse_numeric_or_obfuscated_ip",
    "is_robots_allowed",
    "FetchedPage",
    "SSRFValidationError",
    "FetchQuotaExceededError",
    "query_hybrid_rag",
    "ingest_document_text",
    "reindex_thread_chunks",
    "invalidate_thread_cache",
    "clear_all_rag_caches",
    "RAG_MIN_SCORE",
    "RAG_MAX_CONTEXT_CHARS",
]
