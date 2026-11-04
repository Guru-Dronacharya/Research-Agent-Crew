-- 003_rag_documents_and_chunks.sql
-- Documents and chunks tables for RAG ingestion, retrieval, and thread uploads.

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
    kind TEXT NOT NULL CHECK (kind IN ('web', 'upload')),
    title TEXT NOT NULL,
    url_or_name TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'indexed' CHECK (status IN ('pending', 'fetching', 'indexed', 'failed')),
    created_at TEXT NOT NULL,
    meta_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (thread_id, content_hash)
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    thread_id TEXT NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    embedding BLOB,
    embedding_model TEXT,
    embedding_dim INTEGER,
    meta_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_documents_thread_id ON documents(thread_id);
CREATE INDEX IF NOT EXISTS idx_documents_thread_content_hash ON documents(thread_id, content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_thread_id ON chunks(thread_id);
CREATE INDEX IF NOT EXISTS idx_chunks_thread_model ON chunks(thread_id, embedding_model);
