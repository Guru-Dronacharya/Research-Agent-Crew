-- 004_chat_locks.sql
-- DB-backed cross-process chat locks with heartbeats and owner tracking.

CREATE TABLE IF NOT EXISTS chat_locks (
    thread_id TEXT PRIMARY KEY REFERENCES threads(id) ON DELETE CASCADE,
    owner_id TEXT NOT NULL,
    heartbeat_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chat_locks_heartbeat ON chat_locks(heartbeat_at);
