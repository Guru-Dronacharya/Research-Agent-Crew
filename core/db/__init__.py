"""Database layer for Research Agent Crew.

Provides SQLite connection management in WAL mode with foreign keys enabled,
versioned schema migrations, and repository abstractions for threads, messages,
runs, and append-only run events.
"""

from core.db.connection import get_db, get_db_path
from core.db.migrator import apply_migrations
from core.db.repositories import (
    ThreadRepo,
    MessageRepo,
    RunRepo,
    EventRepo,
    DocumentRepo,
    ChunkRepo,
    ThreadRecord,
    MessageRecord,
    RunRecord,
    RunEventRecord,
    DocumentRecord,
    ChunkRecord,
    ChunkQuotaExceededError,
)

__all__ = [
    "get_db",
    "get_db_path",
    "apply_migrations",
    "ThreadRepo",
    "MessageRepo",
    "RunRepo",
    "EventRepo",
    "DocumentRepo",
    "ChunkRepo",
    "ThreadRecord",
    "MessageRecord",
    "RunRecord",
    "RunEventRecord",
    "DocumentRecord",
    "ChunkRecord",
    "ChunkQuotaExceededError",
]
