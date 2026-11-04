"""SQLite database connection manager for Research Agent Crew.

Provides an async context manager ensuring WAL mode, foreign keys enforcement,
and busy timeout handling across all database operations.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
import aiosqlite


def get_db_path() -> str:
    """Return configured database path from environment or default."""
    path = os.environ.get("DATABASE_PATH", "data/app.db")
    abs_dir = os.path.dirname(os.path.abspath(path))
    if abs_dir and not os.path.exists(abs_dir):
        os.makedirs(abs_dir, exist_ok=True)
    return path


@asynccontextmanager
async def get_db(db_path: Optional[str] = None) -> AsyncIterator[aiosqlite.Connection]:
    """Async context manager yielding a configured SQLite connection.

    Enforces foreign keys ON and WAL journal mode on every connection.
    """
    target_path = db_path or get_db_path()
    abs_dir = os.path.dirname(os.path.abspath(target_path))
    if abs_dir and not os.path.exists(abs_dir):
        os.makedirs(abs_dir, exist_ok=True)

    conn = await aiosqlite.connect(target_path)
    conn.row_factory = aiosqlite.Row
    try:
        await conn.execute("PRAGMA foreign_keys = ON;")
        await conn.execute("PRAGMA journal_mode = WAL;")
        await conn.execute("PRAGMA busy_timeout = 5000;")
        yield conn
    finally:
        await conn.close()
