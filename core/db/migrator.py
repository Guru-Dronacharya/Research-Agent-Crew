"""Versioned SQLite schema migration runner for Research Agent Crew.

Applies ordered SQL migration scripts from core/db/migrations/ under
BEGIN IMMEDIATE locks, tracking current schema version via PRAGMA user_version.
Ensures concurrent startup safety and idempotent execution.
"""

import re
from pathlib import Path
from typing import List, Tuple, Optional
import aiosqlite

from core.db.connection import get_db, get_db_path

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def get_migration_files() -> List[Tuple[int, Path]]:
    """Discover and sort available migration files by version integer."""
    if not MIGRATIONS_DIR.is_dir():
        return []

    migrations: List[Tuple[int, Path]] = []
    pattern = re.compile(r"^(\d+)_.*\.sql$")
    for file_path in MIGRATIONS_DIR.iterdir():
        if file_path.is_file():
            match = pattern.match(file_path.name)
            if match:
                version = int(match.group(1))
                migrations.append((version, file_path))

    migrations.sort(key=lambda x: x[0])
    return migrations


async def get_current_user_version(conn: aiosqlite.Connection) -> int:
    """Read the current schema version from SQLite PRAGMA user_version."""
    async with conn.execute("PRAGMA user_version;") as cursor:
        row = await cursor.fetchone()
        return row[0] if row else 0


async def apply_migrations(db_path: Optional[str] = None) -> int:
    """Apply all pending migrations under BEGIN IMMEDIATE lock.

    Returns the latest applied schema version.
    """
    target_path = db_path or get_db_path()
    migrations = get_migration_files()
    if not migrations:
        return 0

    max_target_version = migrations[-1][0]

    async with get_db(target_path) as conn:
        current_version = await get_current_user_version(conn)
        if current_version >= max_target_version:
            return current_version

        # Acquire reserved lock so concurrent processes wait
        await conn.execute("BEGIN IMMEDIATE;")
        try:
            # Re-read version inside transaction
            current_version = await get_current_user_version(conn)
            for version, file_path in migrations:
                if version > current_version:
                    sql_content = file_path.read_text(encoding="utf-8")
                    await conn.executescript(sql_content)
                    await conn.execute(f"PRAGMA user_version = {version};")
                    current_version = version

            await conn.commit()
            return current_version
        except Exception:
            await conn.rollback()
            raise
