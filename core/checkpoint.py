"""LangGraph SQLite Checkpointer with custom serialization allowlists.

Integrates AsyncSqliteSaver from langgraph-checkpoint-sqlite with strict
serialization support for all AgentState Pydantic models.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, List, Tuple
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# Explicit registration of all AgentState Pydantic models for safe msgpack round-tripping
STATE_MSG_PACK_MODULES: List[Tuple[str, str]] = [
    ("core.state", "Claim"),
    ("core.state", "Section"),
    ("core.state", "ComparisonRow"),
    ("core.state", "StructuredReport"),
    ("core.state", "Subtask"),
    ("core.state", "ResearchPlan"),
    ("core.state", "SourceItem"),
    ("core.state", "SubtaskResult"),
    ("core.state", "CriticRubricScore"),
    ("core.state", "CriticEvaluation"),
    ("core.state", "TraceEvent"),
]


def get_state_serializer() -> JsonPlusSerializer:
    """Return JsonPlusSerializer pre-configured with allowed core.state models."""
    return JsonPlusSerializer(allowed_msgpack_modules=STATE_MSG_PACK_MODULES)


def get_checkpoint_db_path() -> str:
    """Return configured checkpointer database path from environment or default."""
    path = os.environ.get("CHECKPOINT_DB_PATH", "data/checkpoints.db")
    abs_dir = os.path.dirname(os.path.abspath(path))
    if abs_dir and not os.path.exists(abs_dir):
        os.makedirs(abs_dir, exist_ok=True)
    return path


@asynccontextmanager
async def get_checkpointer(db_path: Optional[str] = None) -> AsyncIterator[AsyncSqliteSaver]:
    """Async context manager yielding a setup AsyncSqliteSaver instance."""
    target_path = db_path or get_checkpoint_db_path()
    serde = get_state_serializer()
    async with aiosqlite.connect(target_path) as conn:
        saver = AsyncSqliteSaver(conn, serde=serde)
        await saver.setup()
        yield saver


async def delete_checkpoints_for_thread(
    thread_id: str,
    run_ids: Optional[List[str]] = None,
    db_path: Optional[str] = None
) -> None:
    """Delete all checkpoint records and writes for a given thread and its runs via AsyncSqliteSaver.adelete_thread."""
    target_path = db_path or get_checkpoint_db_path()
    if not os.path.exists(target_path):
        return

    async with get_checkpointer(target_path) as saver:
        keys_to_delete = set()
        if run_ids:
            for r_id in run_ids:
                keys_to_delete.add(f"{thread_id}:{r_id}")
        keys_to_delete.add(str(thread_id))

        # Also discover any existing keys in checkpoints or writes starting with thread_id
        async with saver.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints';") as cur:
            if await cur.fetchone():
                async with saver.conn.execute(
                    "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id = ? OR thread_id LIKE ?;",
                    (str(thread_id), f"{thread_id}:%")
                ) as cur:
                    for row in await cur.fetchall():
                        keys_to_delete.add(row[0])
                async with saver.conn.execute(
                    "SELECT DISTINCT thread_id FROM writes WHERE thread_id = ? OR thread_id LIKE ?;",
                    (str(thread_id), f"{thread_id}:%")
                ) as cur:
                    for row in await cur.fetchall():
                        keys_to_delete.add(row[0])

        for key in keys_to_delete:
            await saver.adelete_thread(key)
