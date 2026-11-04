"""Repository abstractions and SQLite implementations for Research Agent Crew.

Provides domain repositories for Threads, Messages, Runs, and Run Events
with explicit interfaces, JSON serialization, and cascade deletion.
"""

import os
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from core.db.connection import get_db
from core.checkpoint import delete_checkpoints_for_thread


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ThreadRecord:
    id: str
    title: str
    created_at: str
    updated_at: str
    archived: bool
    settings: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None


@dataclass
class MessageRecord:
    id: str
    thread_id: str
    role: str
    content: str
    created_at: str
    run_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    id: str
    thread_id: str
    task: str
    status: str
    started_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None
    resume_after: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    report_md: Optional[str] = None
    report_json: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False
    owner_id: Optional[str] = None
    heartbeat_at: Optional[str] = None


@dataclass
class RunEventRecord:
    seq: int
    run_id: str
    ts: str
    type: str
    node: Optional[str] = None
    subtask_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def data(self) -> Dict[str, Any]:
        return self.payload


class ThreadRepo:
    """Repository managing Thread persistence and lifecycle."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    @staticmethod
    def auto_title(prompt: str) -> str:
        """Deterministically derive a clean title from a prompt without an LLM call.

        Extracts the first line and truncates to a maximum of 50 characters,
        breaking cleanly at word boundaries when possible.
        """
        first_line = prompt.strip().splitlines()[0].strip() if prompt else "New Thread"
        if len(first_line) <= 50:
            return first_line or "New Thread"

        cut = first_line[:50]
        last_space = cut.rfind(" ")
        if last_space > 20:
            return cut[:last_space].rstrip(".,;:-_ ") + "..."
        return cut[:47].rstrip() + "..."

    async def create_thread(
        self,
        title: Optional[str] = None,
        prompt: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None
    ) -> ThreadRecord:
        """Create a new thread with optional auto-titling."""
        final_id = thread_id or str(uuid.uuid4())
        final_title = title or (self.auto_title(prompt) if prompt else "New Thread")
        now = utc_now_iso()
        settings_dict = settings or {}
        settings_json = json.dumps(settings_dict)

        async with get_db(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO threads (id, title, created_at, updated_at, archived, settings_json, user_id)
                VALUES (?, ?, ?, ?, 0, ?, ?);
                """,
                (final_id, final_title, now, now, settings_json, user_id)
            )
            await db.commit()

        return ThreadRecord(
            id=final_id,
            title=final_title,
            created_at=now,
            updated_at=now,
            archived=False,
            settings=settings_dict,
            user_id=user_id
        )

    async def get_thread(self, thread_id: str) -> Optional[ThreadRecord]:
        """Fetch a single thread by ID."""
        async with get_db(self.db_path) as db:
            async with db.execute("SELECT * FROM threads WHERE id = ?;", (thread_id,)) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                return ThreadRecord(
                    id=row["id"],
                    title=row["title"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    archived=bool(row["archived"]),
                    settings=json.loads(row["settings_json"] or "{}"),
                    user_id=row["user_id"]
                )

    async def list_threads(self, include_archived: bool = False) -> List[ThreadRecord]:
        """List threads sorted by most recently updated."""
        query = "SELECT * FROM threads"
        params = []
        if not include_archived:
            query += " WHERE archived = 0"
        query += " ORDER BY updated_at DESC;"

        async with get_db(self.db_path) as db:
            async with db.execute(query, params) as cur:
                rows = await cur.fetchall()
                return [
                    ThreadRecord(
                        id=r["id"],
                        title=r["title"],
                        created_at=r["created_at"],
                        updated_at=r["updated_at"],
                        archived=bool(r["archived"]),
                        settings=json.loads(r["settings_json"] or "{}"),
                        user_id=r["user_id"]
                    )
                    for r in rows
                ]

    async def rename_thread(self, thread_id: str, new_title: str) -> Optional[ThreadRecord]:
        """Rename an existing thread."""
        now = utc_now_iso()
        async with get_db(self.db_path) as db:
            await db.execute(
                "UPDATE threads SET title = ?, updated_at = ? WHERE id = ?;",
                (new_title.strip(), now, thread_id)
            )
            await db.commit()
        return await self.get_thread(thread_id)

    async def archive_thread(self, thread_id: str, archived: bool = True) -> Optional[ThreadRecord]:
        """Set thread archived status."""
        now = utc_now_iso()
        async with get_db(self.db_path) as db:
            await db.execute(
                "UPDATE threads SET archived = ?, updated_at = ? WHERE id = ?;",
                (1 if archived else 0, now, thread_id)
            )
            await db.commit()
        return await self.get_thread(thread_id)

    async def touch_thread(self, thread_id: str) -> None:
        """Update the thread updated_at timestamp."""
        now = utc_now_iso()
        async with get_db(self.db_path) as db:
            await db.execute("UPDATE threads SET updated_at = ? WHERE id = ?;", (now, thread_id))
            await db.commit()

    async def delete_thread(self, thread_id: str) -> bool:
        """Delete thread and cascade delete all messages, runs, events, and checkpoints."""
        run_repo = RunRepo(db_path=self.db_path)
        runs = await run_repo.list_runs_by_thread(thread_id)
        run_ids = [r.id for r in runs]

        async with get_db(self.db_path) as db:
            async with db.execute("SELECT id FROM threads WHERE id = ?;", (thread_id,)) as cur:
                if not await cur.fetchone():
                    return False
            # Foreign keys cascade to runs, messages, run_events
            await db.execute("DELETE FROM threads WHERE id = ?;", (thread_id,))
            await db.commit()

        # Delete checkpoints in SQLite checkpointer DB via AsyncSqliteSaver.adelete_thread
        await delete_checkpoints_for_thread(thread_id=thread_id, run_ids=run_ids)
        return True


class MessageRepo:
    """Repository managing Message persistence inside threads."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    async def create_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MessageRecord:
        """Append a message to a thread."""
        msg_id = str(uuid.uuid4())
        now = utc_now_iso()
        meta_dict = metadata or {}
        meta_json = json.dumps(meta_dict)

        async with get_db(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO messages (id, thread_id, role, content, created_at, run_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (msg_id, thread_id, role, content, now, run_id, meta_json)
            )
            await db.execute("UPDATE threads SET updated_at = ? WHERE id = ?;", (now, thread_id))
            await db.commit()

        return MessageRecord(
            id=msg_id,
            thread_id=thread_id,
            role=role,
            content=content,
            created_at=now,
            run_id=run_id,
            metadata=meta_dict
        )

    async def list_messages(self, thread_id: str) -> List[MessageRecord]:
        """List messages in a thread in chronological order."""
        async with get_db(self.db_path) as db:
            async with db.execute(
                "SELECT * FROM messages WHERE thread_id = ? ORDER BY created_at ASC;",
                (thread_id,)
            ) as cur:
                rows = await cur.fetchall()
                return [
                    MessageRecord(
                        id=r["id"],
                        thread_id=r["thread_id"],
                        role=r["role"],
                        content=r["content"],
                        created_at=r["created_at"],
                        run_id=r["run_id"],
                        metadata=json.loads(r["metadata_json"] or "{}")
                    )
                    for r in rows
                ]


class RunRepo:
    """Repository managing Run lifecycle, heartbeats, and status transitions."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    async def create_run(
        self,
        thread_id: str,
        task: str,
        config: Optional[Dict[str, Any]] = None,
        owner_id: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> RunRecord:
        """Create a new run record with pending/running status."""
        final_id = run_id or str(uuid.uuid4())
        now = utc_now_iso()
        cfg_dict = config or {}
        cfg_json = json.dumps(cfg_dict)

        async with get_db(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO runs (
                    id, thread_id, task, status, started_at, config_json,
                    cancel_requested, owner_id, heartbeat_at
                )
                VALUES (?, ?, ?, 'pending', ?, ?, 0, ?, ?);
                """,
                (final_id, thread_id, task, now, cfg_json, owner_id, now)
            )
            await db.execute("UPDATE threads SET updated_at = ? WHERE id = ?;", (now, thread_id))
            await db.commit()

        return RunRecord(
            id=final_id,
            thread_id=thread_id,
            task=task,
            status="pending",
            started_at=now,
            config=cfg_dict,
            cancel_requested=False,
            owner_id=owner_id,
            heartbeat_at=now
        )

    async def get_run(self, run_id: str) -> Optional[RunRecord]:
        """Fetch a single run by ID."""
        async with get_db(self.db_path) as db:
            async with db.execute("SELECT * FROM runs WHERE id = ?;", (run_id,)) as cur:
                r = await cur.fetchone()
                if not r:
                    return None
                return RunRecord(
                    id=r["id"],
                    thread_id=r["thread_id"],
                    task=r["task"],
                    status=r["status"],
                    started_at=r["started_at"],
                    finished_at=r["finished_at"],
                    error=r["error"],
                    resume_after=r["resume_after"],
                    config=json.loads(r["config_json"] or "{}"),
                    report_md=r["report_md"],
                    report_json=r["report_json"],
                    metrics=json.loads(r["metrics_json"] or "{}"),
                    cancel_requested=bool(r["cancel_requested"]),
                    owner_id=r["owner_id"],
                    heartbeat_at=r["heartbeat_at"]
                )

    async def list_runs_by_thread(self, thread_id: str) -> List[RunRecord]:
        """List all runs for a thread in chronological order."""
        async with get_db(self.db_path) as db:
            async with db.execute(
                "SELECT * FROM runs WHERE thread_id = ? ORDER BY started_at ASC;",
                (thread_id,)
            ) as cur:
                rows = await cur.fetchall()
                return [
                    RunRecord(
                        id=r["id"],
                        thread_id=r["thread_id"],
                        task=r["task"],
                        status=r["status"],
                        started_at=r["started_at"],
                        finished_at=r["finished_at"],
                        error=r["error"],
                        resume_after=r["resume_after"],
                        config=json.loads(r["config_json"] or "{}"),
                        report_md=r["report_md"],
                        report_json=r["report_json"],
                        metrics=json.loads(r["metrics_json"] or "{}"),
                        cancel_requested=bool(r["cancel_requested"]),
                        owner_id=r["owner_id"],
                        heartbeat_at=r["heartbeat_at"]
                    )
                    for r in rows
                ]

    async def update_run_status(
        self,
        run_id: str,
        status: str,
        error: Optional[str] = None,
        resume_after: Optional[str] = None,
        finished: bool = False
    ) -> None:
        """Update run status and optional error/resume/completion timestamps."""
        now = utc_now_iso()
        finished_val = now if finished else None
        async with get_db(self.db_path) as db:
            if finished:
                await db.execute(
                    """
                    UPDATE runs
                    SET status = ?, error = ?, resume_after = ?, finished_at = ?
                    WHERE id = ?;
                    """,
                    (status, error, resume_after, finished_val, run_id)
                )
            else:
                await db.execute(
                    """
                    UPDATE runs
                    SET status = ?, error = ?, resume_after = ?
                    WHERE id = ?;
                    """,
                    (status, error, resume_after, run_id)
                )
            await db.commit()

    async def update_heartbeat(self, run_id: str, owner_id: Optional[str] = None) -> None:
        """Update heartbeat timestamp and owner ID for active runs."""
        now = utc_now_iso()
        async with get_db(self.db_path) as db:
            if owner_id:
                await db.execute(
                    "UPDATE runs SET heartbeat_at = ?, owner_id = ? WHERE id = ?;",
                    (now, owner_id, run_id)
                )
            else:
                await db.execute(
                    "UPDATE runs SET heartbeat_at = ? WHERE id = ?;",
                    (now, run_id)
                )
            await db.commit()

    async def request_cancel(self, run_id: str) -> None:
        """Flag run for cooperative cancellation."""
        async with get_db(self.db_path) as db:
            await db.execute(
                "UPDATE runs SET cancel_requested = 1 WHERE id = ?;",
                (run_id,)
            )
            await db.commit()

    async def is_cancel_requested(self, run_id: str) -> bool:
        """Check whether cancellation was requested for a run."""
        async with get_db(self.db_path) as db:
            async with db.execute(
                "SELECT cancel_requested FROM runs WHERE id = ?;",
                (run_id,)
            ) as cur:
                row = await cur.fetchone()
                return bool(row["cancel_requested"]) if row else False

    async def save_run_result(
        self,
        run_id: str,
        status: str,
        report_md: Optional[str] = None,
        report_json: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Persist final report and metrics upon completion or escalation."""
        now = utc_now_iso()
        metrics_json = json.dumps(metrics or {})
        async with get_db(self.db_path) as db:
            await db.execute(
                """
                UPDATE runs
                SET status = ?, report_md = ?, report_json = ?, metrics_json = ?, finished_at = ?
                WHERE id = ?;
                """,
                (status, report_md, report_json, metrics_json, now, run_id)
            )
            await db.commit()

    async def recover_interrupted_runs(self, cutoff_seconds: Optional[int] = None) -> int:
        """At startup, mark orphaned running runs as failed/interrupted.

        Never touches runs that have a fresh heartbeat within cutoff_seconds.
        Defaults cutoff_seconds to env HEARTBEAT_STALE_SECONDS or 60.
        """
        if cutoff_seconds is None:
            try:
                cutoff_seconds = int(os.environ.get("HEARTBEAT_STALE_SECONDS", "60"))
            except ValueError:
                cutoff_seconds = 60

        now = datetime.now(timezone.utc)
        cutoff_time = (now - timedelta(seconds=cutoff_seconds)).isoformat()
        now_str = now.isoformat()

        async with get_db(self.db_path) as db:
            # Running runs with no heartbeat or heartbeat older than cutoff
            async with db.execute(
                """
                UPDATE runs
                SET status = 'failed', error = 'interrupted', finished_at = ?
                WHERE status = 'running'
                  AND (heartbeat_at IS NULL OR heartbeat_at < ?);
                """,
                (now_str, cutoff_time)
            ) as cur:
                recovered_count = cur.rowcount
            await db.commit()
            return max(0, recovered_count)


class EventRepo:
    """Append-only repository for structured Run Events."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    async def append_event(
        self,
        run_id: str,
        ts: str,
        event_type: str,
        node: Optional[str] = None,
        subtask_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None
    ) -> RunEventRecord:
        """Append a single telemetry event."""
        payload_dict = payload or {}
        payload_json = json.dumps(payload_dict)

        async with get_db(self.db_path) as db:
            async with db.execute(
                """
                INSERT INTO run_events (run_id, ts, type, node, subtask_id, payload_json)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (run_id, ts, event_type, node, subtask_id, payload_json)
            ) as cur:
                seq = cur.lastrowid
            await db.commit()

        return RunEventRecord(
            seq=seq,
            run_id=run_id,
            ts=ts,
            type=event_type,
            node=node,
            subtask_id=subtask_id,
            payload=payload_dict
        )

    async def list_events_by_run(self, run_id: str) -> List[RunEventRecord]:
        """Fetch all events for a run ordered by sequence."""
        async with get_db(self.db_path) as db:
            async with db.execute(
                "SELECT * FROM run_events WHERE run_id = ? ORDER BY seq ASC;",
                (run_id,)
            ) as cur:
                rows = await cur.fetchall()
                return [
                    RunEventRecord(
                        seq=r["seq"],
                        run_id=r["run_id"],
                        ts=r["ts"],
                        type=r["type"],
                        node=r["node"],
                        subtask_id=r["subtask_id"] if "subtask_id" in r.keys() else None,
                        payload=json.loads(r["payload_json"] or "{}")
                    )
                    for r in rows
                ]

    async def list_events_after(self, run_id: str, after_seq: int = 0) -> List[RunEventRecord]:
        """Fetch all events for a run with sequence number strictly greater than after_seq."""
        async with get_db(self.db_path) as db:
            async with db.execute(
                "SELECT * FROM run_events WHERE run_id = ? AND seq > ? ORDER BY seq ASC;",
                (run_id, after_seq)
            ) as cur:
                rows = await cur.fetchall()
                return [
                    RunEventRecord(
                        seq=r["seq"],
                        run_id=r["run_id"],
                        ts=r["ts"],
                        type=r["type"],
                        node=r["node"],
                        subtask_id=r["subtask_id"] if "subtask_id" in r.keys() else None,
                        payload=json.loads(r["payload_json"] or "{}")
                    )
                    for r in rows
                ]

    async def count_events_by_run(self, run_id: str) -> int:
        """Count total events stored for a run."""
        async with get_db(self.db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM run_events WHERE run_id = ?;",
                (run_id,)
            ) as cur:
                row = await cur.fetchone()
                return row[0] if row else 0
