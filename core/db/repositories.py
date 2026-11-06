"""Repository abstractions and SQLite implementations for Research Agent Crew.

Provides domain repositories for Threads, Messages, Runs, and Run Events
with explicit interfaces, JSON serialization, and cascade deletion.
"""

import os
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
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

    async def update_thread_settings(self, thread_id: str, settings: Dict[str, Any]) -> Optional[ThreadRecord]:
        """Update settings_json dictionary for a thread."""
        now = utc_now_iso()
        current = await self.get_thread(thread_id)
        if not current:
            return None
        merged = {**current.settings, **settings}
        async with get_db(self.db_path) as db:
            await db.execute(
                "UPDATE threads SET settings_json = ?, updated_at = ? WHERE id = ?;",
                (json.dumps(merged), now, thread_id)
            )
            await db.commit()
        return await self.get_thread(thread_id)

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


MAX_CHUNKS_PER_THREAD = int(os.environ.get("MAX_CHUNKS_PER_THREAD", "5000"))


class ChunkQuotaExceededError(Exception):
    """Raised when inserting chunks would exceed the thread chunk limit."""
    pass


@dataclass
class DocumentRecord:
    id: str
    thread_id: str
    kind: str  # 'web' | 'upload'
    title: str
    url_or_name: str
    content_hash: str
    status: str = "indexed"  # 'pending' | 'fetching' | 'indexed' | 'failed'
    created_at: str = field(default_factory=utc_now_iso)
    run_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    chunk_count: int = 0
    is_duplicate: bool = False


@dataclass
class ChunkRecord:
    id: str
    document_id: str
    thread_id: str
    idx: int
    text: str
    token_count: int = 0
    embedding: Optional[bytes] = None
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class DocumentRepo:
    """Repository managing RAG Document entities."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    async def create_document(
        self,
        thread_id: str,
        kind: str,
        title: str,
        url_or_name: str,
        content_hash: str,
        run_id: Optional[str] = None,
        status: str = "indexed",
        meta: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> Tuple[DocumentRecord, bool]:
        """Insert or retrieve existing document on (thread_id, content_hash) conflict.

        Returns (document_record, is_duplicate).
        """
        id_str = doc_id or str(uuid.uuid4())
        created_at = utc_now_iso()
        meta_dict = meta or {}
        meta_json = json.dumps(meta_dict)

        async with get_db(self.db_path) as db:
            async with db.execute(
                """
                INSERT INTO documents (id, thread_id, run_id, kind, title, url_or_name, content_hash, status, created_at, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id, content_hash) DO NOTHING;
                """,
                (id_str, thread_id, run_id, kind, title, url_or_name, content_hash, status, created_at, meta_json)
            ) as cur:
                inserted = cur.rowcount > 0

            if inserted:
                await db.commit()
                return DocumentRecord(
                    id=id_str,
                    thread_id=thread_id,
                    run_id=run_id,
                    kind=kind,
                    title=title,
                    url_or_name=url_or_name,
                    content_hash=content_hash,
                    status=status,
                    created_at=created_at,
                    meta=meta_dict,
                    chunk_count=0,
                    is_duplicate=False
                ), False

            # Conflict: fetch existing
            async with db.execute(
                """
                SELECT d.*, COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.thread_id = ? AND d.content_hash = ?
                GROUP BY d.id;
                """,
                (thread_id, content_hash)
            ) as cur:
                row = await cur.fetchone()
                if not row:
                    raise RuntimeError("Failed to resolve conflicting document record")
                doc = DocumentRecord(
                    id=row["id"],
                    thread_id=row["thread_id"],
                    run_id=row["run_id"],
                    kind=row["kind"],
                    title=row["title"],
                    url_or_name=row["url_or_name"],
                    content_hash=row["content_hash"],
                    status=row["status"],
                    created_at=row["created_at"],
                    meta=json.loads(row["meta_json"] or "{}"),
                    chunk_count=row["chunk_count"],
                    is_duplicate=True
                )
                return doc, True

    async def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        async with get_db(self.db_path) as db:
            async with db.execute(
                """
                SELECT d.*, COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.id = ?
                GROUP BY d.id;
                """,
                (doc_id,)
            ) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                return DocumentRecord(
                    id=row["id"],
                    thread_id=row["thread_id"],
                    run_id=row["run_id"],
                    kind=row["kind"],
                    title=row["title"],
                    url_or_name=row["url_or_name"],
                    content_hash=row["content_hash"],
                    status=row["status"],
                    created_at=row["created_at"],
                    meta=json.loads(row["meta_json"] or "{}"),
                    chunk_count=row["chunk_count"]
                )

    async def get_document_by_hash(self, thread_id: str, content_hash: str) -> Optional[DocumentRecord]:
        async with get_db(self.db_path) as db:
            async with db.execute(
                """
                SELECT d.*, COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.thread_id = ? AND d.content_hash = ?
                GROUP BY d.id;
                """,
                (thread_id, content_hash)
            ) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                return DocumentRecord(
                    id=row["id"],
                    thread_id=row["thread_id"],
                    run_id=row["run_id"],
                    kind=row["kind"],
                    title=row["title"],
                    url_or_name=row["url_or_name"],
                    content_hash=row["content_hash"],
                    status=row["status"],
                    created_at=row["created_at"],
                    meta=json.loads(row["meta_json"] or "{}"),
                    chunk_count=row["chunk_count"]
                )

    async def list_documents_by_thread(self, thread_id: str) -> List[DocumentRecord]:
        async with get_db(self.db_path) as db:
            async with db.execute(
                """
                SELECT d.*, COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.thread_id = ?
                GROUP BY d.id
                ORDER BY d.created_at DESC;
                """,
                (thread_id,)
            ) as cur:
                rows = await cur.fetchall()
                return [
                    DocumentRecord(
                        id=r["id"],
                        thread_id=r["thread_id"],
                        run_id=r["run_id"],
                        kind=r["kind"],
                        title=r["title"],
                        url_or_name=r["url_or_name"],
                        content_hash=r["content_hash"],
                        status=r["status"],
                        created_at=r["created_at"],
                        meta=json.loads(r["meta_json"] or "{}"),
                        chunk_count=r["chunk_count"]
                    )
                    for r in rows
                ]

    async def update_document_status(
        self,
        doc_id: str,
        status: str,
        meta: Optional[Dict[str, Any]] = None
    ) -> bool:
        async with get_db(self.db_path) as db:
            if meta is not None:
                meta_json = json.dumps(meta)
                await db.execute(
                    "UPDATE documents SET status = ?, meta_json = ? WHERE id = ?;",
                    (status, meta_json, doc_id)
                )
            else:
                await db.execute(
                    "UPDATE documents SET status = ? WHERE id = ?;",
                    (status, doc_id)
                )
            await db.commit()
            return True

    async def delete_document(self, doc_id: str) -> bool:
        async with get_db(self.db_path) as db:
            async with db.execute("DELETE FROM documents WHERE id = ?;", (doc_id,)) as cur:
                deleted = cur.rowcount > 0
            await db.commit()
            return deleted


class ChunkRepo:
    """Repository managing RAG Chunk entities and vector persistence."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    async def count_chunks_by_thread(self, thread_id: str) -> int:
        async with get_db(self.db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM chunks WHERE thread_id = ?;",
                (thread_id,)
            ) as cur:
                row = await cur.fetchone()
                return row[0] if row else 0

    async def insert_chunks(self, thread_id: str, chunks: List[ChunkRecord]) -> int:
        """Insert a batch of chunks for a thread, validating thread chunk ceiling."""
        if not chunks:
            return 0

        max_allowed = int(os.environ.get("MAX_CHUNKS_PER_THREAD", "5000"))
        async with get_db(self.db_path) as db:
            async with db.execute(
                "SELECT COUNT(*) FROM chunks WHERE thread_id = ?;",
                (thread_id,)
            ) as cur:
                row = await cur.fetchone()
                current_count = row[0] if row else 0

            if current_count + len(chunks) > max_allowed:
                raise ChunkQuotaExceededError(
                    f"Thread {thread_id} chunk limit exceeded: {current_count} existing + {len(chunks)} new > limit of {max_allowed}"
                )

            rows = [
                (
                    c.id,
                    c.document_id,
                    c.thread_id,
                    c.idx,
                    c.text,
                    c.token_count,
                    c.embedding,
                    c.embedding_model,
                    c.embedding_dim,
                    json.dumps(c.meta or {})
                )
                for c in chunks
            ]

            await db.executemany(
                """
                INSERT INTO chunks (id, document_id, thread_id, idx, text, token_count, embedding, embedding_model, embedding_dim, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                rows
            )
            await db.commit()
            return len(chunks)

    async def list_chunks_by_thread(
        self,
        thread_id: str,
        embedding_model: Optional[str] = None
    ) -> List[ChunkRecord]:
        """Fetch chunks for thread, optionally filtering strictly by embedding_model."""
        async with get_db(self.db_path) as db:
            if embedding_model:
                query = "SELECT * FROM chunks WHERE thread_id = ? AND embedding_model = ? ORDER BY idx ASC;"
                params = (thread_id, embedding_model)
            else:
                query = "SELECT * FROM chunks WHERE thread_id = ? ORDER BY idx ASC;"
                params = (thread_id,)

            async with db.execute(query, params) as cur:
                rows = await cur.fetchall()
                return [
                    ChunkRecord(
                        id=r["id"],
                        document_id=r["document_id"],
                        thread_id=r["thread_id"],
                        idx=r["idx"],
                        text=r["text"],
                        token_count=r["token_count"],
                        embedding=r["embedding"],
                        embedding_model=r["embedding_model"],
                        embedding_dim=r["embedding_dim"],
                        meta=json.loads(r["meta_json"] or "{}")
                    )
                    for r in rows
                ]

    async def list_chunks_needing_reindex(
        self,
        thread_id: str,
        target_model: str
    ) -> List[ChunkRecord]:
        """Fetch chunks that were embedded with a different model or have no embedding."""
        async with get_db(self.db_path) as db:
            async with db.execute(
                """
                SELECT * FROM chunks
                WHERE thread_id = ? AND (embedding_model IS NULL OR embedding_model != ?)
                ORDER BY idx ASC;
                """,
                (thread_id, target_model)
            ) as cur:
                rows = await cur.fetchall()
                return [
                    ChunkRecord(
                        id=r["id"],
                        document_id=r["document_id"],
                        thread_id=r["thread_id"],
                        idx=r["idx"],
                        text=r["text"],
                        token_count=r["token_count"],
                        embedding=r["embedding"],
                        embedding_model=r["embedding_model"],
                        embedding_dim=r["embedding_dim"],
                        meta=json.loads(r["meta_json"] or "{}")
                    )
                    for r in rows
                ]

    async def update_chunk_embedding(
        self,
        chunk_id: str,
        embedding: bytes,
        embedding_model: str,
        embedding_dim: int
    ) -> bool:
        """Update the embedding vector and model metadata for an individual chunk."""
        async with get_db(self.db_path) as db:
            await db.execute(
                """
                UPDATE chunks
                SET embedding = ?, embedding_model = ?, embedding_dim = ?
                WHERE id = ?;
                """,
                (embedding, embedding_model, embedding_dim, chunk_id)
            )
            await db.commit()
            return True

    async def delete_chunks_by_document(self, document_id: str) -> int:
        async with get_db(self.db_path) as db:
            async with db.execute(
                "DELETE FROM chunks WHERE document_id = ?;",
                (document_id,)
            ) as cur:
                deleted = cur.rowcount
            await db.commit()
            return deleted


class ChatLockRepo:
    """Repository managing cross-process DB-backed chat locks with atomic acquisition and staleness checks."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    async def acquire_lock(
        self,
        thread_id: str,
        owner_id: str,
        stale_seconds: Optional[int] = None
    ) -> bool:
        """Atomically acquire the chat lock for thread_id.

        Succeeds if no lock exists, or if existing lock's heartbeat is older than
        HEARTBEAT_STALE_SECONDS, or if already owned by this owner_id.
        """
        now = utc_now_iso()
        stale_sec = stale_seconds if stale_seconds is not None else float(os.environ.get("HEARTBEAT_STALE_SECONDS", "60"))
        cutoff_dt = datetime.now(timezone.utc) - timedelta(seconds=stale_sec)
        cutoff_iso = cutoff_dt.isoformat()

        async with get_db(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO chat_locks (thread_id, owner_id, heartbeat_at)
                VALUES (?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE
                SET owner_id = excluded.owner_id,
                    heartbeat_at = excluded.heartbeat_at
                WHERE chat_locks.heartbeat_at < ? OR chat_locks.owner_id = excluded.owner_id;
                """,
                (thread_id, owner_id, now, cutoff_iso)
            )
            await db.commit()

            async with db.execute(
                "SELECT owner_id, heartbeat_at FROM chat_locks WHERE thread_id = ?;",
                (thread_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row and row["owner_id"] == owner_id:
                    return True
                return False

    async def refresh_heartbeat(self, thread_id: str, owner_id: str) -> bool:
        """Update heartbeat_at timestamp for a held lock."""
        now = utc_now_iso()
        async with get_db(self.db_path) as db:
            cursor = await db.execute(
                """
                UPDATE chat_locks
                SET heartbeat_at = ?
                WHERE thread_id = ? AND owner_id = ?;
                """,
                (now, thread_id, owner_id)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def release_lock(self, thread_id: str, owner_id: str) -> bool:
        """Release the chat lock if owned by owner_id."""
        async with get_db(self.db_path) as db:
            cursor = await db.execute(
                """
                DELETE FROM chat_locks
                WHERE thread_id = ? AND owner_id = ?;
                """,
                (thread_id, owner_id)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def get_active_lock(
        self,
        thread_id: str,
        stale_seconds: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Return lock dict if lock is currently active and not stale, else None."""
        stale_sec = stale_seconds if stale_seconds is not None else float(os.environ.get("HEARTBEAT_STALE_SECONDS", "60"))
        cutoff_dt = datetime.now(timezone.utc) - timedelta(seconds=stale_sec)
        cutoff_iso = cutoff_dt.isoformat()

        async with get_db(self.db_path) as db:
            async with db.execute(
                """
                SELECT thread_id, owner_id, heartbeat_at
                FROM chat_locks
                WHERE thread_id = ? AND heartbeat_at >= ?;
                """,
                (thread_id, cutoff_iso)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return {
                        "thread_id": row["thread_id"],
                        "owner_id": row["owner_id"],
                        "heartbeat_at": row["heartbeat_at"]
                    }
                return None
