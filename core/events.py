"""Event streaming, Pydantic event models, in-process event bus, and database tailing.

Provides real-time event emission for multi-agent runs, single-writer SQLite persistence,
payload capping, replay-then-live event streams, and cross-process database tailing.
"""

import os
import re
import json
import time
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, AsyncIterator, Set
from pydantic import BaseModel, Field, field_validator

from core.db.repositories import EventRepo, RunRepo


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


EVENT_PAYLOAD_MAX_BYTES = int(os.environ.get("EVENT_PAYLOAD_MAX_BYTES", "8192"))

VALID_EVENT_TYPES = {
    "run_started",
    "node_started",
    "node_finished",
    "tool_call",
    "tool_result",
    "source_found",
    "budget_alert",
    "quota_paused",
    "critic_verdict",
    "report_section",
    "run_finished",
    "error",
    "heartbeat",
    "stale",
    "decomposition",
    "rubric_evaluation",
    "escalation",
    "document_fetched",
    "chunks_indexed"
}

SENSITIVE_KEY_SUBSTRINGS = ("api_key", "secret", "password", "token", "authorization")

GROQ_KEY_PATTERN = re.compile(r"gsk_[a-zA-Z0-9]{20,}")
GOOGLE_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z-_]{35}")


def redact_sensitive_strings(text: str) -> str:
    """Redact known API key patterns (Groq gsk_..., Google AIza...) from text."""
    if not isinstance(text, str):
        return text
    text = GROQ_KEY_PATTERN.sub("[REDACTED]", text)
    text = GOOGLE_KEY_PATTERN.sub("[REDACTED]", text)
    return text


def sanitize_and_cap_payload(data: Dict[str, Any], max_bytes: int = EVENT_PAYLOAD_MAX_BYTES) -> Dict[str, Any]:
    """Sanitize sensitive keys and cap payload size to avoid memory bloat.

    Truncates long string values if the total payload serialization exceeds max_bytes.
    """
    if not isinstance(data, dict):
        return {}

    cleaned: Dict[str, Any] = {}
    for k, v in data.items():
        k_lower = str(k).lower()
        if any(sens in k_lower for sens in SENSITIVE_KEY_SUBSTRINGS):
            cleaned[k] = "[REDACTED]"
            continue

        if isinstance(v, str):
            v_red = redact_sensitive_strings(v)
            if len(v_red) > 2000:
                cleaned[k] = v_red[:2000] + "... [truncated]"
            else:
                cleaned[k] = v_red
        elif isinstance(v, dict):
            cleaned[k] = sanitize_and_cap_payload(v, max_bytes=max_bytes // 2)
        elif isinstance(v, list):
            cleaned[k] = [
                sanitize_and_cap_payload(item, max_bytes=1000) if isinstance(item, dict)
                else (redact_sensitive_strings(item)[:500] + "... [truncated]" if isinstance(item, str) and len(item) > 500 else redact_sensitive_strings(item) if isinstance(item, str) else item)
                for item in v[:50]
            ]
        else:
            cleaned[k] = v

    # If still exceeds max_bytes, aggressively truncate strings
    try:
        dumped = json.dumps(cleaned)
        if len(dumped.encode("utf-8")) > max_bytes:
            compact: Dict[str, Any] = {}
            for k, v in cleaned.items():
                if isinstance(v, str) and len(v) > 300:
                    compact[k] = v[:300] + "... [truncated]"
                else:
                    compact[k] = v
            return compact
    except Exception:
        pass

    return cleaned


class RunEvent(BaseModel):
    """Pydantic model representing a granular telemetry or progress event."""
    seq: Optional[int] = Field(default=None, description="Monotonic sequence number assigned upon persistence")
    run_id: str = Field(..., description="ID of the associated run")
    ts: str = Field(default_factory=utc_now_iso, description="ISO-8601 UTC timestamp")
    type: str = Field(..., description="Event type identifier")
    node: Optional[str] = Field(default=None, description="Graph node name if emitted from a node")
    subtask_id: Optional[str] = Field(default=None, description="Subtask identifier if associated with a subtask")
    data: Dict[str, Any] = Field(default_factory=dict, description="Structured event payload")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in VALID_EVENT_TYPES:
            raise ValueError(f"Invalid event type '{v}'. Must be one of {sorted(VALID_EVENT_TYPES)}")
        return v

    @property
    def payload(self) -> Dict[str, Any]:
        """Backward compatibility alias for data dictionary."""
        return self.data


class SubscriberQueueLaggedError(Exception):
    """Raised when an in-process subscriber queue overflows and must re-sync from DB."""
    pass


class SubscriberQueue:
    """Wrapper around an asyncio.Queue with overflow tracking."""

    def __init__(self, maxsize: int = 100):
        self.queue: asyncio.Queue[RunEvent] = asyncio.Queue(maxsize=maxsize)
        self.is_lagged: bool = False
        self.is_closed: bool = False

    def put_nowait(self, item: RunEvent) -> bool:
        """Attempt non-blocking put; if full, mark lagged and close queue."""
        if self.is_closed or self.is_lagged:
            return False
        try:
            self.queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            self.is_lagged = True
            self.is_closed = True
            return False

    def close(self):
        self.is_closed = True


class RunEventBus:
    """In-process event bus for a single active run.

    Provides:
    - Asynchronous event emission into an in-memory queue (nodes never wait on SQLite).
    - A dedicated persistence writer task that serializes writes to run_events table,
      guaranteeing seq order equals emission order.
    - Guarantees flush on node finish, run finish, and crash.
    - Fan-out to in-process subscribers with bounded queues and overflow policy.
    - In-process wake signal for DB-tailing subscribers.
    """

    def __init__(self, run_id: str, db_path: Optional[str] = None):
        self.run_id = run_id
        self.db_path = db_path
        self.event_repo = EventRepo(db_path=db_path)
        self._queue: asyncio.Queue[RunEvent] = asyncio.Queue()
        self._subscribers: Set[SubscriberQueue] = set()
        self._wake_event: asyncio.Event = asyncio.Event()
        self._running: bool = True
        self._writer_task: Optional[asyncio.Task] = None
        self._write_lock: asyncio.Lock = asyncio.Lock()

    def start(self):
        """Start the background writer task if not already started."""
        if self._writer_task is None or self._writer_task.done():
            self._writer_task = asyncio.create_task(self._persistence_writer())

    async def emit(
        self,
        event_type: str,
        node: Optional[str] = None,
        subtask_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        ts: Optional[str] = None
    ) -> RunEvent:
        """Emit an event onto the bus without waiting on disk I/O.

        Guarantees event is queued in emission order.
        """
        self.start()
        clean_data = sanitize_and_cap_payload(data or {})
        event = RunEvent(
            run_id=self.run_id,
            ts=ts or utc_now_iso(),
            type=event_type,
            node=node,
            subtask_id=subtask_id,
            data=clean_data
        )
        await self._queue.put(event)
        return event

    async def emit_event(self, event: RunEvent) -> RunEvent:
        """Directly queue a pre-constructed RunEvent."""
        self.start()
        event.data = sanitize_and_cap_payload(event.data)
        await self._queue.put(event)
        return event

    async def _persistence_writer(self):
        """Single-writer background loop persisting queued events to SQLite in order."""
        while self._running or not self._queue.empty():
            try:
                try:
                    event = await asyncio.wait_for(self._queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue

                async with self._write_lock:
                    if event.type != "heartbeat":
                        # Persist to SQLite
                        persisted = await self.event_repo.append_event(
                            run_id=self.run_id,
                            ts=event.ts,
                            event_type=event.type,
                            node=event.node,
                            subtask_id=event.subtask_id,
                            payload=event.data
                        )
                        event.seq = persisted.seq

                    # Wake local DB-tailing subscribers
                    self._wake_event.set()
                    self._wake_event.clear()

                    # Fan out to in-process subscriber queues
                    lagged_subs: List[SubscriberQueue] = []
                    for sub in list(self._subscribers):
                        ok = sub.put_nowait(event)
                        if not ok and sub.is_lagged:
                            lagged_subs.append(sub)

                    for sub in lagged_subs:
                        self._subscribers.discard(sub)

                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log writer error but continue writing
                print(f"[RunEventBus Writer Error for {self.run_id}] {e}", flush=True)

    async def flush(self, timeout: float = 10.0):
        """Wait until all currently queued events are fully written to SQLite."""
        self.start()
        start = time.perf_counter()
        while not self._queue.empty():
            if time.perf_counter() - start > timeout:
                break
            await asyncio.sleep(0.01)
        # Ensure the current DB transaction finishes
        async with self._write_lock:
            pass

    async def close(self):
        """Flush remaining events, stop the writer task, and cleanup subscribers."""
        await self.flush(timeout=5.0)
        self._running = False
        if self._writer_task and not self._writer_task.done():
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass
        self._subscribers.clear()
        remove_run_event_bus(self.run_id)

    def subscribe_queue(self, maxsize: int = 100) -> SubscriberQueue:
        """Register an in-process subscriber queue with bounded capacity."""
        sub = SubscriberQueue(maxsize=maxsize)
        self._subscribers.add(sub)
        return sub

    def unsubscribe_queue(self, sub: SubscriberQueue):
        """Unregister an in-process subscriber queue."""
        self._subscribers.discard(sub)


# Global registry of active buses per run
_RUN_BUSES: Dict[str, RunEventBus] = {}
_BUSES_LOCK = asyncio.Lock()


def get_run_event_bus(run_id: str, db_path: Optional[str] = None) -> RunEventBus:
    """Retrieve or instantiate singleton event bus for a run."""
    if run_id not in _RUN_BUSES:
        bus = RunEventBus(run_id=run_id, db_path=db_path)
        _RUN_BUSES[run_id] = bus
        bus.start()
    return _RUN_BUSES[run_id]


def remove_run_event_bus(run_id: str) -> Optional[RunEventBus]:
    """Remove run event bus from registry."""
    return _RUN_BUSES.pop(run_id, None)


async def emit_run_event(
    run_id: Optional[str],
    event_type: str,
    node: Optional[str] = None,
    subtask_id: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None
) -> Optional[RunEvent]:
    """Convenience helper to emit an event to the run's bus if run_id is valid."""
    if not run_id:
        return None
    bus = get_run_event_bus(run_id, db_path=db_path)
    return await bus.emit(
        event_type=event_type,
        node=node,
        subtask_id=subtask_id,
        data=data or {}
    )


async def tail_run_events(
    run_id: str,
    after_seq: int = 0,
    db_path: Optional[str] = None,
    poll_interval: float = 0.5,
    client_disconnect_fn: Optional[Any] = None,
    emit_heartbeat_interval: float = 15.0
) -> AsyncIterator[RunEvent]:
    """Asynchronously tail events for a run directly from SQLite (the source of truth).

    Replays all persisted events where seq > after_seq, then follows new events in
    real time. If the run is in the same process, wakes immediately via bus notification;
    if in another process, polls every poll_interval seconds.

    Terminates cleanly after:
    - run_finished
    - error
    - quota_paused
    - already finished run (after replaying remaining events)
    - stale heartbeat detection (emits 'stale' event and closes)
    - client disconnect
    """
    event_repo = EventRepo(db_path=db_path)
    run_repo = RunRepo(db_path=db_path)
    current_seq = after_seq

    bus = _RUN_BUSES.get(run_id)
    last_heartbeat_time = time.perf_counter()

    while True:
        # 1. Check if client disconnected
        if client_disconnect_fn:
            try:
                if asyncio.iscoroutinefunction(client_disconnect_fn):
                    if await client_disconnect_fn():
                        break
                elif client_disconnect_fn():
                    break
            except Exception:
                pass

        # 2. Fetch all new events from SQLite where seq > current_seq
        new_events = await event_repo.list_events_after(run_id=run_id, after_seq=current_seq)
        for ev in new_events:
            current_seq = ev.seq
            yield ev
            if ev.type in ("run_finished", "error", "quota_paused"):
                return

        # 3. Check run status from DB
        run = await run_repo.get_run(run_id)
        if not run:
            # Run does not exist
            return

        # Terminal run check: if run is already finished, ensure no trailing events then close
        if run.status in ("completed", "escalated", "failed", "paused_quota", "cancelled"):
            trailing = await event_repo.list_events_after(run_id=run_id, after_seq=current_seq)
            for ev in trailing:
                current_seq = ev.seq
                yield ev

            # If no terminal event in trailing yet (e.g. cross-process writer committing), retry briefly
            has_term = any(e.type in ("run_finished", "error", "quota_paused") for e in trailing)
            if not has_term:
                for _ in range(5):
                    await asyncio.sleep(0.1)
                    trailing2 = await event_repo.list_events_after(run_id=run_id, after_seq=current_seq)
                    for ev in trailing2:
                        current_seq = ev.seq
                        yield ev
                        if ev.type in ("run_finished", "error", "quota_paused"):
                            return
            return

        # Stale heartbeat check: if status is running but heartbeat has gone stale
        if run.status == "running":
            stale_threshold = int(os.environ.get("HEARTBEAT_STALE_SECONDS", "60"))
            if run.heartbeat_at:
                try:
                    hb_time = datetime.fromisoformat(run.heartbeat_at)
                    now_utc = datetime.now(timezone.utc)
                    if (now_utc - hb_time).total_seconds() > stale_threshold:
                        stale_ev = RunEvent(
                            run_id=run_id,
                            type="stale",
                            node="system",
                            data={
                                "message": f"Run heartbeat is stale (> {stale_threshold}s). Process appears dead.",
                                "heartbeat_at": run.heartbeat_at
                            }
                        )
                        yield stale_ev
                        return
                except Exception:
                    pass

        # 4. Periodic heartbeat generation (not persisted to DB)
        now = time.perf_counter()
        if now - last_heartbeat_time >= emit_heartbeat_interval:
            last_heartbeat_time = now
            yield RunEvent(
                run_id=run_id,
                type="heartbeat",
                node="system",
                data={"timestamp": utc_now_iso()}
            )

        # 5. Wait for new events: wake immediately via in-process bus or poll DB after poll_interval
        if bus and bus._running:
            try:
                await asyncio.wait_for(bus._wake_event.wait(), timeout=poll_interval)
            except asyncio.TimeoutError:
                pass
        else:
            # Subprocess or no in-process bus: standard poll sleep
            await asyncio.sleep(poll_interval)
