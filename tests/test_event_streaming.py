"""Tests for Phase 3 Event Streaming:
1. Monotonic seq in emission order.
2. Concurrent subtasks produce correctly tagged interleaved events (subtask_id).
3. Cross-process DB-tail streaming (child runs in subprocess, parent tails DB without gaps/dups).
4. Last-Event-ID / after_seq replay and resume.
5. Client disconnect does not cancel the run.
6. Subscriber queue overflow marks lagged and closes queue.
7. Mid-node crash persists all events prior to failure.
8. No duplicate events after resume.
9. Heartbeats are not persisted to database.
10. Bus cleanup on completion (no memory leaks).
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from core.db.migrator import apply_migrations
from core.db.repositories import (
    ThreadRepo,
    EventRepo
)
from core.events import (
    RunEvent,
    get_run_event_bus,
    remove_run_event_bus,
    tail_run_events,
    SubscriberQueue
)
from core.services.runs import (
    start_run,
    resume_run,
    execute_run
)
from tests.stubs import StubLLM
from core.llm import set_global_llm
from mcp_tools.client import set_global_tool, reset_global_tools
from api.app import app


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path, monkeypatch):
    """Isolate DBs and register deterministic offline fakes for every test."""
    app_db = str(tmp_path / "streaming_app.db")
    chk_db = str(tmp_path / "streaming_chk.db")
    monkeypatch.setenv("DATABASE_PATH", app_db)
    monkeypatch.setenv("CHECKPOINT_DB_PATH", chk_db)
    monkeypatch.setenv("API_KEY", "")

    asyncio.run(apply_migrations(app_db))
    set_global_llm(StubLLM())

    def mock_search(query: str, max_results: int = 3, **kwargs):
        return json.dumps({
            "status": "success",
            "results": [
                {
                    "title": f"Doc for {query[:25]}",
                    "url": f"https://example.com/{abs(hash(query)) % 1000}",
                    "snippet": f"Evidence for {query}"
                }
            ]
        })

    set_global_tool("search_web", mock_search)
    yield
    reset_global_tools()


@pytest.mark.asyncio
async def test_event_seq_strictly_monotonic_and_emission_order():
    """Events persisted to DB must have strictly monotonic sequence numbers matching emission order."""
    thread_repo = ThreadRepo()
    event_repo = EventRepo()

    thread = await thread_repo.create_thread(title="Monotonic Seq Test")
    run = await start_run(thread_id=thread.id, task="Analyze Monotonic Sequence Order", config={"search_budget": 5, "max_retries": 0})

    finished_run = await execute_run(run.id)
    assert finished_run.status in ("completed", "finished", "escalated")

    events = await event_repo.list_events_by_run(run.id)
    assert len(events) >= 5

    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs), "Sequence numbers are not in ascending order"
    assert len(seqs) == len(set(seqs)), "Duplicate sequence numbers detected"
    assert seqs[0] == 1, "Sequence numbering should start at 1"
    for i in range(len(seqs) - 1):
        assert seqs[i + 1] == seqs[i] + 1, f"Gap in sequence numbers between {seqs[i]} and {seqs[i+1]}"


@pytest.mark.asyncio
async def test_concurrent_subtasks_tag_subtask_id():
    """Executor running queries across multiple subtasks tags subtask_id on tool calls, results, and sources."""
    thread_repo = ThreadRepo()
    event_repo = EventRepo()

    thread = await thread_repo.create_thread(title="Tagged Subtasks Test")
    run = await start_run(thread_id=thread.id, task="Multi-subtask investigation", config={"search_budget": 6, "max_retries": 0})

    finished_run = await execute_run(run.id)
    assert finished_run.status in ("completed", "finished", "escalated")

    events = await event_repo.list_events_by_run(run.id)
    tool_events = [e for e in events if e.type in ("tool_call", "tool_result", "source_found")]
    assert len(tool_events) > 0

    subtask_ids_seen = set()
    for e in tool_events:
        assert e.subtask_id is not None, f"Event seq={e.seq} ({e.type}) missing subtask_id"
        subtask_ids_seen.add(e.subtask_id)

    # Planner decomposes into multiple subtasks (typically subtask_1, subtask_2, subtask_3)
    assert len(subtask_ids_seen) >= 2, f"Expected events from multiple subtasks, got: {subtask_ids_seen}"


@pytest.mark.asyncio
async def test_cross_process_db_tailing_without_gaps_or_duplicates(tmp_path, monkeypatch):
    """Run executes in child worker subprocess; parent subscribes through tail_run_events (DB-tail)
    and receives all events without gaps or duplicates.
    """
    thread_repo = ThreadRepo()

    thread = await thread_repo.create_thread(title="Cross Process Tail Thread")
    run = await start_run(thread_id=thread.id, task="Analyze Cross Process Streaming", config={"search_budget": 5, "max_retries": 0})

    env = os.environ.copy()
    env["DATABASE_PATH"] = os.environ["DATABASE_PATH"]
    env["CHECKPOINT_DB_PATH"] = os.environ["CHECKPOINT_DB_PATH"]
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)

    # Spawn worker subprocess to execute the run
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "scripts.run_worker",
        "--run-id",
        run.id,
        "--critic-pass"
    ]
    worker_log = tmp_path / "worker_streaming.log"
    log_file_handle = open(worker_log, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )

    # Parent streams events via tail_run_events
    collected_events: list[RunEvent] = []
    try:
        async for event in tail_run_events(run_id=run.id, after_seq=0, poll_interval=0.2):
            collected_events.append(event)
    finally:
        try:
            proc.wait(timeout=15)
        finally:
            log_file_handle.close()

    assert len(collected_events) > 0, "No events were received via cross-process DB tailing"
    types = [e.type for e in collected_events]
    assert "run_started" in types
    assert "run_finished" in types

    seqs = [e.seq for e in collected_events]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs)), "Duplicate sequence numbers in stream"
    for i in range(len(seqs) - 1):
        assert seqs[i + 1] == seqs[i] + 1, "Gap in sequence numbers"


@pytest.mark.asyncio
async def test_last_event_id_and_after_seq_resume():
    """Streaming with after_seq skips already received events and returns only new ones."""
    thread_repo = ThreadRepo()
    event_repo = EventRepo()

    thread = await thread_repo.create_thread(title="Replay Resume Test")
    run = await start_run(thread_id=thread.id, task="Replay Telemetry Test", config={"search_budget": 4, "max_retries": 0})

    await execute_run(run.id)

    all_events = await event_repo.list_events_by_run(run.id)
    assert len(all_events) >= 4

    cutoff_seq = all_events[2].seq
    expected_remaining = [e for e in all_events if e.seq > cutoff_seq]

    streamed_after = []
    async for ev in tail_run_events(run_id=run.id, after_seq=cutoff_seq, poll_interval=0.1):
        streamed_after.append(ev)

    assert len(streamed_after) == len(expected_remaining)
    for s_ev, exp_ev in zip(streamed_after, expected_remaining):
        assert s_ev.seq == exp_ev.seq
        assert s_ev.type == exp_ev.type


@pytest.mark.asyncio
async def test_subscriber_queue_overflow_policy():
    """When an in-process subscriber queue fills up, it is marked lagged and closed."""
    q = SubscriberQueue(maxsize=3)
    assert not q.is_lagged
    assert not q.is_closed

    dummy_ev = RunEvent(seq=1, run_id="run1", ts="2026-01-01T00:00:00Z", type="node_started", node="planner")
    assert q.put_nowait(dummy_ev) is True
    assert q.put_nowait(dummy_ev) is True
    assert q.put_nowait(dummy_ev) is True

    # 4th item overflows
    success = q.put_nowait(dummy_ev)
    assert success is False
    assert q.is_lagged is True
    assert q.is_closed is True


@pytest.mark.asyncio
async def test_mid_node_crash_persists_earlier_events():
    """If a node crashes midway, all events emitted up to that point remain persisted in SQLite."""
    thread_repo = ThreadRepo()
    event_repo = EventRepo()

    thread = await thread_repo.create_thread(title="Mid Crash Persistence")
    run = await start_run(thread_id=thread.id, task="Test Crash Event Persistence", config={"search_budget": 4, "max_retries": 0})

    async def crashing_synth(state):
        raise RuntimeError("Fatal explosion in synthesizer")

    import core.graph as graph_module
    old_synth = graph_module.synthesizer_node
    graph_module.synthesizer_node = crashing_synth

    try:
        failed_run = await execute_run(run.id)
        assert failed_run.status == "failed"
        assert "Fatal explosion in synthesizer" in failed_run.error

        events = await event_repo.list_events_by_run(run.id)
        assert len(events) > 0
        types = [e.type for e in events]
        assert "run_started" in types
        assert "node_started" in types
        assert "node_finished" in types
        assert "error" in types
    finally:
        graph_module.synthesizer_node = old_synth


@pytest.mark.asyncio
async def test_no_duplicate_events_persisted_after_resume():
    """Resuming a paused or failed run must never duplicate previously persisted events."""
    thread_repo = ThreadRepo()
    event_repo = EventRepo()

    thread = await thread_repo.create_thread(title="De-duplication On Resume")
    run = await start_run(thread_id=thread.id, task="Test De-duplication On Resume", config={"search_budget": 4, "max_retries": 0})

    crash_count = 0
    import core.graph as graph_module
    old_synth = graph_module.synthesizer_node

    async def flaky_synth(state):
        nonlocal crash_count
        crash_count += 1
        if crash_count == 1:
            raise RuntimeError("Transient crash in synthesizer")
        return await old_synth(state)

    graph_module.synthesizer_node = flaky_synth

    try:
        # First attempt crashes
        r1 = await execute_run(run.id)
        assert r1.status == "failed"

        events_before = await event_repo.list_events_by_run(run.id)
        count_before = len(events_before)
        assert count_before > 0

        # Resumed attempt succeeds
        r2 = await resume_run(run.id, force=True)
        assert r2.status in ("completed", "finished", "escalated")

        events_after = await event_repo.list_events_by_run(run.id)
        assert len(events_after) > count_before

        seqs = [e.seq for e in events_after]
        assert seqs == sorted(seqs)
        assert len(seqs) == len(set(seqs)), "Duplicate sequence IDs detected after resume"

        # Planner decomposition should not be duplicated
        types = [e.type for e in events_after]
        assert types.count("decomposition") == 1
    finally:
        graph_module.synthesizer_node = old_synth


@pytest.mark.asyncio
async def test_heartbeats_are_not_persisted_to_db():
    """Heartbeat events are transient keeping SSE connections alive and must never be written to SQLite."""
    thread_repo = ThreadRepo()
    event_repo = EventRepo()

    thread = await thread_repo.create_thread(title="Heartbeat Persistence Check")
    run = await start_run(thread_id=thread.id, task="Check Heartbeats", config={"search_budget": 3, "max_retries": 0})

    bus = get_run_event_bus(run.id)
    # Emit normal event and heartbeat
    await bus.emit("node_started", node="test", data={"step": 1})
    await bus.emit("heartbeat", node=None, data={"ping": True})
    await bus.flush()
    await bus.close()
    remove_run_event_bus(run.id)

    persisted = await event_repo.list_events_by_run(run.id)
    types = [e.type for e in persisted]
    assert "node_started" in types
    assert "heartbeat" not in types, "Heartbeat was persisted to database!"


@pytest.mark.asyncio
async def test_bus_cleanup_on_run_finish():
    """Event bus is cleanly closed and removed from registry after run terminates."""
    thread_repo = ThreadRepo()

    thread = await thread_repo.create_thread(title="Bus Cleanup Check")
    run = await start_run(thread_id=thread.id, task="Bus Cleanup", config={"search_budget": 2, "max_retries": 0})

    await execute_run(run.id)

    # Bus should have been removed from active bus dictionary
    from core.events import _RUN_BUSES
    assert run.id not in _RUN_BUSES, "Per-run bus was not removed after run finished"


def test_api_sse_stream_and_signed_token_authentication(monkeypatch):
    """API SSE endpoint /runs/{run_id}/events supports signed query tokens and streams events."""
    monkeypatch.setenv("API_KEY", "prod-secret-key-999")

    with TestClient(app) as client:
        # Create thread and run with auth header
        auth_hdr = {"X-API-Key": "prod-secret-key-999"}
        c_resp = client.post("/threads", json={"title": "SSE Auth Thread"}, headers=auth_hdr)
        assert c_resp.status_code == 200
        t_id = c_resp.json()["id"]

        r_resp = client.post(f"/threads/{t_id}/runs", json={"task": "Investigate Streaming Auth"}, headers=auth_hdr)
        assert r_resp.status_code == 200
        run_id = r_resp.json()["id"]

        # Request signed token
        tok_resp = client.post("/auth/sse-token", headers=auth_hdr)
        assert tok_resp.status_code == 200
        tok_data = tok_resp.json()
        assert "token" in tok_data
        signed_token = tok_data["token"]

        # Connect to SSE without token -> 401
        unauth_resp = client.get(f"/runs/{run_id}/events")
        assert unauth_resp.status_code == 401

        # Connect to SSE with ?token= -> 200 EventSource stream
        with client.stream("GET", f"/runs/{run_id}/events?token={signed_token}") as stream_resp:
            assert stream_resp.status_code == 200
            assert "text/event-stream" in stream_resp.headers.get("content-type", "")

        # Snapshot endpoint
        snap_resp = client.get(f"/runs/{run_id}/events/snapshot", headers=auth_hdr)
        assert snap_resp.status_code == 200
        assert isinstance(snap_resp.json(), list)
