"""Unit tests for Resumable Runs, cooperative cancellation, QuotaExceeded pausing,
event de-duplication, and startup orphan recovery.
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest

from core.db.migrator import apply_migrations
from core.db.repositories import (
    ThreadRepo,
    RunRepo,
    EventRepo
)
from core.services.runs import (
    start_run,
    resume_run,
    cancel_run,
    execute_run
)
from core.ratelimit import QuotaExceededError
from tests.stubs import StubLLM
from core.llm import set_global_llm
from mcp_tools.client import set_global_tool, reset_global_tools


@pytest.fixture(autouse=True)
def setup_databases(tmp_path, monkeypatch):
    """Isolate DBs for every test run."""
    app_db = str(tmp_path / "app.db")
    chk_db = str(tmp_path / "checkpoints.db")
    monkeypatch.setenv("DATABASE_PATH", app_db)
    monkeypatch.setenv("CHECKPOINT_DB_PATH", chk_db)
    # Apply initial schema
    asyncio.run(apply_migrations(app_db))
    yield
    reset_global_tools()


@pytest.mark.asyncio
async def test_completed_runs_not_resumable():
    """Attempting to resume a completed or cancelled run must be refused."""
    thread_repo = ThreadRepo()
    run_repo = RunRepo()

    thread = await thread_repo.create_thread(title="Resumption Test")
    run = await run_repo.create_run(thread_id=thread.id, task="Task")

    # Mark completed
    await run_repo.update_run_status(run.id, status="completed", finished=True)
    with pytest.raises(ValueError, match="already completed and cannot be resumed"):
        await resume_run(run.id)

    # Mark cancelled
    await run_repo.update_run_status(run.id, status="cancelled", finished=True)
    with pytest.raises(ValueError, match="already cancelled and cannot be resumed"):
        await resume_run(run.id)


@pytest.mark.asyncio
async def test_quota_exceeded_pauses_with_resume_after_and_refuses_early_resume():
    """QuotaExceededError pauses run with resume_after timestamp; early resume is refused unless force=True."""
    thread_repo = ThreadRepo()

    thread = await thread_repo.create_thread(title="Quota Test")

    future_reset = (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat()

    class QuotaFailingLLM(StubLLM):
        async def generate_plan(self, *args, **kwargs):
            raise QuotaExceededError(
                "Daily quota exhausted",
                resume_after_iso=future_reset
            )

    set_global_llm(QuotaFailingLLM())

    run = await start_run(thread_id=thread.id, task="Investigate Quota Handling")
    result_run = await execute_run(run.id)

    assert result_run.status == "paused_quota"
    assert result_run.resume_after == future_reset

    # Attempting early resume without force should be refused
    with pytest.raises(ValueError, match="paused for daily quota until"):
        await resume_run(run.id, force=False)

    # Switching to working LLM and resuming with force=True should succeed
    set_global_llm(StubLLM())
    resumed_run = await resume_run(run.id, force=True)
    assert resumed_run.status in ("completed", "finished", "escalated")


@pytest.mark.asyncio
async def test_startup_recovery_marks_orphaned_running_runs_failed_and_preserves_live():
    """At startup, only orphaned runs older than 60s are marked failed; live runs are preserved."""
    thread_repo = ThreadRepo()
    run_repo = RunRepo()

    thread = await thread_repo.create_thread(title="Recovery Thread")

    # 1. Orphaned run (heartbeat 120s ago)
    old_time = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
    r_orphan = await run_repo.create_run(thread_id=thread.id, task="Orphan Task")
    await run_repo.update_run_status(r_orphan.id, status="running")
    # Manually backdate heartbeat
    from core.db.connection import get_db
    async with get_db() as conn:
        await conn.execute("UPDATE runs SET heartbeat_at = ? WHERE id = ?", (old_time, r_orphan.id))
        await conn.commit()

    # 2. Live run (heartbeat 5s ago)
    r_live = await run_repo.create_run(thread_id=thread.id, task="Live Task")
    await run_repo.update_run_status(r_live.id, status="running")
    await run_repo.update_heartbeat(r_live.id)

    # Run startup recovery
    recovered = await run_repo.recover_interrupted_runs(cutoff_seconds=60)
    assert recovered == 1

    check_orphan = await run_repo.get_run(r_orphan.id)
    assert check_orphan.status == "failed"
    assert check_orphan.error == "interrupted"

    check_live = await run_repo.get_run(r_live.id)
    assert check_live.status == "running"


@pytest.mark.asyncio
async def test_cancel_stops_run_cooperatively():
    """cancel_run sets cancel_requested and halts run execution cooperatively."""
    thread_repo = ThreadRepo()

    thread = await thread_repo.create_thread(title="Cancel Thread")
    run = await start_run(thread_id=thread.id, task="Long Running Query")

    # Request cancel before execution
    await cancel_run(run.id)

    finished_run = await execute_run(run.id)
    assert finished_run.status == "cancelled"


@pytest.mark.asyncio
async def test_interrupted_run_resumes_without_rerunning_completed_subtasks(monkeypatch):
    """An interrupted run resumes from checkpoint without re-running completed subtasks or duplicating searches."""
    thread_repo = ThreadRepo()
    event_repo = EventRepo()

    thread = await thread_repo.create_thread(title="Interruption & Resumption")
    run = await start_run(thread_id=thread.id, task="Compare LangGraph and AutoGen", config={"search_budget": 6, "max_retries": 0})

    searches_conducted: list[str] = []

    def tracking_search(query: str, max_results: int = 3, **kwargs):
        searches_conducted.append(query)
        return '{"status": "success", "results": [{"title": "Doc", "url": "https://example.com/doc", "snippet": "Text"}]}'

    set_global_tool("search_web", tracking_search)

    # First attempt: Synthesizer fails mid-graph (after Planner and Executor finished)
    attempt_count = 0

    from core.nodes.synthesizer import synthesizer_node as orig_synth

    async def flaky_synth(state):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise RuntimeError("Simulated crash in synthesizer")
        return await orig_synth(state)

    monkeypatch.setattr("core.graph.synthesizer_node", flaky_synth)

    # Run first attempt (should fail at synthesizer)
    failed_run = await execute_run(run.id)
    assert failed_run.status == "failed"
    assert "Simulated crash in synthesizer" in failed_run.error

    initial_search_count = len(searches_conducted)
    assert initial_search_count > 0  # Executor did run queries!

    events_before_resume = await event_repo.list_events_by_run(run.id)
    count_before = len(events_before_resume)
    assert count_before > 0

    # Resume the run!
    resumed_run = await resume_run(run.id, force=True)
    assert resumed_run.status in ("completed", "finished", "escalated")
    assert resumed_run.report_md is not None

    # Crucial check: Executor was NOT re-run; completed subtasks were NOT repeated!
    final_search_count = len(searches_conducted)
    assert final_search_count == initial_search_count, (
        f"Searches were re-run on resume! Before: {initial_search_count}, After: {final_search_count}"
    )

    # Event persistence check: no duplicate events after resume
    events_after_resume = await event_repo.list_events_by_run(run.id)
    # The events should continue sequentially without repeating the previous events
    event_types = [e.type for e in events_after_resume]
    # Decomposition should occur only once
    assert event_types.count("decomposition") == 1


@pytest.mark.asyncio
async def test_real_subprocess_kill_and_resumption(tmp_path, monkeypatch):
    """Real kill test:
    1. Spawn run in worker subprocess.
    2. Terminate child with process.kill() mid-run (after executor completes subtasks).
    3. Assert run remains status='running' with a heartbeat that goes stale.
    4. Shorten HEARTBEAT_STALE_SECONDS and run startup recovery -> status='failed', error='interrupted'.
    5. Call resume_run.
    6. Assert completed subtasks are NOT repeated and no run_events are duplicated.
    """
    thread_repo = ThreadRepo()
    run_repo = RunRepo()
    event_repo = EventRepo()

    thread = await thread_repo.create_thread(title="Subprocess Kill Thread")
    run = await start_run(
        thread_id=thread.id,
        task="Investigate Subprocess Resilience",
        config={"search_budget": 6, "max_retries": 0}
    )

    env = os.environ.copy()
    env["DATABASE_PATH"] = os.environ["DATABASE_PATH"]
    env["CHECKPOINT_DB_PATH"] = os.environ["CHECKPOINT_DB_PATH"]
    env["CACHE_ENABLED"] = "false"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)

    signal_file = tmp_path / "mid_executor.signal"
    worker_log = tmp_path / "worker.log"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "scripts.run_worker",
        "--run-id",
        run.id,
        "--kill-point",
        "executor",
        "--signal-file",
        str(signal_file)
    ]

    log_file_handle = open(worker_log, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )

    # 1. Wait non-blocking for child to signal executor completion pause point
    reached_pause = False
    for _ in range(150):  # 15s max
        if signal_file.exists():
            reached_pause = True
            break
        await asyncio.sleep(0.1)

    if not reached_pause:
        proc.kill()
        proc.wait()
        log_file_handle.close()
        log_content = worker_log.read_text(encoding="utf-8") if worker_log.exists() else "No log"
        pytest.fail(f"Worker subprocess did not reach pause point.\nWorker Log:\n{log_content}")

    # 2. Terminate subprocess abruptly with real process.kill()
    proc.kill()
    proc.wait()
    log_file_handle.close()
    assert proc.poll() is not None

    # 3. In parent: assert run is still status='running' with heartbeat from dead child
    run_record = await run_repo.get_run(run.id)
    assert run_record.status == "running"
    assert run_record.heartbeat_at is not None

    # 4. Shorten staleness threshold and assert heartbeat goes stale
    monkeypatch.setenv("HEARTBEAT_STALE_SECONDS", "1")
    await asyncio.sleep(1.2)

    # 5. Run startup recovery -> run becomes failed/'interrupted'
    recovered = await run_repo.recover_interrupted_runs()
    assert recovered == 1

    run_after_recovery = await run_repo.get_run(run.id)
    assert run_after_recovery.status == "failed"
    assert run_after_recovery.error == "interrupted"

    # Events before resume
    events_before_resume = await event_repo.list_events_by_run(run.id)
    assert len(events_before_resume) > 0

    # 6. Setup parent test doubles and resume
    set_global_llm(StubLLM())
    searches_conducted_on_resume: list[str] = []

    def mock_search_on_resume(query: str, max_results: int = 3, **kwargs):
        searches_conducted_on_resume.append(query)
        return json.dumps({
            "status": "success",
            "results": [{"title": "Doc", "url": "https://example.com/doc", "snippet": "Text"}]
        })

    set_global_tool("search_web", mock_search_on_resume)

    resumed_run = await resume_run(run.id, force=True)
    assert resumed_run.status in ("completed", "finished", "escalated")
    assert resumed_run.report_md is not None

    # 7. Assert completed subtasks were NOT repeated (0 searches conducted on resume)
    assert len(searches_conducted_on_resume) == 0, (
        f"Executor subtasks were repeated on resume! Searches: {searches_conducted_on_resume}"
    )

    # 8. Assert no run_events are duplicated
    events_after_resume = await event_repo.list_events_by_run(run.id)
    assert len(events_after_resume) > len(events_before_resume)

    # Ensure monotonic sequence numbering without duplicate seq numbers
    seqs = [e.seq for e in events_after_resume]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))

    # Initial events preserved exactly in prefix
    for idx, ev_before in enumerate(events_before_resume):
        assert events_after_resume[idx].seq == ev_before.seq
        assert events_after_resume[idx].type == ev_before.type

    # Single decomposition event
    types = [e.type for e in events_after_resume]
    assert types.count("decomposition") == 1


@pytest.mark.asyncio
async def test_critic_event_ordering_and_seq_order():
    """Critic must emit rubric_evaluation before escalation, and run_events seq order
    strictly matches emission order across a resume."""
    thread_repo = ThreadRepo()
    event_repo = EventRepo()
    set_global_llm(StubLLM())

    def mock_search(query: str, max_results: int = 3, **kwargs):
        return json.dumps({
            "status": "success",
            "results": [{"title": "Doc", "url": "https://example.com/doc", "snippet": "Evidence"}]
        })

    set_global_tool("search_web", mock_search)

    thread = await thread_repo.create_thread(title="Critic Ordering Thread")
    run = await start_run(
        thread_id=thread.id,
        task="Audit Ordering Invariants",
        config={"max_retries": 0, "search_budget": 4}
    )

    result_run = await execute_run(run.id)
    assert result_run.status == "escalated"

    events = await event_repo.list_events_by_run(run.id)
    types = [e.type for e in events]

    assert "rubric_evaluation" in types
    assert "escalation" in types

    rubric_idx = types.index("rubric_evaluation")
    escalation_idx = types.index("escalation")

    # Critical requirement: rubric_evaluation MUST precede escalation
    assert rubric_idx < escalation_idx, (
        f"Expected rubric_evaluation (idx {rubric_idx}) before escalation (idx {escalation_idx})"
    )

    # Sequence numbers strictly monotonic
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))

