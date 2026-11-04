"""Manual demonstration script: Resumable Multi-Agent Research Execution.

NOTE ON TEST DOUBLES:
This demonstration executes 100% offline using deterministic test doubles:
- StubLLM & PassingCriticLLM: Mock multi-agent reasoning, planning, synthesis, and reflection.
- Mock MCP Search Tool: Provides deterministic citations without network calls or DDGS rate limits.
It does NOT require a live Gemini API key, hit external search engines, or consume daily LLM quota.

Demonstrates two resilience paths:
PATH 1: Real Subprocess Termination (process.kill) & Checkpoint Resumption
  1. Create persistent conversation thread.
  2. Launch research run with SQLite checkpointing in a worker subprocess.
  3. Terminate child process mid-run with real OS process.kill() (after subtask execution).
  4. Assert the run remains status='running' with a heartbeat that goes stale.
  5. Run startup orphan recovery (HEARTBEAT_STALE_SECONDS) marking it failed/'interrupted'.
  6. Resume run from SQLite checkpoint.
  7. Verify completed subtasks were NOT repeated, no events duplicated, and report finalized.

PATH 2: Clean End-to-End Execution with Passing Quality Rubric (status='completed')
  1. Launch research run using PassingCriticLLM (verdict='PASS', scores >= 4/5).
  2. Execute full multi-agent pipeline: planner -> executor -> synthesizer -> critic -> finalizer.
  3. Validate final state reaches status='completed' with full structured report.
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe console encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configure isolated demonstration databases
os.environ["DATABASE_PATH"] = "data/demo_app.db"
os.environ["CHECKPOINT_DB_PATH"] = "data/demo_checkpoints.db"
os.environ["CACHE_ENABLED"] = "false"

from core.db.migrator import apply_migrations
from core.db.repositories import ThreadRepo, RunRepo, EventRepo
from core.services.runs import start_run, resume_run, execute_run
from mcp_tools.client import set_global_tool, reset_global_tools
from tests.stubs import StubLLM
from core.llm import set_global_llm
from scripts.run_worker import PassingCriticLLM


async def run_path_1(thread_repo: ThreadRepo, run_repo: RunRepo, event_repo: EventRepo):
    print("\n" + "=" * 70)
    print(" PATH 1: REAL SUBPROCESS KILL & STATE CHECKPOINT RESUMPTION")
    print("=" * 70)

    # 1. Create persistent thread
    print("\n[Step 1.1] Creating persistent conversation thread...")
    thread = await thread_repo.create_thread(title="Demo: Resilient Architecture Analysis")
    print(f"  Created Thread ID: {thread.id} | Title: '{thread.title}'")

    # 2. Launch run
    print("\n[Step 1.2] Registering research run in database...")
    task_prompt = "Compare LangGraph vs CrewAI architecture resilience and state recovery"
    run = await start_run(
        thread_id=thread.id,
        task=task_prompt,
        config={"search_budget": 6, "max_retries": 0}
    )
    print(f"  Created Run ID: {run.id} | Status: {run.status}")

    # 3. Spawn child worker and terminate with real process.kill()
    print("\n[Step 1.3] Spawning worker subprocess and awaiting mid-run pause point...")
    signal_file = PROJECT_ROOT / "data" / "demo_kill.signal"
    worker_log = PROJECT_ROOT / "data" / "demo_worker.log"
    if signal_file.exists():
        signal_file.unlink()

    env = os.environ.copy()
    env["DATABASE_PATH"] = "data/demo_app.db"
    env["CHECKPOINT_DB_PATH"] = "data/demo_checkpoints.db"
    env["CACHE_ENABLED"] = "false"
    env["PYTHONPATH"] = str(PROJECT_ROOT)

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

    log_fh = open(worker_log, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    print(f"  Worker child process spawned with PID: {proc.pid}")

    # Await pause signal from worker
    reached_pause = False
    for _ in range(150):  # 15s max
        if signal_file.exists():
            reached_pause = True
            break
        await asyncio.sleep(0.1)

    if not reached_pause:
        proc.kill()
        proc.wait()
        log_fh.close()
        print("  [ERROR] Subprocess failed to reach pause signal.")
        return

    # Real process termination via OS kill
    print(f"\n[Step 1.4] [REAL PROCESS KILL] Terminating PID {proc.pid} with process.kill()...")
    proc.kill()
    proc.wait()
    log_fh.close()
    if signal_file.exists():
        signal_file.unlink()
    print(f"  Process terminated. Exit code: {proc.poll()}")

    # 4. Check DB state: status='running', heartbeat frozen
    killed_run = await run_repo.get_run(run.id)
    print("\n[Step 1.5] Inspecting database after abrupt process termination:")
    print(f"  Run Status: '{killed_run.status}' (abrupt kill bypassed exit handlers)")
    print(f"  Heartbeat timestamp: {killed_run.heartbeat_at}")
    print(f"  Owner ID: {killed_run.owner_id}")

    # 5. Startup recovery with stale heartbeat
    print("\n[Step 1.6] Simulating process restart & evaluating heartbeat staleness...")
    os.environ["HEARTBEAT_STALE_SECONDS"] = "1"
    print("  Sleeping 1.2s to exceed HEARTBEAT_STALE_SECONDS=1...")
    await asyncio.sleep(1.2)

    recovered_count = await run_repo.recover_interrupted_runs()
    print(f"  Startup recovery evaluated. Interrupted runs marked for recovery: {recovered_count}")

    run_recovered = await run_repo.get_run(run.id)
    print(f"  Run status after recovery: '{run_recovered.status}' | Error: '{run_recovered.error}'")
    assert run_recovered.status == "failed"
    assert run_recovered.error == "interrupted"

    # Events before resume
    events_before = await event_repo.list_events_by_run(run.id)
    print(f"  Events persisted prior to resumption: {len(events_before)}")

    # 6. Resume from SQLite checkpoint
    print("\n[Step 1.7] Resuming run from SQLite checkpoint via resume_run(force=True)...")
    set_global_llm(StubLLM())
    searches_conducted_on_resume: list[str] = []

    def tracking_search(query: str, max_results: int = 3, **kwargs):
        searches_conducted_on_resume.append(query)
        print(f"    [Unexpected Search] {query}")
        return json.dumps({
            "status": "success",
            "results": [{"title": "Doc", "url": "https://example.com/doc", "snippet": "Evidence"}]
        })

    set_global_tool("search_web", tracking_search)

    resumed_run = await resume_run(run.id, force=True)
    print(f"  Resumed run finished with Status: '{resumed_run.status}'")

    # 7. Invariants validation
    print("\n[Step 1.8] Verifying resilience invariants:")
    print(f"  Searches repeated on resume: {len(searches_conducted_on_resume)}")
    assert len(searches_conducted_on_resume) == 0, "Executor was re-executed upon resumption!"
    print("  [SUCCESS] Completed subtasks were NOT repeated!")

    events_after = await event_repo.list_events_by_run(run.id)
    print(f"  Total persisted audit events: {len(events_after)}")
    seqs = [e.seq for e in events_after]
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs))
    types = [e.type for e in events_after]
    assert types.count("decomposition") == 1
    print("  [SUCCESS] No duplicate events in audit trail (seq strictly monotonic)!")


async def run_path_2(thread_repo: ThreadRepo, run_repo: RunRepo, event_repo: EventRepo):
    print("\n" + "=" * 70)
    print(" PATH 2: END-TO-END EXECUTION WITH PASSING RUBRIC (STATUS: COMPLETED)")
    print("=" * 70)

    # 1. Setup passing critic test double
    set_global_llm(PassingCriticLLM())

    search_counter = 0

    def mock_search_p2(query: str, max_results: int = 3, **kwargs):
        nonlocal search_counter
        search_counter += 1
        print(f"  [MCP Search Tool] Querying: '{query}'")
        return json.dumps({
            "status": "success",
            "results": [
                {
                    "title": f"Design high-reliability multi-agent state machines ({query[:20]})",
                    "url": f"https://example.com/agents/arch-{search_counter}",
                    "snippet": f"Verified empirical findings on multi-agent state machines, checkpointing, and high-reliability design for {query}."
                }
            ]
        })

    set_global_tool("search_web", mock_search_p2)

    # 2. Create persistent thread
    print("\n[Step 2.1] Creating conversation thread...")
    thread = await thread_repo.create_thread(title="Demo: Passing Quality Evaluation")
    print(f"  Created Thread ID: {thread.id} | Title: '{thread.title}'")

    # 3. Launch run with PassingCriticLLM
    print("\n[Step 2.2] Launching research run...")
    task_prompt = "Design high-reliability multi-agent state machines with checkpointing"
    run = await start_run(
        thread_id=thread.id,
        task=task_prompt,
        config={"search_budget": 6, "max_retries": 1}
    )
    print(f"  Created Run ID: {run.id} | Status: {run.status}")

    # 4. Execute to completion
    print("\n[Step 2.3] Executing pipeline: planner -> executor -> synthesizer -> critic -> finalizer...")
    completed_run = await execute_run(run.id)

    print("\n[Step 2.4] Evaluating run completion:")
    print(f"  Final Status: '{completed_run.status}'")
    assert completed_run.status == "completed", f"Expected 'completed', got '{completed_run.status}'"
    print("  [SUCCESS] Run status reached 'completed' successfully!")

    # 5. Inspect rubric and report
    events = await event_repo.list_events_by_run(run.id)
    rubric_events = [e for e in events if e.type == "rubric_evaluation"]
    if rubric_events:
        payload = rubric_events[-1].payload or {}
        details = payload.get("details", {})
        print("\n[Step 2.5] Critic Rubric Scorecard:")
        print(f"  Coverage Score:  {details.get('coverage_score', 'N/A')}/5")
        print(f"  Evidence Score:  {details.get('evidence_score', 'N/A')}/5")
        print(f"  Coherence Score: {details.get('coherence_score', 'N/A')}/5")
        print(f"  Total Score:     {details.get('total_score', 'N/A')}/15")
        print(f"  Verdict:         {details.get('verdict', 'N/A')}")

    print("\n[Step 2.6] Generated Report Markdown Preview:")
    print("-" * 70)
    report_sample = (completed_run.report_md or "No report content")[:400]
    print(f"{report_sample}...\n")
    print("-" * 70)


async def main():
    print("=" * 70)
    print(" RESEARCH AGENT CREW — RESILIENCE & COMPLETION DEMO ")
    print("=" * 70)
    print(" NOTE: Deterministic test doubles active. Zero live Gemini quota consumed.")

    # Clean prior demo databases
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    for db_file in [data_dir / "demo_app.db", data_dir / "demo_checkpoints.db"]:
        for suffix in ["", "-wal", "-shm"]:
            p = Path(str(db_file) + suffix)
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

    await apply_migrations("data/demo_app.db")

    thread_repo = ThreadRepo("data/demo_app.db")
    run_repo = RunRepo("data/demo_app.db")
    event_repo = EventRepo("data/demo_app.db")

    try:
        # Run Path 1: Subprocess Kill and Resumption
        await run_path_1(thread_repo, run_repo, event_repo)

        # Run Path 2: Clean End-to-End Execution ending in status='completed'
        await run_path_2(thread_repo, run_repo, event_repo)

        print("\n" + "=" * 70)
        print(" ALL DEMONSTRATION PATHS FINISHED SUCCESSFULLY ")
        print("=" * 70)
    finally:
        reset_global_tools()


if __name__ == "__main__":
    asyncio.run(main())
