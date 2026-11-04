"""Central Run orchestration and service layer for Research Agent Crew.

Provides domain operations to start, resume, cancel, and inspect runs,
enforcing lifecycle states, incremental event persistence, cooperative cancellation,
periodic owner heartbeats, and LangGraph SQLite checkpointing.
"""

import os
import json
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from core.db.repositories import (
    ThreadRepo,
    MessageRepo,
    RunRepo,
    RunRecord
)
from core.checkpoint import get_checkpointer
from core.graph import build_research_graph, create_initial_state
from core.ratelimit import QuotaExceededError, get_daily_quota_tracker
from core.cancellation import (
    is_cancellation_requested,
    request_cancellation_in_process,
    clear_cancellation_in_process,
    RunCancelledError
)
from core.events import get_run_event_bus, remove_run_event_bus


async def start_run(
    thread_id: str,
    task: str,
    config: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None
) -> RunRecord:
    """Create a new run in a thread and append the initial user message."""
    thread_repo = ThreadRepo(db_path=db_path)
    run_repo = RunRepo(db_path=db_path)
    msg_repo = MessageRepo(db_path=db_path)

    thread = await thread_repo.get_thread(thread_id)
    if not thread:
        raise ValueError(f"Thread '{thread_id}' not found")

    cfg = config or {}
    owner_id = f"worker_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    run = await run_repo.create_run(
        thread_id=thread_id,
        task=task,
        config=cfg,
        owner_id=owner_id
    )

    # Append user task message to thread
    await msg_repo.create_message(
        thread_id=thread_id,
        role="user",
        content=task,
        run_id=run.id
    )

    return run


async def resume_run(
    run_id: str,
    force: bool = False,
    db_path: Optional[str] = None,
    checkpoint_db_path: Optional[str] = None
) -> RunRecord:
    """Validate and execute resumption of a paused or interrupted run."""
    run_repo = RunRepo(db_path=db_path)
    run = await run_repo.get_run(run_id)
    if not run:
        raise ValueError(f"Run '{run_id}' not found")

    if run.status in ("completed", "cancelled"):
        raise ValueError(f"Run '{run_id}' is already {run.status} and cannot be resumed.")

    if run.status == "paused_quota":
        if run.resume_after and not force:
            try:
                resume_time = datetime.fromisoformat(run.resume_after)
                now_utc = datetime.now(timezone.utc)
                if now_utc < resume_time:
                    raise ValueError(
                        f"Run '{run_id}' is paused for daily quota until {run.resume_after}. "
                        f"Use force=True to resume immediately."
                    )
            except (ValueError, TypeError) as parse_err:
                if "paused for daily quota" in str(parse_err):
                    raise

    # Execute resumed run
    return await execute_run(
        run_id=run_id,
        is_resume=True,
        force=force,
        db_path=db_path,
        checkpoint_db_path=checkpoint_db_path
    )


async def cancel_run(run_id: str, db_path: Optional[str] = None) -> None:
    """Request cooperative cancellation for an in-flight run."""
    request_cancellation_in_process(run_id)
    run_repo = RunRepo(db_path=db_path)
    await run_repo.request_cancel(run_id)


async def get_run(run_id: str, db_path: Optional[str] = None) -> Optional[RunRecord]:
    """Retrieve run record by ID."""
    run_repo = RunRepo(db_path=db_path)
    return await run_repo.get_run(run_id)


async def list_runs(thread_id: str, db_path: Optional[str] = None) -> List[RunRecord]:
    """List all runs belonging to a thread."""
    run_repo = RunRepo(db_path=db_path)
    return await run_repo.list_runs_by_thread(thread_id)


async def execute_run(
    run_id: str,
    is_resume: bool = False,
    force: bool = False,
    db_path: Optional[str] = None,
    checkpoint_db_path: Optional[str] = None
) -> RunRecord:
    """Execute or resume a run through the LangGraph checkpointed loop."""
    run_repo = RunRepo(db_path=db_path)
    msg_repo = MessageRepo(db_path=db_path)

    run = await run_repo.get_run(run_id)
    if not run:
        raise ValueError(f"Run '{run_id}' not found")

    owner_id = f"worker_{os.getpid()}_{uuid.uuid4().hex[:6]}"
    clear_cancellation_in_process(run_id)

    # Mark run as running and set owner
    await run_repo.update_run_status(run_id, status="running")
    await run_repo.update_heartbeat(run_id, owner_id=owner_id)

    # Initialize per-run event bus
    bus = get_run_event_bus(run_id, db_path=db_path)

    # Heartbeat background task
    heartbeat_stop = asyncio.Event()

    async def _heartbeat_loop():
        while not heartbeat_stop.is_set():
            try:
                await asyncio.sleep(10.0)
                if not heartbeat_stop.is_set():
                    await run_repo.update_heartbeat(run_id, owner_id=owner_id)
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    heartbeat_task = asyncio.create_task(_heartbeat_loop())

    try:
        await bus.emit(
            event_type="run_started",
            node=None,
            data={"task": run.task, "thread_id": run.thread_id, "is_resume": is_resume}
        )

        async with get_checkpointer(checkpoint_db_path) as checkpointer:
            graph = build_research_graph(checkpointer=checkpointer)
            langgraph_thread_id = f"{run.thread_id}:{run.id}"
            graph_config = {
                "configurable": {
                    "thread_id": langgraph_thread_id,
                    "run_id": run.id
                }
            }

            # Check if cancellation was already requested before launch
            if await is_cancellation_requested(run.id, db_path=db_path):
                raise RunCancelledError(f"Run '{run.id}' was cancelled before starting.")

            if is_resume:
                try:
                    await graph.aupdate_state(graph_config, {"is_resume": True})
                except Exception:
                    pass
                input_state = None
            else:
                cfg = run.config or {}
                max_retries = int(cfg.get("max_retries", 1))
                search_budget = int(cfg.get("search_budget", 10))
                model_name = cfg.get("model_name") or os.environ.get("GEMINI_MODEL", "")
                input_state = create_initial_state(
                    task=run.task,
                    max_retries=max_retries,
                    max_search_budget=search_budget,
                    model_name=model_name,
                    run_id=run.id,
                    is_resume=False
                )

            # Stream execution chunk by chunk
            async for _ in graph.astream(input_state, config=graph_config, stream_mode="updates"):
                if await is_cancellation_requested(run.id, db_path=db_path):
                    raise RunCancelledError(f"Run '{run.id}' cancelled by user.")
                await bus.flush()

            # Execution completed cleanly; fetch final checkpoint state
            final_state_tuple = await graph.aget_state(graph_config)
            final_state = final_state_tuple.values if final_state_tuple else {}

            final_status = "escalated" if final_state.get("human_escalation_needed") else "completed"
            report_md = final_state.get("final_report")
            struct_obj = final_state.get("final_report_structured") or final_state.get("draft_report_structured")
            report_json = struct_obj.model_dump_json() if hasattr(struct_obj, "model_dump_json") else (
                json.dumps(struct_obj) if isinstance(struct_obj, dict) else None
            )

            metrics = {
                "searches_performed": final_state.get("total_searches_performed", 0),
                "cache_hits": final_state.get("cache_hits", 0),
                "retries": final_state.get("retry_count", 0),
                "timings": final_state.get("timings", {}),
                "mcp_saved_path": final_state.get("mcp_saved_path")
            }

            # Emit run_finished event via bus and flush to DB BEFORE updating run status in DB
            await bus.emit(
                event_type="run_finished",
                node="finalizer",
                data={"status": final_status, "metrics": metrics}
            )
            await bus.flush()

            await run_repo.save_run_result(
                run_id=run.id,
                status=final_status,
                report_md=report_md,
                report_json=report_json,
                metrics=metrics
            )

            # Append assistant message with final report to thread
            if report_md:
                await msg_repo.create_message(
                    thread_id=run.thread_id,
                    role="assistant",
                    content=report_md,
                    run_id=run.id,
                    metadata={"status": final_status, "metrics": metrics}
                )

    except QuotaExceededError as q_err:
        reset_time_iso = q_err.resume_after_iso or get_daily_quota_tracker().get_reset_time_iso()
        await bus.emit(
            event_type="quota_paused",
            node=None,
            data={"error": str(q_err), "resume_after": reset_time_iso}
        )
        await bus.flush()
        await run_repo.update_run_status(
            run_id=run.id,
            status="paused_quota",
            error=str(q_err),
            resume_after=reset_time_iso
        )
    except RunCancelledError as c_err:
        await bus.emit(
            event_type="error",
            node=None,
            data={"error": str(c_err), "cancelled": True}
        )
        await bus.flush()
        await run_repo.update_run_status(
            run_id=run.id,
            status="cancelled",
            error=str(c_err),
            finished=True
        )
    except Exception as exc:
        await bus.emit(
            event_type="error",
            node=None,
            data={"error": str(exc)}
        )
        await bus.flush()
        await run_repo.update_run_status(
            run_id=run.id,
            status="failed",
            error=str(exc),
            finished=True
        )
    finally:
        heartbeat_stop.set()
        heartbeat_task.cancel()
        clear_cancellation_in_process(run_id)
        try:
            await bus.flush()
            await bus.close()
        except Exception:
            pass
        remove_run_event_bus(run_id)

    updated_run = await run_repo.get_run(run_id)
    return updated_run or run
