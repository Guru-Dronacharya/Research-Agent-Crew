"""FastAPI Service for Research Agent Crew.

Exposes REST and SSE endpoints to launch research workflows, stream live agent
telemetry, inspect MCP tool registries, manage persistent conversation threads,
and resume/cancel execution runs with SQLite checkpointing.
"""

import os
import hmac
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Security, Request, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ResearchRequest, ResearchResponse, CriticSummarySchema,
    SubtaskSummarySchema, TraceEventSummarySchema,
    ThreadCreateRequest, ThreadUpdateRequest, ThreadResponse,
    MessageResponse, RunCreateRequest, ResumeRunRequest,
    RunResponse, RunEventResponse, ThreadExportResponse
)
from core.graph import build_research_graph, create_initial_state
from core.state import AgentState
from mcp_tools.client import MCPClientManager
from core.db.migrator import apply_migrations
from core.db.repositories import (
    ThreadRepo,
    MessageRepo,
    RunRepo,
    EventRepo
)
from core.services.runs import (
    start_run,
    resume_run,
    cancel_run,
    get_run,
    list_runs,
    execute_run
)

# API Key security header
api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    request: Request,
    api_key_header: Optional[str] = Security(api_key_header_scheme)
):
    """Verify API key using constant-time comparison if API_KEY is set in environment."""
    # Exclude health check endpoint from authentication
    if request.url.path == "/health":
        return

    expected_key = os.environ.get("API_KEY", "").strip()
    if not expected_key:
        return  # Authentication optional if API_KEY is unset

    # Support X-API-Key header or Authorization: Bearer <key>
    provided_key = api_key_header
    if not provided_key:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            provided_key = auth_header[7:].strip()

    if not provided_key or not hmac.compare_digest(provided_key.encode("utf-8"), expected_key.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: runs migrations and recovers interrupted runs on startup."""
    # 1. Apply database migrations
    await apply_migrations()
    # 2. Recover runs that were interrupted / killed without fresh heartbeats
    run_repo = RunRepo()
    await run_repo.recover_interrupted_runs(cutoff_seconds=60)
    yield


app = FastAPI(
    title="Research Agent Crew API",
    description="Multi-Agent Planner/Executor/Critic Research Service with MCP Tools, Threads, and Resumable Runs",
    version="2.0.0",
    lifespan=lifespan
)

# Secure CORS Configuration (No wildcard with credentials)
cors_env = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:8501").strip()
allowed_origins = [o.strip() for o in cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for legacy task runs (deprecated)
TASKS: Dict[str, Dict[str, Any]] = {}
mcp_client = MCPClientManager()
graph = build_research_graph()


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Enforce API key verification across all endpoints except /health."""
    if request.method != "OPTIONS" and request.url.path != "/health":
        expected_key = os.environ.get("API_KEY", "").strip()
        if expected_key:
            header_val = request.headers.get("X-API-Key")
            if not header_val:
                auth_val = request.headers.get("Authorization", "")
                if auth_val.startswith("Bearer "):
                    header_val = auth_val[7:].strip()
            if not header_val or not hmac.compare_digest(header_val.encode("utf-8"), expected_key.encode("utf-8")):
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=401, content={"detail": "Invalid or missing API Key"})
    return await call_next(request)


# ==============================================================================
# Health and System Endpoints
# ==============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint and MCP server status."""
    tools = mcp_client.list_tools()
    return {
        "status": "healthy",
        "service": "Research Agent Crew",
        "mcp_tools_available": len(tools),
        "active_tasks": len(TASKS)
    }


@app.get("/mcp/tools")
async def get_mcp_tools():
    """Return all discovered Model Context Protocol (MCP) tools."""
    return {"tools": mcp_client.list_tools()}


# ==============================================================================
# Persistent Threads Endpoints
# ==============================================================================

@app.get("/threads", response_model=List[ThreadResponse])
async def list_threads_endpoint(include_archived: bool = False):
    """List all research threads, sorted by recent activity."""
    repo = ThreadRepo()
    threads = await repo.list_threads(include_archived=include_archived)
    return [
        ThreadResponse(
            id=t.id,
            title=t.title,
            created_at=t.created_at,
            updated_at=t.updated_at,
            archived=t.archived,
            settings=t.settings,
            user_id=t.user_id
        )
        for t in threads
    ]


@app.post("/threads", response_model=ThreadResponse)
async def create_thread_endpoint(req: ThreadCreateRequest):
    """Create a new research conversation thread with optional auto-titling."""
    repo = ThreadRepo()
    thread = await repo.create_thread(
        title=req.title,
        prompt=req.prompt,
        settings=req.settings
    )
    return ThreadResponse(
        id=thread.id,
        title=thread.title,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
        archived=thread.archived,
        settings=thread.settings,
        user_id=thread.user_id
    )


@app.get("/threads/{thread_id}", response_model=ThreadResponse)
async def get_thread_endpoint(thread_id: str):
    """Retrieve details for a single thread."""
    repo = ThreadRepo()
    thread = await repo.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    return ThreadResponse(
        id=thread.id,
        title=thread.title,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
        archived=thread.archived,
        settings=thread.settings,
        user_id=thread.user_id
    )


@app.patch("/threads/{thread_id}", response_model=ThreadResponse)
async def update_thread_endpoint(thread_id: str, req: ThreadUpdateRequest):
    """Rename, archive/unarchive, or update settings of a thread."""
    repo = ThreadRepo()
    thread = await repo.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    if req.title is not None:
        thread = await repo.rename_thread(thread_id, req.title)
    if req.archived is not None:
        thread = await repo.archive_thread(thread_id, req.archived)
    if req.settings is not None:
        # Update settings if supplied
        async with repo.db_path or "data/app.db":
            from core.db.connection import get_db
            async with get_db(repo.db_path) as db:
                import json
                await db.execute(
                    "UPDATE threads SET settings_json = ? WHERE id = ?;",
                    (json.dumps(req.settings), thread_id)
                )
                await db.commit()
        thread = await repo.get_thread(thread_id)

    return ThreadResponse(
        id=thread.id,
        title=thread.title,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
        archived=thread.archived,
        settings=thread.settings,
        user_id=thread.user_id
    )


@app.delete("/threads/{thread_id}")
async def delete_thread_endpoint(thread_id: str):
    """Delete a thread, cascading to runs, messages, events, and checkpoints."""
    repo = ThreadRepo()
    deleted = await repo.delete_thread(thread_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"status": "deleted", "thread_id": thread_id}


@app.get("/threads/{thread_id}/messages", response_model=List[MessageResponse])
async def get_thread_messages_endpoint(thread_id: str):
    """Retrieve all messages in a thread in chronological order."""
    repo = MessageRepo()
    messages = await repo.list_messages(thread_id)
    return [
        MessageResponse(
            id=m.id,
            thread_id=m.thread_id,
            role=m.role,
            content=m.content,
            created_at=m.created_at,
            run_id=m.run_id,
            metadata=m.metadata
        )
        for m in messages
    ]


@app.get("/threads/{thread_id}/export", response_model=ThreadExportResponse)
async def export_thread_endpoint(thread_id: str):
    """Export the entire thread bundle including messages, runs, and telemetry events."""
    thread_repo = ThreadRepo()
    msg_repo = MessageRepo()
    run_repo = RunRepo()
    event_repo = EventRepo()

    thread = await thread_repo.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    messages = await msg_repo.list_messages(thread_id)
    runs = await run_repo.list_runs_by_thread(thread_id)

    all_events = []
    for r in runs:
        events = await event_repo.list_events_by_run(r.id)
        all_events.extend(events)

    return ThreadExportResponse(
        thread=ThreadResponse(
            id=thread.id,
            title=thread.title,
            created_at=thread.created_at,
            updated_at=thread.updated_at,
            archived=thread.archived,
            settings=thread.settings,
            user_id=thread.user_id
        ),
        messages=[
            MessageResponse(
                id=m.id,
                thread_id=m.thread_id,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
                run_id=m.run_id,
                metadata=m.metadata
            )
            for m in messages
        ],
        runs=[
            RunResponse(
                id=r.id,
                thread_id=r.thread_id,
                task=r.task,
                status=r.status,
                started_at=r.started_at,
                finished_at=r.finished_at,
                error=r.error,
                resume_after=r.resume_after,
                config=r.config,
                report_md=r.report_md,
                report_json=r.report_json,
                metrics=r.metrics,
                cancel_requested=r.cancel_requested
            )
            for r in runs
        ],
        events=[
            RunEventResponse(
                seq=e.seq,
                run_id=e.run_id,
                ts=e.ts,
                type=e.type,
                node=e.node,
                payload=e.payload
            )
            for e in all_events
        ]
    )


# ==============================================================================
# Runs & Checkpoint Resumption Endpoints
# ==============================================================================

@app.get("/threads/{thread_id}/runs", response_model=List[RunResponse])
async def list_thread_runs_endpoint(thread_id: str):
    """List all runs executed on a thread."""
    runs = await list_runs(thread_id)
    return [
        RunResponse(
            id=r.id,
            thread_id=r.thread_id,
            task=r.task,
            status=r.status,
            started_at=r.started_at,
            finished_at=r.finished_at,
            error=r.error,
            resume_after=r.resume_after,
            config=r.config,
            report_md=r.report_md,
            report_json=r.report_json,
            metrics=r.metrics,
            cancel_requested=r.cancel_requested
        )
        for r in runs
    ]


@app.post("/threads/{thread_id}/runs", response_model=RunResponse)
async def create_and_start_run_endpoint(
    thread_id: str,
    req: RunCreateRequest,
    background_tasks: BackgroundTasks
):
    """Launch a new research run on a thread."""
    try:
        run = await start_run(
            thread_id=thread_id,
            task=req.task,
            config={
                "max_retries": req.max_retries,
                "search_budget": req.max_search_budget,
                "model_name": req.model_name
            }
        )
    except ValueError as val_err:
        raise HTTPException(status_code=404, detail=str(val_err))

    # Execute in background task
    background_tasks.add_task(execute_run, run.id)

    return RunResponse(
        id=run.id,
        thread_id=run.thread_id,
        task=run.task,
        status="running",
        started_at=run.started_at,
        finished_at=run.finished_at,
        error=run.error,
        resume_after=run.resume_after,
        config=run.config,
        report_md=run.report_md,
        report_json=run.report_json,
        metrics=run.metrics,
        cancel_requested=run.cancel_requested
    )


@app.get("/runs/{run_id}", response_model=RunResponse)
async def get_run_endpoint(run_id: str):
    """Retrieve run record status and outputs."""
    run = await get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunResponse(
        id=run.id,
        thread_id=run.thread_id,
        task=run.task,
        status=run.status,
        started_at=run.started_at,
        finished_at=run.finished_at,
        error=run.error,
        resume_after=run.resume_after,
        config=run.config,
        report_md=run.report_md,
        report_json=run.report_json,
        metrics=run.metrics,
        cancel_requested=run.cancel_requested
    )


@app.post("/runs/{run_id}/resume", response_model=RunResponse)
async def resume_run_endpoint(
    run_id: str,
    req: ResumeRunRequest,
    background_tasks: BackgroundTasks
):
    """Resume an interrupted or quota-paused run from its last checkpoint."""
    run_repo = RunRepo()
    run = await run_repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status in ("completed", "cancelled"):
        raise HTTPException(status_code=400, detail=f"Run '{run_id}' is already {run.status} and cannot be resumed.")

    if run.status == "paused_quota" and run.resume_after and not req.force:
        from datetime import datetime, timezone
        try:
            resume_dt = datetime.fromisoformat(run.resume_after)
            if datetime.now(timezone.utc) < resume_dt:
                raise HTTPException(
                    status_code=400,
                    detail=f"Run is paused for daily quota until {run.resume_after}. Set force=True to resume immediately."
                )
        except HTTPException:
            raise
        except Exception:
            pass

    # Launch resumption in background
    background_tasks.add_task(resume_run, run_id, req.force)

    return RunResponse(
        id=run.id,
        thread_id=run.thread_id,
        task=run.task,
        status="running",
        started_at=run.started_at,
        finished_at=None,
        error=run.error,
        resume_after=run.resume_after,
        config=run.config,
        report_md=run.report_md,
        report_json=run.report_json,
        metrics=run.metrics,
        cancel_requested=False
    )


@app.post("/runs/{run_id}/cancel")
async def cancel_run_endpoint(run_id: str):
    """Request cooperative cancellation for an active run."""
    run = await get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    await cancel_run(run_id)
    return {"status": "cancel_requested", "run_id": run_id}


# ==============================================================================
# Legacy Endpoints (Maintained for Backward Compatibility)
# ==============================================================================

@app.post("/research/run", response_model=ResearchResponse)
async def run_research_sync(req: ResearchRequest):
    """Execute a research task synchronously through the LangGraph multi-agent loop."""
    task_id = str(uuid.uuid4())[:8]
    initial_state = create_initial_state(
        task=req.task,
        max_retries=req.max_retries,
        max_search_budget=req.max_search_budget,
        model_name=req.model_name or os.environ.get("GEMINI_MODEL", "")
    )

    final_state: AgentState = await graph.ainvoke(initial_state)
    TASKS[task_id] = {
        "task_id": task_id,
        "request": req.model_dump(),
        "state": final_state
    }

    last_critic = final_state.get("critic_evaluations", [])[-1] if final_state.get("critic_evaluations") else None
    critic_summary = None
    if last_critic:
        critic_summary = CriticSummarySchema(
            verdict=last_critic.verdict,
            total_score=last_critic.total_score,
            coverage=last_critic.coverage.score,
            evidence=last_critic.evidence.score,
            coherence=last_critic.coherence.score,
            feedback=last_critic.overall_feedback
        )

    subtask_summaries = [
        SubtaskSummarySchema(
            id=r.subtask_id,
            title=r.title,
            sources_count=len(r.sources)
        )
        for r in final_state.get("subtask_results", [])
    ]

    trace_summaries = [
        TraceEventSummarySchema(
            timestamp=e.timestamp,
            step=e.step,
            event_type=e.event_type,
            iteration=e.iteration,
            summary=e.summary
        )
        for e in final_state.get("trace_events", [])
    ]

    struct_obj = final_state.get("final_report_structured") or final_state.get("draft_report_structured")
    structured_dict = struct_obj.model_dump() if hasattr(struct_obj, "model_dump") else (struct_obj if isinstance(struct_obj, dict) else None)

    return ResearchResponse(
        task_id=task_id,
        task=req.task,
        status=final_state.get("status", "finished"),
        iterations=final_state.get("retry_count", 0),
        searches_performed=final_state.get("total_searches_performed", 0),
        cache_hits=final_state.get("cache_hits", 0),
        human_escalation_needed=final_state.get("human_escalation_needed", False),
        escalation_reason=final_state.get("escalation_reason"),
        mcp_saved_path=final_state.get("mcp_saved_path"),
        report_markdown=final_state.get("final_report"),
        report_metadata=final_state.get("report_metadata"),
        structured_report=structured_dict,
        critic_summary=critic_summary,
        subtasks=subtask_summaries,
        traces=trace_summaries
    )


async def _run_async_worker(task_id: str, req: ResearchRequest):
    start_time = time.time()
    initial_state = create_initial_state(
        task=req.task,
        max_retries=req.max_retries,
        max_search_budget=req.max_search_budget,
        model_name=req.model_name or os.environ.get("GEMINI_MODEL", "")
    )
    TASKS[task_id]["state"] = initial_state
    TASKS[task_id]["status"] = "running"
    TASKS[task_id]["start_time"] = start_time
    TASKS[task_id]["current_node"] = "planner"

    try:
        async for chunk in graph.astream(initial_state, stream_mode="updates"):
            for node_name, node_update in chunk.items():
                current_state = TASKS[task_id]["state"]
                current_state.update(node_update)
                TASKS[task_id]["current_node"] = node_name
                TASKS[task_id]["elapsed_seconds"] = round(time.time() - start_time, 2)

        TASKS[task_id]["status"] = TASKS[task_id]["state"].get("status", "finished")
        TASKS[task_id]["elapsed_seconds"] = round(time.time() - start_time, 2)
    except Exception as exc:
        TASKS[task_id]["status"] = "error"
        TASKS[task_id]["error"] = str(exc)
        TASKS[task_id]["elapsed_seconds"] = round(time.time() - start_time, 2)


@app.post("/research/start")
async def start_research_async(req: ResearchRequest, background_tasks: BackgroundTasks):
    """Launch a legacy asynchronous background research job (deprecated, use /threads/{id}/runs)."""
    task_id = str(uuid.uuid4())[:8]
    TASKS[task_id] = {
        "task_id": task_id,
        "request": req.model_dump(),
        "status": "pending",
        "current_node": "planner",
        "start_time": time.time(),
        "elapsed_seconds": 0.0,
        "state": {}
    }
    background_tasks.add_task(_run_async_worker, task_id, req)
    return {"task_id": task_id, "status": "started", "message": "Research task dispatched."}


@app.get("/research/{task_id}/status")
async def get_task_status(task_id: str):
    """Check status and intermediate progress of a research task."""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    task_info = TASKS[task_id]
    state = task_info.get("state", {})
    start_time = task_info.get("start_time")
    elapsed = round(time.time() - start_time, 2) if start_time else task_info.get("elapsed_seconds", 0.0)
    return {
        "task_id": task_id,
        "status": task_info.get("status", "unknown"),
        "current_node": task_info.get("current_node"),
        "elapsed_seconds": elapsed,
        "iteration": state.get("retry_count", 0),
        "searches_performed": state.get("total_searches_performed", 0),
        "cache_hits": state.get("cache_hits", 0),
        "trace_events_count": len(state.get("trace_events", []))
    }


@app.get("/research/{task_id}/trace")
async def get_task_trace(task_id: str):
    """Retrieve full observability trace events for detailed debugging."""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    state = TASKS[task_id].get("state", {})
    return {
        "task_id": task_id,
        "traces": [e.model_dump() if hasattr(e, "model_dump") else e for e in state.get("trace_events", [])]
    }


@app.get("/research/{task_id}/report")
async def get_task_report(task_id: str):
    """Download the final report and metadata."""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    state = TASKS[task_id].get("state", {})
    struct_obj = state.get("final_report_structured") or state.get("draft_report_structured")
    structured_dict = struct_obj.model_dump() if hasattr(struct_obj, "model_dump") else (struct_obj if isinstance(struct_obj, dict) else None)
    return {
        "task_id": task_id,
        "markdown": state.get("final_report"),
        "structured_report": structured_dict,
        "metadata": state.get("report_metadata"),
        "mcp_saved_path": state.get("mcp_saved_path")
    }


@app.get("/reports")
async def list_saved_reports():
    """List all reports stored on disk via MCP Filesystem server."""
    res = await mcp_client.call_tool("list_reports", {})
    return res.get("output", {})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
