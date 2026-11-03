"""FastAPI Service for Research Agent Crew.

Exposes REST and SSE endpoints to launch research workflows, stream live agent
telemetry, inspect MCP tool registries, and retrieve generated reports.
"""

import os
import time
import uuid
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ResearchRequest, ResearchResponse, CriticSummarySchema,
    SubtaskSummarySchema, TraceEventSummarySchema
)
from core.graph import build_research_graph, create_initial_state
from core.state import AgentState
from mcp_tools.client import MCPClientManager

app = FastAPI(
    title="Research Agent Crew API",
    description="Multi-Agent Planner/Executor/Critic Research Service with Model Context Protocol (MCP) Tools",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for task runs
TASKS: Dict[str, Dict[str, Any]] = {}
mcp_client = MCPClientManager()
graph = build_research_graph()


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
    """Launch an asynchronous background research job."""
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
