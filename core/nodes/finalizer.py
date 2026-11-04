from core.events import emit_run_event, get_run_event_bus
"""Finalizer Node for Research Agent Crew.

Appends the Critic audit scorecard, packages machine-readable JSON summaries,
and persists all final artifacts via the Model Context Protocol (MCP) filesystem server.
"""

import re
import json
from datetime import datetime, timezone
from typing import Dict, Any
from core.state import AgentState
from core.tracer import TraceLogger
from mcp_tools.client import MCPClientManager
from core.timing import timed_node
from core.cancellation import is_cancellation_requested, RunCancelledError


def _slugify(text: str) -> str:
    """Generate a clean filesystem slug from a task description."""
    clean = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[-\s]+", "_", clean)
    return slug[:40] if slug else "research_report"


@timed_node("finalizer")
async def finalizer_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node: Finalizer Agent.

    Enriches draft report with audit scorecard and persists artifacts using MCP.
    """
    run_id = state.get("run_id")
    if await is_cancellation_requested(run_id):
        raise RunCancelledError(f"Run {run_id} cancelled by user")

    is_resumed = bool(state.get("is_resume", False))
    await emit_run_event(run_id=run_id, event_type="node_started", node="finalizer", data={"node": "finalizer", "resumed": is_resumed})

    task = state["task"]
    draft = state.get("draft_report") or ""
    critic_evals = state.get("critic_evaluations", [])
    last_eval = critic_evals[-1] if critic_evals else None
    escalated = state.get("human_escalation_needed", False)
    escalation_reason = state.get("escalation_reason", "")
    subtask_results = state.get("subtask_results", [])
    total_searches = state.get("total_searches_performed", 0)
    retry_count = state.get("retry_count", 0)
    trace_events = list(state.get("trace_events", []))

    mcp_client = MCPClientManager()
    slug = _slugify(task)
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # 1. Build Critic Audit Scorecard
    scorecard_lines = [
        "",
        "---",
        "## Multi-Agent Quality Audit & Observability Scorecard",
        ""
    ]

    if escalated:
        scorecard_lines.extend([
            "> [!WARNING]",
            "> **Human Review Recommended**: The research agent crew completed the task with partial confidence.",
            f"> *Reason*: {escalation_reason}",
            ""
        ])

    if last_eval:
        scorecard_lines.extend([
            "| Evaluation Dimension | Score | Assessment Justification |",
            "| :--- | :---: | :--- |",
            f"| **Coverage** | `{last_eval.coverage.score}/5` | {last_eval.coverage.rationale} |",
            f"| **Evidence** | `{last_eval.evidence.score}/5` | {last_eval.evidence.rationale} |",
            f"| **Coherence** | `{last_eval.coherence.score}/5` | {last_eval.coherence.rationale} |",
            f"| **Final Verdict** | **`{last_eval.verdict}`** | Total: `{last_eval.total_score}/15` (Pass threshold: 12/15) |",
            "",
            f"**Synthesis Feedback**: {last_eval.overall_feedback}",
            ""
        ])

    cache_hits = state.get("cache_hits", 0)

    scorecard_lines.extend([
        f"- **Reflection Loops Completed**: {retry_count}",
        f"- **MCP Searches Executed**: {total_searches}",
        f"- **Cache Hits Reused**: {cache_hits}",
        f"- **Generated At**: {datetime.now(timezone.utc).isoformat()}",
        ""
    ])

    final_report_content = draft + "\n" + "\n".join(scorecard_lines)

    # 2. Build structured JSON summary
    all_sources = []
    for r in subtask_results:
        for s in r.sources:
            all_sources.append({
                "title": s.title,
                "url": s.url,
                "snippet": s.snippet
            })

    json_summary = {
        "task": task,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "escalated" if escalated else "completed",
        "iterations": retry_count,
        "searches_performed": total_searches,
        "cache_hits": cache_hits,
        "critic_evaluation": {
            "verdict": last_eval.verdict if last_eval else "UNCHECKED",
            "total_score": last_eval.total_score if last_eval else 0,
            "coverage": last_eval.coverage.score if last_eval else 0,
            "evidence": last_eval.evidence.score if last_eval else 0,
            "coherence": last_eval.coherence.score if last_eval else 0,
        } if last_eval else {},
        "subtasks": [
            {
                "id": r.subtask_id,
                "title": r.title,
                "sources_count": len(r.sources)
            }
            for r in subtask_results
        ],
        "sources": all_sources,
        "timings": {
            "nodes": (state.get("timings") or {}).get("nodes", {}),
            "total_node_time_ms": round(sum(((state.get("timings") or {}).get("nodes", {})).values()), 2),
            "tool_calls_count": len((state.get("timings") or {}).get("tool_calls", []))
        }
    }

    # 3. Persist Markdown report via MCP Filesystem Tool
    md_filename = f"{slug}_{timestamp_str}.md"
    save_md_res = await mcp_client.call_tool(
        "save_report",
        arguments={
            "filename": md_filename,
            "content": final_report_content,
            "folder": "reports"
        }
    )

    # 4. Persist JSON summary via MCP Filesystem Tool
    json_filename = f"{slug}_{timestamp_str}_summary.json"
    await mcp_client.call_tool(
        "save_report",
        arguments={
            "filename": json_filename,
            "content": json.dumps(json_summary, indent=2),
            "folder": "reports"
        }
    )

    # 5. Persist Structured Report JSON via MCP Filesystem Tool
    draft_structured = state.get("draft_report_structured")
    structured_filename = f"{slug}_{timestamp_str}_structured.json"
    if draft_structured:
        if hasattr(draft_structured, "model_dump_json"):
            structured_content = draft_structured.model_dump_json(indent=2)
        else:
            structured_content = json.dumps(draft_structured, indent=2)
    else:
        structured_content = "{}"

    await mcp_client.call_tool(
        "save_report",
        arguments={
            "filename": structured_filename,
            "content": structured_content,
            "folder": "reports"
        }
    )

    saved_path = save_md_res.get("output", {}).get("filepath", f"reports/{md_filename}")

    # 6. Log trace event
    trace_events.append(
        TraceLogger.create_event(
            step="finalizer",
            event_type="artifact_persistence",
            summary=f"Persisted report to {md_filename} via MCP Filesystem server",
            details={
                "markdown_file": md_filename,
                "json_file": json_filename,
                "structured_file": structured_filename,
                "saved_path": saved_path,
                "bytes": save_md_res.get("output", {}).get("bytes_written", 0)
            },
            iteration=retry_count
        )
    )

    await emit_run_event(
        run_id=run_id,
        event_type="node_finished",
        node="finalizer",
        data={"node": "finalizer", "status": "finished", "saved_path": saved_path}
    )
    if run_id:
        await get_run_event_bus(run_id).flush()

    return {
        "final_report": final_report_content,
        "final_report_structured": draft_structured,
        "report_metadata": json_summary,
        "mcp_saved_path": saved_path,
        "status": "finished",
        "trace_events": trace_events
    }
