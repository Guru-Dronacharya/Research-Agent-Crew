from core.events import emit_run_event, get_run_event_bus
"""Synthesizer Agent Node for Research Agent Crew.

Compiles subtask findings and verified evidence into a cohesive, structured
research report ready for multidimensional Critic evaluation.
"""

from typing import Dict, Any
from core.state import AgentState, StructuredReport
from core.report_render import render_structured_report_to_markdown
from core.llm import get_synth_llm
from core.tracer import TraceLogger
from core.timing import timed_node
from core.ratelimit import QuotaExceededError
from core.cancellation import is_cancellation_requested, RunCancelledError


@timed_node("synthesizer")
async def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node: Synthesizer Agent.

    Synthesizes raw subtask findings and citations into a structured draft report.
    """
    if state.get("human_escalation_needed"):
        return {"status": "escalated"}

    run_id = state.get("run_id")
    if await is_cancellation_requested(run_id):
        raise RunCancelledError(f"Run {run_id} cancelled by user")

    is_resumed = bool(state.get("is_resume", False))
    await emit_run_event(run_id=run_id, event_type="node_started", node="synthesizer", data={"node": "synthesizer", "resumed": is_resumed})

    task = state["task"]
    plan = state["plan"]
    subtask_results = state.get("subtask_results", [])
    iteration = state.get("retry_count", 0)
    critic_evals = state.get("critic_evaluations", [])
    last_critic_eval = critic_evals[-1] if critic_evals else None

    model_name = state.get("model_name")
    llm = get_synth_llm(model_name=model_name)

    try:
        synth_output = await llm.synthesize_findings(
            task=task,
            plan=plan,
            subtask_results=subtask_results,
            critic_evaluation=last_critic_eval,
            iteration=iteration
        )
        if isinstance(synth_output, tuple):
            draft_report, draft_report_structured = synth_output
        elif isinstance(synth_output, StructuredReport):
            draft_report_structured = synth_output
            all_srcs = [s for r in subtask_results for s in r.sources]
            draft_report = render_structured_report_to_markdown(draft_report_structured, all_srcs)
        else:
            draft_report = str(synth_output)
            draft_report_structured = None
    except QuotaExceededError as exc:
        if state.get("run_id"):
            raise
        trace_events = list(state.get("trace_events", []))
        escalation_reason = f"Daily LLM quota exceeded: {str(exc)}"
        trace_events.append(
            TraceLogger.log_escalation(
                reason=escalation_reason,
                retry_count=iteration,
                iteration=iteration
            )
        )
        return {
            "status": "escalated",
            "human_escalation_needed": True,
            "escalation_reason": escalation_reason,
            "draft_report": f"# Research Escalation\n\n> [!WARNING]\n> {escalation_reason}\n",
            "draft_report_structured": None,
            "trace_events": trace_events
        }

    trace_events = list(state.get("trace_events", []))
    trace_events.append(
        TraceLogger.create_event(
            step="synthesizer",
            event_type="draft_synthesis",
            summary=f"Synthesized draft report ({len(draft_report.split())} words, {len(subtask_results)} subtasks)",
            details={
                "word_count": len(draft_report.split()),
                "subtasks_included": len(subtask_results),
                "has_structured_report": draft_report_structured is not None,
                "iteration": iteration
            },
            iteration=iteration
        )
    )

    if draft_report_structured and getattr(draft_report_structured, "sections", None):
        for sec in draft_report_structured.sections:
            await emit_run_event(
                run_id=run_id,
                event_type="report_section",
                node="synthesizer",
                subtask_id=sec.subtask_id,
                data={
                    "heading": sec.heading,
                    "summary": sec.summary,
                    "claims_count": len(sec.claims),
                    "claims": [c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in sec.claims]
                }
            )

    await emit_run_event(
        run_id=run_id,
        event_type="node_finished",
        node="synthesizer",
        data={
            "node": "synthesizer",
            "word_count": len(draft_report.split()),
            "sections_count": len(draft_report_structured.sections) if draft_report_structured else 0
        }
    )
    if run_id:
        await get_run_event_bus(run_id).flush()

    return {
        "draft_report": draft_report,
        "draft_report_structured": draft_report_structured,
        "status": "evaluating",
        "trace_events": trace_events
    }
