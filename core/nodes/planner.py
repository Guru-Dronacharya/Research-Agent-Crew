from core.events import emit_run_event, get_run_event_bus
"""Planner Agent Node for Research Agent Crew.

Decomposes the master research task into targeted subtasks, sets initial hypotheses,
and generates reflection-guided revisions when instructed by the Critic.
"""

from typing import Dict, Any
from core.state import AgentState
from core.llm import get_planner_llm
from core.tracer import TraceLogger
from core.timing import timed_node
from core.ratelimit import QuotaExceededError
from core.cancellation import is_cancellation_requested, RunCancelledError


@timed_node("planner")
async def planner_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node: Planner Agent.

    Decomposes the initial task or revises the plan based on Critic feedback.
    """
    if state.get("human_escalation_needed"):
        return {"status": "escalated"}

    run_id = state.get("run_id")
    if await is_cancellation_requested(run_id):
        raise RunCancelledError(f"Run {run_id} cancelled by user")

    is_resumed = bool(state.get("is_resume", False))
    await emit_run_event(run_id=run_id, event_type="node_started", node="planner", data={"node": "planner", "resumed": is_resumed})

    task = state["task"]
    iteration = state.get("retry_count", 0)
    critic_evals = state.get("critic_evaluations", [])
    last_critic_eval = critic_evals[-1] if critic_evals else None

    model_name = state.get("model_name")
    llm = get_planner_llm(model_name=model_name)
    subtask_results = state.get("subtask_results", [])

    try:
        new_plan = await llm.generate_plan(
            task=task,
            critic_evaluation=last_critic_eval,
            iteration=iteration,
            subtask_results=subtask_results
        )
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
            "trace_events": trace_events
        }

    # Log telemetry
    trace_events = list(state.get("trace_events", []))
    if iteration == 0 or not last_critic_eval:
        event = TraceLogger.log_planner_decomposition(
            task=task,
            subtasks=new_plan.subtasks,
            hypotheses=new_plan.key_hypotheses,
            iteration=iteration
        )
    else:
        event = TraceLogger.log_planner_revision(
            task=task,
            subtasks=new_plan.subtasks,
            revision_notes=new_plan.revision_notes or "Revised subtasks according to Critic directives",
            critic_guidance=last_critic_eval.next_step_instructions,
            iteration=iteration
        )
    trace_events.append(event)

    await emit_run_event(
        run_id=run_id,
        event_type="decomposition",
        node="planner",
        data={
            "task": task,
            "subtasks": [s.model_dump() for s in new_plan.subtasks],
            "hypotheses": getattr(new_plan, "key_hypotheses", []) or getattr(new_plan, "hypotheses", []),
            "iteration": iteration
        }
    )

    await emit_run_event(
        run_id=run_id,
        event_type="node_finished",
        node="planner",
        data={
            "node": "planner",
            "subtasks_count": len(new_plan.subtasks),
            "estimated_searches": new_plan.estimated_searches,
            "iteration": iteration
        }
    )
    if run_id:
        await get_run_event_bus(run_id).flush()

    return {
        "plan": new_plan,
        "status": "executing",
        "trace_events": trace_events
    }
