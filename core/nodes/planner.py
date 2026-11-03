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


@timed_node("planner")
async def planner_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node: Planner Agent.

    Decomposes the initial task or revises the plan based on Critic feedback.
    """
    if state.get("human_escalation_needed"):
        return {"status": "escalated"}

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

    return {
        "plan": new_plan,
        "status": "executing",
        "trace_events": trace_events
    }
