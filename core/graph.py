"""LangGraph StateGraph orchestration for Research Agent Crew.

Defines the multi-agent execution pipeline:
Planner -> Executor (MCP Tools) -> Synthesizer -> Critic (Rubric Reflection) -> Conditional Loop -> Finalizer -> End.
Includes explicit budget allocation across initial and revision rounds.
"""

import os
from typing import Literal, Optional, Any
from langgraph.graph import StateGraph, START, END
from core.state import AgentState
from core.nodes.planner import planner_node
from core.nodes.executor import executor_node
from core.nodes.synthesizer import synthesizer_node
from core.nodes.critic import critic_node
from core.nodes.finalizer import finalizer_node


def should_continue(state: AgentState) -> Literal["planner", "finalizer"]:
    """Conditional routing edge following Critic evaluation.

    Routes back to Planner if Critic requested revisions (and retry limit not exceeded
    and revision reserve remaining), or proceeds to Finalizer if report PASSED or was
    escalated.
    """
    if state.get("human_escalation_needed"):
        return "finalizer"
    status = state.get("status", "completed")
    if status == "revising":
        return "planner"
    return "finalizer"


def create_initial_state(
    task: str,
    max_retries: int = 2,
    max_search_budget: int = 10,
    model_name: Optional[str] = None,
    run_id: Optional[str] = None
) -> AgentState:
    """Helper to initialize a clean AgentState with split budget allocation."""
    resolved_model = model_name or ""
    try:
        initial_fraction = float(os.environ.get("INITIAL_BUDGET_FRACTION", "0.6"))
    except ValueError:
        initial_fraction = 0.6

    total_budget = max(0, max_search_budget)
    if max_retries > 0 and total_budget > 1:
        initial_alloc = int(round(total_budget * initial_fraction))
        if initial_alloc >= total_budget:
            initial_alloc = total_budget - 1
        initial_alloc = max(1, initial_alloc)
        revision_reserve = total_budget - initial_alloc
    elif max_retries > 0 and total_budget == 1:
        initial_alloc = 1
        revision_reserve = 0
    else:
        initial_alloc = total_budget
        revision_reserve = 0

    return {
        "task": task,
        "plan": None,
        "subtask_results": [],
        "draft_report": None,
        "draft_report_structured": None,
        "final_report": None,
        "final_report_structured": None,
        "report_metadata": None,
        "critic_evaluations": [],
        "retry_count": 0,
        "max_retries": max_retries,
        "search_budget_remaining": total_budget,
        "max_search_budget": total_budget,
        "initial_budget_remaining": initial_alloc,
        "revision_reserve_remaining": revision_reserve,
        "iteration_budget_remaining": initial_alloc,
        "total_searches_performed": 0,
        "cache_hits": 0,
        "trace_events": [],
        "human_escalation_needed": False,
        "escalation_reason": None,
        "status": "planning",
        "mcp_saved_path": None,
        "model_name": resolved_model,
        "run_id": run_id,
        "timings": {
            "nodes": {},
            "llm_calls": [],
            "tool_calls": [],
            "node_events": [],
            "total_wall_clock_ms": 0.0
        }
    }


def build_research_graph(checkpointer: Optional[Any] = None):
    """Build and compile the LangGraph research workflow with optional checkpointer."""
    workflow = StateGraph(AgentState)

    # Add agent nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("finalizer", finalizer_node)

    # Connect linear edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "synthesizer")
    workflow.add_edge("synthesizer", "critic")

    # Connect reflection conditional edge
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "planner": "planner",
            "finalizer": "finalizer"
        }
    )

    # Terminate workflow
    workflow.add_edge("finalizer", END)

    return workflow.compile(checkpointer=checkpointer)


# Singleton compiled graph instance
research_graph = build_research_graph()
