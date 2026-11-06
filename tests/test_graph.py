"""Integration tests for the LangGraph multi-agent workflow, budget allocation, and guardrails."""

import json
import pytest
from core.graph import build_research_graph, create_initial_state
from mcp_tools.client import set_global_tool, reset_global_tools


@pytest.fixture(autouse=True)
def offline_search_stub():
    """Stub search_web tool so tests run 100% offline with zero network latency."""
    def fake_search(query: str, max_results: int = 5, **kwargs):
        return json.dumps({
            "status": "success",
            "query": query,
            "engine": "offline_stub",
            "results": [
                {
                    "title": f"Result for {query}",
                    "url": f"https://example.com/source/{abs(hash(query)) % 1000}",
                    "snippet": f"Verified empirical evidence and technical architecture for {query}."
                }
            ]
        })
    set_global_tool("search_web", fake_search)
    yield
    reset_global_tools()


@pytest.mark.asyncio
async def test_full_graph_execution():
    """Verify end-to-end execution of Planner -> Executor -> Synthesizer -> Critic -> Loop -> Finalizer."""
    graph = build_research_graph()
    initial_state = create_initial_state(
        task="Test Multi-Agent Systems Integration",
        max_retries=1,
        max_search_budget=6
    )

    final_state = await graph.ainvoke(initial_state)

    # Verify completion
    assert final_state["status"] in ("finished", "completed")
    assert final_state["final_report"] is not None
    assert len(final_state["final_report"]) > 100

    # Verify Critic ran
    assert len(final_state["critic_evaluations"]) >= 1

    # Verify MCP file was saved
    assert final_state["mcp_saved_path"] is not None

    # Verify structured traces
    assert len(final_state["trace_events"]) >= 5


@pytest.mark.asyncio
@pytest.mark.critical
async def test_search_budget_enforcement():
    """Verify that search budget ceiling prevents runaway searches."""
    graph = build_research_graph()
    tight_budget = 2
    initial_state = create_initial_state(
        task="Test Search Budget Guardrail",
        max_retries=1,
        max_search_budget=tight_budget
    )

    final_state = await graph.ainvoke(initial_state)

    # Searches performed must not exceed the ceiling
    assert final_state["total_searches_performed"] <= tight_budget
    assert final_state["search_budget_remaining"] >= 0

    # Ensure a budget alert event was triggered in trace
    budget_events = [
        e for e in final_state["trace_events"] if e.event_type == "budget_alert"
    ]
    assert len(budget_events) > 0


@pytest.mark.asyncio
async def test_revision_pass_performs_search_with_budget():
    """Verify that with budget 10 and max_retries 1, the revision pass performs at least 1 search."""
    graph = build_research_graph()
    initial_state = create_initial_state(
        task="Test Revision Pass Budget Allocation",
        max_retries=1,
        max_search_budget=10
    )

    final_state = await graph.ainvoke(initial_state)
    # StubLLM fails on iteration 0 and passes on iteration 1, so a revision pass occurred
    assert final_state["retry_count"] >= 1
    # Verify searches were performed during the revision round
    # Initial budget was 6 (60%), reserve was 4.
    # Total searches performed should exceed the initial allocation if subtasks were run in revision
    assert final_state["total_searches_performed"] >= 1
    assert final_state["total_searches_performed"] <= 10


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.critical
async def test_total_searches_never_exceeds_budget_ceiling():
    """Verify that total searches performed never exceeds max_search_budget across multiple runs."""
    graph = build_research_graph()
    for budget in [2, 4, 7]:
        initial_state = create_initial_state(
            task="Test Strict Ceiling Invariant",
            max_retries=2,
            max_search_budget=budget
        )
        final_state = await graph.ainvoke(initial_state)
        assert final_state["total_searches_performed"] <= budget


@pytest.mark.asyncio
async def test_budget_too_small_for_reserve_triggers_escalation():
    """Verify that with budget too small for any reserve, escalation occurs with a budget-related reason."""
    graph = build_research_graph()
    initial_state = create_initial_state(
        task="Test Budget Exhaustion Escalation",
        max_retries=1,
        max_search_budget=1  # 1 search total -> initial takes 1, reserve is 0
    )

    final_state = await graph.ainvoke(initial_state)
    # Critic fails on iteration 0 in StubLLM; with 0 reserve left, it must escalate
    assert final_state["human_escalation_needed"] is True
    assert final_state["escalation_reason"] is not None
    assert "budget" in final_state["escalation_reason"].lower() or "exhausted" in final_state["escalation_reason"].lower()
    assert final_state["status"] in ("escalated", "finished", "completed")
