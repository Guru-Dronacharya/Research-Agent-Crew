"""Unit tests for concurrent executor execution, budget safety, and failure isolation."""

import time
import pytest
from core.state import AgentState, ResearchPlan, Subtask
from core.nodes.executor import executor_node
from mcp_tools.client import MCPClientManager


@pytest.mark.slow
@pytest.mark.asyncio
async def test_executor_parallel_speedup():
    """Verify that two subtasks with 1.0s delay each run concurrently in ~1.0s total, not ~2.0s."""
    client = MCPClientManager()

    def slow_search(query: str, max_results: int = 3, timeout_sec: float = 10.0):
        time.sleep(1.0)
        return {
            "status": "success",
            "results": [{"title": f"Result for {query}", "url": f"https://example.com/{query}", "snippet": "data"}]
        }

    client.register_tool("search_web", slow_search)

    plan = ResearchPlan(
        task_summary="Parallel Speedup Test",
        subtasks=[
            Subtask(id="s1", title="Topic 1", description="", search_queries=["q1"], rationale=""),
            Subtask(id="s2", title="Topic 2", description="", search_queries=["q2"], rationale="")
        ]
    )

    state: AgentState = {
        "task": "Parallel Speedup Test",
        "plan": plan,
        "subtask_results": [],
        "draft_report": None,
        "final_report": None,
        "report_metadata": None,
        "critic_evaluations": [],
        "retry_count": 0,
        "max_retries": 1,
        "search_budget_remaining": 4,
        "max_search_budget": 4,
        "initial_budget_remaining": 4,
        "revision_reserve_remaining": 0,
        "iteration_budget_remaining": 4,
        "total_searches_performed": 0,
        "trace_events": [],
        "human_escalation_needed": False,
        "escalation_reason": None,
        "status": "executing",
        "mcp_saved_path": None,
        "model_name": "gemini-2.0-flash",
        "timings": {"nodes": {}, "tool_calls": []}
    }

    t0 = time.perf_counter()
    res = await executor_node(state)
    elapsed = time.perf_counter() - t0

    assert len(res["subtask_results"]) == 2
    assert elapsed < 1.8  # Well under 2.0s sequential execution


@pytest.mark.asyncio
async def test_executor_failing_subtask_does_not_abort_others():
    """Verify a failing subtask yields evidence_gap=True and other subtasks still complete."""
    client = MCPClientManager()

    def buggy_search(query: str, max_results: int = 3, timeout_sec: float = 10.0):
        if "bad" in query:
            raise RuntimeError("Simulated search failure")
        return {
            "status": "success",
            "results": [{"title": f"Result for {query}", "url": f"https://example.com/{query}", "snippet": "ok"}]
        }

    client.register_tool("search_web", buggy_search)

    plan = ResearchPlan(
        task_summary="Failure Isolation Test",
        subtasks=[
            Subtask(id="s_bad", title="Bad Topic", description="", search_queries=["bad query"], rationale=""),
            Subtask(id="s_good", title="Good Topic", description="", search_queries=["good query"], rationale="")
        ]
    )

    state: AgentState = {
        "task": "Failure Isolation Test",
        "plan": plan,
        "subtask_results": [],
        "draft_report": None,
        "final_report": None,
        "report_metadata": None,
        "critic_evaluations": [],
        "retry_count": 0,
        "max_retries": 1,
        "search_budget_remaining": 4,
        "max_search_budget": 4,
        "initial_budget_remaining": 4,
        "revision_reserve_remaining": 0,
        "iteration_budget_remaining": 4,
        "total_searches_performed": 0,
        "trace_events": [],
        "human_escalation_needed": False,
        "escalation_reason": None,
        "status": "executing",
        "mcp_saved_path": None,
        "model_name": "gemini-2.0-flash",
        "timings": {"nodes": {}, "tool_calls": []}
    }

    res = await executor_node(state)
    results = {r.subtask_id: r for r in res["subtask_results"]}

    assert len(results) == 2
    assert results["s_bad"].evidence_gap is True
    assert "Error" in results["s_bad"].findings or "No evidence" in results["s_bad"].findings
    assert results["s_good"].evidence_gap is False
    assert len(results["s_good"].sources) == 1


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.critical
async def test_executor_strict_budget_ceiling_under_parallelism():
    """Verify that total searches performed never exceeds max_search_budget and shares are respected."""
    client = MCPClientManager()

    call_count = 0

    def counting_search(query: str, max_results: int = 3, timeout_sec: float = 10.0):
        nonlocal call_count
        call_count += 1
        return {
            "status": "success",
            "results": [{"title": "Title", "url": "https://example.com", "snippet": "Text"}]
        }

    client.register_tool("search_web", counting_search)

    for budget in [1, 2, 3, 5, 7]:
        call_count = 0
        plan = ResearchPlan(
            task_summary="Budget Test",
            subtasks=[
                Subtask(id=f"s{i}", title=f"T{i}", description="", search_queries=[f"q{i}_a", f"q{i}_b"], rationale="")
                for i in range(5)
            ]
        )

        state: AgentState = {
            "task": "Budget Test",
            "plan": plan,
            "subtask_results": [],
            "draft_report": None,
            "final_report": None,
            "report_metadata": None,
            "critic_evaluations": [],
            "retry_count": 0,
            "max_retries": 1,
            "search_budget_remaining": budget,
            "max_search_budget": budget,
            "initial_budget_remaining": budget,
            "revision_reserve_remaining": 0,
            "iteration_budget_remaining": budget,
            "total_searches_performed": 0,
            "trace_events": [],
            "human_escalation_needed": False,
            "escalation_reason": None,
            "status": "executing",
            "mcp_saved_path": None,
            "model_name": "gemini-2.0-flash",
            "timings": {"nodes": {}, "tool_calls": []}
        }

        res = await executor_node(state)
        assert res["total_searches_performed"] <= budget
        assert call_count <= budget
