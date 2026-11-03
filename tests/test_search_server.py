"""Unit tests for faster, more reliable search and provider ordering."""

import os
import json
import pytest
from unittest.mock import patch
from mcp_tools.servers.search_server import (
    search_web,
    filter_relevant_search_results,
)


def test_provider_order_tavily_succeeds_ddgs_never_called():
    """When TAVILY_API_KEY is present and succeeds, ddgs must NEVER be called."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "mock_tavily_key"}, clear=False), \
         patch("mcp_tools.servers.search_server._search_tavily") as mock_tavily, \
         patch("mcp_tools.servers.search_server._search_brave") as mock_brave, \
         patch("mcp_tools.servers.search_server._search_ddg") as mock_ddg:

        mock_tavily.return_value = [
            {"title": "Tavily LangGraph Guide", "url": "https://example.com/lg", "snippet": "LangGraph tutorial and overview"}
        ]

        raw_res = search_web("LangGraph overview", max_results=5)
        res = json.loads(raw_res)

        assert res["status"] == "success"
        assert res["engine"] == "tavily"
        assert len(res["results"]) == 1
        mock_tavily.assert_called_once()
        mock_brave.assert_not_called()
        mock_ddg.assert_not_called()


def test_provider_order_brave_succeeds_ddgs_never_called():
    """When Tavily is absent but BRAVE_API_KEY is present and succeeds, ddgs is not called."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "", "BRAVE_API_KEY": "mock_brave_key"}, clear=False), \
         patch("mcp_tools.servers.search_server._search_brave") as mock_brave, \
         patch("mcp_tools.servers.search_server._search_ddg") as mock_ddg:

        mock_brave.return_value = [
            {"title": "Brave LangGraph Guide", "url": "https://example.com/brave_lg", "snippet": "Brave LangGraph documentation"}
        ]

        raw_res = search_web("LangGraph documentation", max_results=5)
        res = json.loads(raw_res)

        assert res["status"] == "success"
        assert res["engine"] == "brave"
        mock_brave.assert_called_once()
        mock_ddg.assert_not_called()


def test_provider_order_fallback_to_ddgs_when_tavily_fails():
    """When Tavily fails, fallback to ddgs (when Brave key is absent)."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "bad_key", "BRAVE_API_KEY": ""}, clear=False), \
         patch("mcp_tools.servers.search_server._search_tavily", side_effect=Exception("401 Unauthorized")), \
         patch("mcp_tools.servers.search_server._search_ddg") as mock_ddg:

        mock_ddg.return_value = [
            {"title": "DDG LangGraph Overview", "url": "https://example.com/ddg", "snippet": "DDG LangGraph architecture notes"}
        ]

        raw_res = search_web("LangGraph architecture", max_results=5)
        res = json.loads(raw_res)

        assert res["status"] == "success"
        assert res["engine"] == "duckduckgo"
        mock_ddg.assert_called_once()


def test_rate_limited_fast_failure():
    """429 / Ratelimit returns status 'rate_limited'."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "", "BRAVE_API_KEY": ""}, clear=False), \
         patch("mcp_tools.servers.search_server._search_ddg", side_effect=Exception("429 Too Many Requests Ratelimit")):

        raw_res = search_web("LangGraph architecture", max_results=5)
        res = json.loads(raw_res)

        assert res["status"] == "rate_limited"
        assert res["results"] == []


def test_timeout_fast_failure():
    """Timeout returns status 'timeout'."""
    with patch.dict(os.environ, {"TAVILY_API_KEY": "", "BRAVE_API_KEY": ""}, clear=False), \
         patch("mcp_tools.servers.search_server._search_ddg", side_effect=TimeoutError("timed out after 6s")):

        raw_res = search_web("LangGraph architecture", max_results=5)
        res = json.loads(raw_res)

        assert res["status"] == "timeout"
        assert res["results"] == []


def test_relevance_prefilter_drops_spam_and_non_english():
    """Relevance prefilter drops non-English and spam results, keeping valid ones."""
    query = "LangGraph multi-agent architecture"
    candidates = [
        # Valid relevant result
        {
            "title": "LangGraph: Multi-Agent Architecture Guide",
            "url": "https://langchain.ai/docs/langgraph",
            "snippet": "Learn about multi-agent architecture patterns using LangGraph and state machines."
        },
        # Obvious emulator/ROM spam
        {
            "title": "Download GameBoy ROMs & GBA Emulator",
            "url": "https://coolroms-emulator.net/download",
            "snippet": "Free game roms and emulator download for android and pc."
        },
        # Non-English result (Chinese script)
        {
            "title": "这是一个关于智能体架构的测试标题",
            "url": "https://example.cn/agents",
            "snippet": "中文多智能体系统的设计与实践，详细介绍各种架构模式与优化方法。"
        },
        # Zero keyword match off-topic result
        {
            "title": "Best Italian Pizza Dough Recipe",
            "url": "https://cooking.com/pizza-dough",
            "snippet": "Traditional sourdough pizza crust recipe with olive oil and yeast."
        }
    ]

    filtered = filter_relevant_search_results(candidates, query)
    assert len(filtered) == 1
    assert filtered[0]["url"] == "https://langchain.ai/docs/langgraph"


def test_ddg_exception_no_results_found_maps_to_no_results():
    """DDGSException('No results found.') must map cleanly to status 'no_results' instead of 'error'."""
    from ddgs.exceptions import DDGSException

    with patch.dict(os.environ, {"TAVILY_API_KEY": "", "BRAVE_API_KEY": ""}, clear=False), \
         patch("mcp_tools.servers.search_server._search_ddg", side_effect=DDGSException("No results found.")):

        raw_res = search_web("Rare obscure framework query", max_results=5)
        res = json.loads(raw_res)

        assert res["status"] == "no_results"
        assert res["results"] == []
        assert "No web search results found" in res.get("message", "")


def test_search_error_surfaces_real_exception_message():
    """Unexpected search exceptions must surface the actual error string in message."""
    distinct_err_msg = "TLS certificate validation failed for search gateway"
    with patch.dict(os.environ, {"TAVILY_API_KEY": "", "BRAVE_API_KEY": ""}, clear=False), \
         patch("mcp_tools.servers.search_server._search_ddg", side_effect=RuntimeError(distinct_err_msg)):

        raw_res = search_web("LangGraph architecture", max_results=5)
        res = json.loads(raw_res)

        assert res["status"] == "error"
        assert distinct_err_msg in res.get("message", "")


@pytest.mark.asyncio
async def test_executor_records_real_exception_message_in_trace_events():
    """Executor must log the actual search error message in tool result trace events."""
    from core.nodes.executor import executor_node
    from core.state import Subtask, ResearchPlan, AgentState
    from mcp_tools.client import set_global_tool, reset_global_tools

    error_detail = "Backend socket timed out during TLS handshake"

    def fake_error_search(query: str, max_results: int = 5, **kwargs):
        return json.dumps({
            "status": "error",
            "query": query,
            "message": error_detail,
            "results": []
        })

    set_global_tool("search_web", fake_error_search)

    try:
        state: AgentState = {
            "task": "Test Trace Error Logging",
            "plan": ResearchPlan(
                task_summary="Summary",
                subtasks=[
                    Subtask(id="s1", title="Subtask 1", description="desc", search_queries=["failing query"], rationale="r")
                ],
                key_hypotheses=[]
            ),
            "subtask_results": [],
            "draft_report": None,
            "draft_report_structured": None,
            "final_report": None,
            "final_report_structured": None,
            "report_metadata": None,
            "critic_evaluations": [],
            "retry_count": 0,
            "max_retries": 1,
            "search_budget_remaining": 5,
            "max_search_budget": 5,
            "initial_budget_remaining": 3,
            "revision_reserve_remaining": 2,
            "iteration_budget_remaining": 3,
            "total_searches_performed": 0,
            "cache_hits": 0,
            "trace_events": [],
            "human_escalation_needed": False,
            "escalation_reason": None,
            "status": "executing",
            "mcp_saved_path": None,
            "model_name": "",
            "timings": {}
        }

        result = await executor_node(state)
        traces = result.get("trace_events", [])
        tool_results = [e for e in traces if e.event_type == "tool_result" and e.details.get("tool_name") == "search_web"]
        assert len(tool_results) >= 1
        # Real exception message must be present in the tool result details
        assert any(error_detail in t.details.get("result_summary", "") for t in tool_results)
    finally:
        reset_global_tools()

