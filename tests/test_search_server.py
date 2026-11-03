"""Unit tests for faster, more reliable search and provider ordering."""

import os
import json
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
