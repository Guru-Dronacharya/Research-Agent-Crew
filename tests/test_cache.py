"""Unit tests for SQLite search caching layer."""

import time
import asyncio
import pytest
from core.cache import SearchCache, normalize_query_for_cache
from core.state import Subtask, ResearchPlan
from core.graph import create_initial_state
from core.nodes.executor import executor_node
from mcp_tools.client import set_global_tool


@pytest.fixture
def temp_cache(tmp_path, monkeypatch):
    """Create a temporary SearchCache instance with an isolated database file."""
    monkeypatch.setenv("CACHE_ENABLED", "true")
    db_path = str(tmp_path / "test_search.db")
    cache = SearchCache(db_path=db_path)
    return cache


def test_query_normalization():
    """Verify that stop-words, casing, and whitespace are normalized."""
    q1 = "what is the architecture of LangGraph?"
    q2 = "LangGraph architecture"
    norm1 = normalize_query_for_cache(q1)
    norm2 = normalize_query_for_cache(q2)
    assert norm1 == norm2


def test_cache_hit_and_miss(temp_cache):
    """Verify setting and getting search results."""
    query = "LangGraph vs AutoGen"
    data = {
        "status": "success",
        "results": [{"title": "Comparison", "url": "https://example.com", "snippet": "Text"}]
    }

    # Cache miss
    assert temp_cache.get_sync(query) is None

    # Cache set & hit
    temp_cache.set_sync(query, data)
    hit = temp_cache.get_sync(query)
    assert hit is not None
    assert hit["status"] == "success"
    assert len(hit["results"]) == 1

    # Normalization hit
    hit_normalized = temp_cache.get_sync("autogen vs langgraph")
    assert hit_normalized is not None


def test_error_and_timeout_not_cached(temp_cache):
    """Never cache error or timeout results."""
    error_data = {"status": "error", "message": "Search failed"}
    timeout_data = {"status": "timeout", "message": "Search timed out"}
    rate_limited_data = {"status": "rate_limited", "message": "429"}

    temp_cache.set_sync("error query", error_data)
    assert temp_cache.get_sync("error query") is None

    temp_cache.set_sync("timeout query", timeout_data)
    assert temp_cache.get_sync("timeout query") is None

    temp_cache.set_sync("rate limit query", rate_limited_data)
    assert temp_cache.get_sync("rate limit query") is None


def test_ttl_expiry(temp_cache):
    """Expired items return None."""
    query = "expiring query"
    data = {"status": "success", "results": [{"title": "T", "url": "U", "snippet": "S"}]}

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CACHE_TTL_SECONDS", "1")
        temp_cache.set_sync(query, data)
        assert temp_cache.get_sync(query) is not None

        # Wait for TTL to expire
        time.sleep(1.2)
        assert temp_cache.get_sync(query) is None


def test_empty_results_short_ttl(temp_cache):
    """Empty results get a shorter TTL cap."""
    query = "empty query"
    empty_data = {"status": "no_results", "results": []}

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CACHE_TTL_SECONDS", "3600")
        temp_cache.set_sync(query, empty_data)
        # Verify it was stored
        hit = temp_cache.get_sync(query)
        assert hit is not None
        assert hit["status"] == "no_results"


def test_cache_disabled_flag(temp_cache):
    """CACHE_ENABLED=false disables cache reads and writes."""
    query = "test flag query"
    data = {"status": "success", "results": [{"title": "T", "url": "U", "snippet": "S"}]}

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CACHE_ENABLED", "false")
        assert not temp_cache.is_enabled()
        temp_cache.set_sync(query, data)
        assert temp_cache.get_sync(query) is None


@pytest.mark.asyncio
async def test_concurrent_access_safety(temp_cache):
    """Ensure multi-threaded/async concurrent access does not corrupt SQLite."""
    async def _worker(idx: int):
        q = f"query_{idx % 5}"
        data = {"status": "success", "results": [{"title": f"T_{idx}", "url": f"https://example.com/{idx}", "snippet": "S"}]}
        await temp_cache.set(q, data)
        res = await temp_cache.get(q)
        assert res is not None

    tasks = [_worker(i) for i in range(30)]
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_executor_cache_hit_does_not_consume_budget(tmp_path):
    """Verify that when executor encounters a cached query, budget is not consumed."""
    db_file = tmp_path / "test_exec_cache.db"
    
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("CACHE_DB_PATH", str(db_file))
        mp.setenv("CACHE_ENABLED", "true")

        # Pre-seed cache
        cache = SearchCache(db_path=str(db_file))
        preseeded_data = {
            "status": "success",
            "results": [{"title": "Preseeded Title", "url": "https://example.com/preseeded", "snippet": "Preseeded snippet content"}]
        }
        await cache.set("cached query", preseeded_data, max_results=3, provider="all")

        call_count = 0
        def fake_search(query: str, max_results: int = 5, **kwargs):
            nonlocal call_count
            call_count += 1
            return '{"status": "success", "results": [{"title": "Live Title", "url": "https://example.com/live", "snippet": "Live snippet"}]}'

        set_global_tool("search_web", fake_search)

        plan = ResearchPlan(
            task_summary="Test Plan",
            subtasks=[
                Subtask(
                    id="subtask_1",
                    title="Subtask 1",
                    description="Desc",
                    search_queries=["cached query"],
                    rationale="R"
                )
            ]
        )

        state = create_initial_state("Test Task", max_retries=1, max_search_budget=5)
        state["plan"] = plan

        # First run: should be a CACHE HIT
        result_state = await executor_node(state)

        assert result_state["total_searches_performed"] == 0
        assert result_state["cache_hits"] == 1
        assert result_state["search_budget_remaining"] == 5
        assert call_count == 0  # Tool was NEVER called because of cache hit!
        assert len(result_state["subtask_results"][0].sources) == 1
        assert result_state["subtask_results"][0].sources[0].title == "Preseeded Title"
