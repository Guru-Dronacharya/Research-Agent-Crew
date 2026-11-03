"""Unit tests for FastAPI endpoints and async worker progress reporting."""

import pytest
from fastapi.testclient import TestClient
from api.app import app
from mcp_tools.client import set_global_tool, reset_global_tools


@pytest.fixture
def client():
    def fake_search(query: str, max_results: int = 5, **kwargs):
        return '{"status": "success", "results": [{"title": "API Test", "url": "https://example.com/api", "snippet": "Evidence"}]}'
    set_global_tool("search_web", fake_search)
    with TestClient(app) as test_client:
        yield test_client
    reset_global_tools()


def test_health_check(client):
    """Verify health check endpoint returns 200 and mcp tools count."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["mcp_tools_available"] >= 5


def test_sync_research_endpoint(client):
    """Verify /research/run synchronously executes graph and returns cache_hits and status."""
    payload = {
        "task": "FastAPI Testing of Multi-Agent Orchestration",
        "max_retries": 1,
        "max_search_budget": 5
    }
    resp = client.post("/research/run", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("completed", "finished", "escalated")
    assert "searches_performed" in data
    assert "cache_hits" in data
    assert data["report_markdown"] is not None


def test_async_worker_status_endpoint_progress(client):
    """Verify /research/start and /research/{task_id}/status expose progress fields."""
    payload = {
        "task": "Async Progress Test for Web Search Crew",
        "max_retries": 1,
        "max_search_budget": 5
    }
    resp = client.post("/research/start", json=payload)
    assert resp.status_code == 200
    start_data = resp.json()
    task_id = start_data["task_id"]
    assert start_data["status"] == "started"

    # Poll status immediately
    status_resp = client.get(f"/research/{task_id}/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()

    assert status_data["task_id"] == task_id
    assert "status" in status_data
    assert "current_node" in status_data
    assert "elapsed_seconds" in status_data
    assert "searches_performed" in status_data
    assert "cache_hits" in status_data
    assert isinstance(status_data["elapsed_seconds"], (int, float))
    assert isinstance(status_data["searches_performed"], int)
    assert isinstance(status_data["cache_hits"], int)
