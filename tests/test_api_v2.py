"""Unit tests for FastAPI endpoints: Threads, Runs, Resumption, Cancellation,
Export, and API_KEY authentication using hmac.compare_digest.
"""

import pytest
from fastapi.testclient import TestClient
from api.app import app
from core.db.migrator import apply_migrations
from mcp_tools.client import set_global_tool, reset_global_tools


@pytest.fixture(autouse=True)
def setup_api_env(tmp_path, monkeypatch):
    """Set up test DBs and fake tools for API tests."""
    app_db = str(tmp_path / "api_test.db")
    chk_db = str(tmp_path / "api_chk.db")
    monkeypatch.setenv("DATABASE_PATH", app_db)
    monkeypatch.setenv("CHECKPOINT_DB_PATH", chk_db)
    monkeypatch.setenv("API_KEY", "")  # Default no auth required

    import asyncio
    asyncio.run(apply_migrations(app_db))

    def fake_search(query: str, max_results: int = 5, **kwargs):
        return '{"status": "success", "results": [{"title": "API Test", "url": "https://example.com/api", "snippet": "Evidence"}]}'
    set_global_tool("search_web", fake_search)

    yield

    reset_global_tools()


def test_api_key_auth_enforcement_and_health_bypass(monkeypatch):
    """API_KEY must be enforced on protected routes using hmac.compare_digest while /health remains public."""
    monkeypatch.setenv("API_KEY", "super-secret-key-123")

    with TestClient(app) as client:
        # /health must always be accessible without key
        resp_health = client.get("/health")
        assert resp_health.status_code == 200

        # Protected route without key -> 401
        resp_unauth = client.get("/threads")
        assert resp_unauth.status_code == 401

        # Protected route with wrong key -> 401
        resp_bad = client.get("/threads", headers={"X-API-Key": "wrong-key"})
        assert resp_bad.status_code == 401

        # Protected route with valid X-API-Key -> 200
        resp_good = client.get("/threads", headers={"X-API-Key": "super-secret-key-123"})
        assert resp_good.status_code == 200

        # Protected route with valid Bearer token -> 200
        resp_bearer = client.get("/threads", headers={"Authorization": "Bearer super-secret-key-123"})
        assert resp_bearer.status_code == 200


def test_thread_and_run_lifecycle_api_endpoints():
    """Full lifecycle testing via REST API: thread CRUD, run launch, status check, and export."""
    with TestClient(app) as client:
        # 1. Create Thread
        create_resp = client.post("/threads", json={"title": "FastAPI Testing Thread"})
        assert create_resp.status_code == 200
        thread = create_resp.json()
        thread_id = thread["id"]
        assert thread["title"] == "FastAPI Testing Thread"

        # 2. List Threads
        list_resp = client.get("/threads")
        assert list_resp.status_code == 200
        threads = list_resp.json()
        assert any(t["id"] == thread_id for t in threads)

        # 3. Patch Thread (Rename)
        patch_resp = client.patch(f"/threads/{thread_id}", json={"title": "Updated Title"})
        assert patch_resp.status_code == 200
        assert patch_resp.json()["title"] == "Updated Title"

        # 4. Launch Run on Thread
        run_resp = client.post(f"/threads/{thread_id}/runs", json={"task": "Investigate REST APIs"})
        assert run_resp.status_code == 200
        run_data = run_resp.json()
        run_id = run_data["id"]
        assert run_data["status"] == "running"

        # 5. Get Run
        get_run_resp = client.get(f"/runs/{run_id}")
        assert get_run_resp.status_code == 200
        assert get_run_resp.json()["id"] == run_id

        # 6. Cancel Run
        cancel_resp = client.post(f"/runs/{run_id}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancel_requested"

        # 7. Export Thread
        export_resp = client.get(f"/threads/{thread_id}/export")
        assert export_resp.status_code == 200
        export_data = export_resp.json()
        assert export_data["thread"]["id"] == thread_id
        assert len(export_data["runs"]) >= 1

        # 8. Delete Thread
        del_resp = client.delete(f"/threads/{thread_id}")
        assert del_resp.status_code == 200

        # Verify deletion
        get_del_resp = client.get(f"/threads/{thread_id}")
        assert get_del_resp.status_code == 404
