"""Unit and integration tests for Model Context Protocol (MCP) servers and client."""

import pytest
import asyncio
import time
from mcp_tools.client import MCPClientManager
from mcp_tools.servers.fs_server import _safe_path


@pytest.mark.asyncio
async def test_mcp_client_tool_discovery():
    """Verify that MCPClientManager discovers all standard tools."""
    client = MCPClientManager()
    tools = client.list_tools()
    tool_names = [t["name"] for t in tools]
    assert "search_web" in tool_names
    assert "rephrase_search_query" in tool_names
    assert "save_report" in tool_names
    assert "read_report" in tool_names
    assert "list_reports" in tool_names


@pytest.mark.asyncio
async def test_mcp_filesystem_save_and_read():
    """Verify writing and reading reports via Filesystem MCP tools."""
    client = MCPClientManager()
    test_filename = "pytest_mcp_test.md"
    test_content = "# Pytest MCP Header\nVerified persistence."

    # Save
    save_res = await client.call_tool(
        "save_report",
        arguments={"filename": test_filename, "content": test_content}
    )
    assert save_res["status"] == "success"
    assert "bytes_written" in save_res["output"]

    # Read
    read_res = await client.call_tool(
        "read_report",
        arguments={"filename": test_filename}
    )
    assert read_res["status"] == "success"
    assert read_res["output"]["content"] == test_content


def test_mcp_filesystem_directory_traversal_prevention():
    """Verify directory traversal attempts are securely rejected."""
    with pytest.raises(ValueError):
        _safe_path("../../../etc/passwd")


@pytest.mark.asyncio
async def test_mcp_search_rephrase():
    """Verify query rephrasing tool handles zero-result query."""
    client = MCPClientManager()
    res = await client.call_tool(
        "rephrase_search_query",
        arguments={"query": "super_specific_nonexistent_query_xyz123", "reason": "zero_results"}
    )
    assert res["status"] == "success"
    data = res["output"]
    assert "suggested_queries" in data
    assert len(data["suggested_queries"]) > 0


@pytest.mark.asyncio
async def test_mcp_timeout_handling():
    """Verify MCP client handles timeout parameters safely on nonexistent tools."""
    client = MCPClientManager()
    res = await client.call_tool("nonexistent_tool", {})
    assert res["status"] == "error"


@pytest.mark.asyncio
async def test_mcp_tool_timeout_enforcement():
    """Verify that a tool sleeping longer than timeout returns status 'timeout' within ~timeout seconds."""
    client = MCPClientManager()

    def slow_fn(duration: float = 1.0):
        time.sleep(duration)
        return "completed"

    client.register_tool("slow_tool", slow_fn)
    start_t = time.perf_counter()
    res = await client.call_tool("slow_tool", {"duration": 1.2}, timeout_sec=0.25)
    elapsed = time.perf_counter() - start_t

    assert res["status"] == "timeout"
    assert elapsed < 0.9  # Timeout fired promptly well before the 1.2s sleep finished


@pytest.mark.asyncio
async def test_mcp_concurrent_calls_not_blocking():
    """Verify two concurrent call_tool invocations finish in roughly max(t1, t2), not t1 + t2."""
    client = MCPClientManager()

    def slow_worker(sleep_sec: float):
        time.sleep(sleep_sec)
        return {"slept": sleep_sec}

    client.register_tool("slow_worker", slow_worker)
    start_t = time.perf_counter()
    res1, res2 = await asyncio.gather(
        client.call_tool("slow_worker", {"sleep_sec": 0.3}, timeout_sec=2.0),
        client.call_tool("slow_worker", {"sleep_sec": 0.3}, timeout_sec=2.0)
    )
    elapsed = time.perf_counter() - start_t

    assert res1["status"] == "success"
    assert res2["status"] == "success"
    # Threaded concurrent execution takes roughly max(0.3, 0.3) ~0.3s-0.5s, rather than sequential 0.6s+
    assert elapsed < 0.55
