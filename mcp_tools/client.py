"""MCP Client Manager for Research Agent Crew.

Provides a unified client interface to discover and invoke Model Context Protocol (MCP)
tools across configured servers (search, filesystem). Includes timeout enforcement via
asyncio.to_thread, latency tracking, and error resilience.
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from mcp_tools.servers.search_server import search_web, rephrase_search_query
from mcp_tools.servers.fs_server import save_report, read_report, list_reports
from core.timing import log_timing_event
from core.ratelimit import get_search_rate_limiter


_GLOBAL_HANDLERS: Dict[str, Callable] = {}


def set_global_tool(name: str, fn: Optional[Callable], description: str = "") -> None:
    """Register or remove a global tool override for testing and extensions."""
    if fn is None:
        _GLOBAL_HANDLERS.pop(name, None)
    else:
        _GLOBAL_HANDLERS[name] = fn


def reset_global_tools() -> None:
    """Clear all registered global tool overrides."""
    _GLOBAL_HANDLERS.clear()


class MCPToolMetadata:
    def __init__(self, name: str, description: str, server: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.server = server
        self.input_schema = input_schema

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "server": self.server,
            "input_schema": self.input_schema
        }


class MCPClientManager:
    """Manages MCP tool discovery and execution with thread offloading and metrics."""

    def __init__(self, config_path: Optional[str] = None):
        base_dir = Path(__file__).resolve().parent
        self.config_path = Path(config_path) if config_path else (base_dir / "mcp_config.json")
        self.config: Dict[str, Any] = self._load_config()
        self._tools: Dict[str, MCPToolMetadata] = {}
        self._custom_handlers: Dict[str, Callable] = {}
        self._register_default_tools()

    def _load_config(self) -> Dict[str, Any]:
        """Load standard MCP configuration if present."""
        if self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"mcpServers": {}}

    def _register_default_tools(self) -> None:
        """Register MCP tools discovered from servers."""
        self._tools = {
            "search_web": MCPToolMetadata(
                name="search_web",
                description="Search the web for up-to-date facts, benchmarks, and citations",
                server="search",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            ),
            "rephrase_search_query": MCPToolMetadata(
                name="rephrase_search_query",
                description="Rephrase a search query when initial queries return zero results",
                server="search",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Original query"},
                        "reason": {"type": "string", "default": ""}
                    },
                    "required": ["query"]
                }
            ),
            "save_report": MCPToolMetadata(
                name="save_report",
                description="Save research report or summary artifact to filesystem",
                server="filesystem",
                input_schema={
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "Report filename"},
                        "content": {"type": "string", "description": "Text content"},
                        "folder": {"type": "string", "default": "reports"}
                    },
                    "required": ["filename", "content"]
                }
            ),
            "read_report": MCPToolMetadata(
                name="read_report",
                description="Read an existing research report or file from filesystem",
                server="filesystem",
                input_schema={
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "Filename to read"},
                        "folder": {"type": "string", "default": "reports"}
                    },
                    "required": ["filename"]
                }
            ),
            "list_reports": MCPToolMetadata(
                name="list_reports",
                description="List all available generated research reports",
                server="filesystem",
                input_schema={
                    "type": "object",
                    "properties": {
                        "folder": {"type": "string", "default": "reports"}
                    }
                }
            )
        }

    def register_tool(self, name: str, fn: Callable, description: str = "") -> None:
        """Register a custom tool handler (useful for testing and extensions)."""
        self._tools[name] = MCPToolMetadata(
            name=name,
            description=description or name,
            server="custom",
            input_schema={}
        )
        self._custom_handlers[name] = fn
        _GLOBAL_HANDLERS[name] = fn

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return list of discovered MCP tools with metadata."""
        return [t.to_dict() for t in self._tools.values()]

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout_sec: Optional[float] = None
    ) -> Dict[str, Any]:
        """Invoke an MCP tool with timeout protection and latency telemetry.

        Uses asyncio.to_thread for synchronous tool functions so the event loop
        is not blocked and asyncio.wait_for can accurately fire.

        Args:
            tool_name: Name of tool to execute.
            arguments: Dictionary of arguments.
            timeout_sec: Maximum execution time in seconds. Defaults to MCP_TOOL_TIMEOUT_SEC or 15.0.

        Returns:
            Dict with keys: 'status', 'tool', 'duration_ms', 'output', 'raw'
        """
        start_time = time.perf_counter()
        effective_timeout = (
            timeout_sec
            if timeout_sec is not None
            else float(os.environ.get("MCP_TOOL_TIMEOUT_SEC", "15.0"))
        )

        if tool_name not in self._tools:
            return {
                "status": "error",
                "tool": tool_name,
                "duration_ms": 0.0,
                "output": f"Tool '{tool_name}' not found in MCP registry.",
                "raw": None
            }

        async def _dispatch():
            # Check custom handlers and global test overrides first
            handler = self._custom_handlers.get(tool_name) or _GLOBAL_HANDLERS.get(tool_name)
            if handler:
                if asyncio.iscoroutinefunction(handler):
                    return await handler(**arguments)
                return await asyncio.to_thread(handler, **arguments)

            # In-process execution dispatched to worker thread to prevent event loop blocking
            if tool_name == "search_web":
                await get_search_rate_limiter().acquire()
                return await asyncio.to_thread(
                    search_web,
                    query=arguments.get("query", ""),
                    max_results=int(arguments.get("max_results", 5)),
                    timeout_sec=effective_timeout
                )
            elif tool_name == "rephrase_search_query":
                return await asyncio.to_thread(
                    rephrase_search_query,
                    query=arguments.get("query", ""),
                    reason=arguments.get("reason", "")
                )
            elif tool_name == "save_report":
                return await asyncio.to_thread(
                    save_report,
                    filename=arguments.get("filename", "report.md"),
                    content=arguments.get("content", ""),
                    folder=arguments.get("folder", "reports")
                )
            elif tool_name == "read_report":
                return await asyncio.to_thread(
                    read_report,
                    filename=arguments.get("filename", ""),
                    folder=arguments.get("folder", "reports")
                )
            elif tool_name == "list_reports":
                return await asyncio.to_thread(
                    list_reports,
                    folder=arguments.get("folder", "reports")
                )
            else:
                raise ValueError(f"Unknown MCP tool: {tool_name}")

        try:
            raw_result = await asyncio.wait_for(_dispatch(), timeout=effective_timeout)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            # Parse JSON result if returned as json string
            parsed = None
            if isinstance(raw_result, str):
                try:
                    parsed = json.loads(raw_result)
                except Exception:
                    parsed = raw_result
            else:
                parsed = raw_result

            # Check if output explicitly signaled timeout or error internally
            status = "success"
            if isinstance(parsed, dict):
                if parsed.get("status") == "timeout":
                    status = "timeout"
                elif parsed.get("status") == "rate_limited":
                    status = "rate_limited"
                elif parsed.get("status") == "error":
                    status = "error"

            log_timing_event("tool", tool_name, duration_ms, {"status": status})
            return {
                "status": status,
                "tool": tool_name,
                "duration_ms": duration_ms,
                "output": parsed,
                "raw": raw_result
            }

        except (asyncio.TimeoutError, TimeoutError):
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            log_timing_event("tool", tool_name, duration_ms, {"status": "timeout"})
            return {
                "status": "timeout",
                "tool": tool_name,
                "duration_ms": duration_ms,
                "output": {
                    "status": "timeout",
                    "message": f"MCP Tool '{tool_name}' timed out after {effective_timeout}s.",
                    "results": []
                },
                "raw": None
            }
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            log_timing_event("tool", tool_name, duration_ms, {"status": "error", "error": str(exc)})
            return {
                "status": "error",
                "tool": tool_name,
                "duration_ms": duration_ms,
                "output": f"MCP execution failed: {str(exc)}",
                "raw": None
            }
