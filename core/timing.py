"""Wall-clock timing instrumentation and telemetry for Research Agent Crew.

Provides context managers, decorators, and state logging to record durations
per node, per LLM call, and per tool call.
"""

import time
import json
import logging
import functools
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager, asynccontextmanager

logger = logging.getLogger("research_agent_crew.timing")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def log_timing_event(
    category: str,
    name: str,
    duration_ms: float,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Emit a JSON line log at INFO level for a timing event."""
    record = {
        "timestamp": now_iso(),
        "category": category,
        "name": name,
        "duration_ms": round(duration_ms, 2)
    }
    if details:
        record["details"] = details
    logger.info(json.dumps(record))
    return record


@asynccontextmanager
async def measure_async(category: str, name: str, details: Optional[Dict[str, Any]] = None):
    """Async context manager to measure wall-clock duration of a block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        log_timing_event(category, name, duration_ms, details)


@contextmanager
def measure_sync(category: str, name: str, details: Optional[Dict[str, Any]] = None):
    """Sync context manager to measure wall-clock duration of a block."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000.0
        log_timing_event(category, name, duration_ms, details)


def timed_node(node_name: str):
    """Decorator for LangGraph nodes to record wall-clock duration in state['timings']."""
    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def wrapper(state: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
            start = time.perf_counter()
            res = await fn(state, *args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000.0

            # Update timings dict
            timings = dict(state.get("timings") or {})
            nodes = dict(timings.get("nodes") or {})
            nodes[node_name] = round(nodes.get(node_name, 0.0) + duration_ms, 2)
            timings["nodes"] = nodes

            # Append to node_events
            node_events = list(timings.get("node_events") or [])
            node_events.append({
                "node": node_name,
                "duration_ms": round(duration_ms, 2),
                "timestamp": now_iso()
            })
            timings["node_events"] = node_events

            log_timing_event("node", node_name, duration_ms)

            if isinstance(res, dict):
                # If res already updated timings, merge node duration
                res_timings = res.get("timings")
                if isinstance(res_timings, dict):
                    res_nodes = dict(res_timings.get("nodes") or {})
                    res_nodes[node_name] = nodes[node_name]
                    res_timings["nodes"] = res_nodes
                    res_timings["node_events"] = node_events
                else:
                    res["timings"] = timings
            return res
        return wrapper
    return decorator


def record_tool_call_timing(
    state_timings: Optional[Dict[str, Any]],
    tool_name: str,
    duration_ms: float,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Record a tool timing event into state timings and emit JSON log."""
    log_timing_event("tool", tool_name, duration_ms, details)
    if state_timings is not None:
        tools = list(state_timings.get("tool_calls") or [])
        tools.append({
            "tool": tool_name,
            "duration_ms": round(duration_ms, 2),
            "timestamp": now_iso(),
            **(details or {})
        })
        state_timings["tool_calls"] = tools


def record_llm_call_timing(
    call_name: str,
    model_name: str,
    duration_ms: float,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Record an LLM timing event and emit JSON log."""
    payload = {"model": model_name, **(details or {})}
    log_timing_event("llm", call_name, duration_ms, payload)
