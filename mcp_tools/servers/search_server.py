"""MCP Server for Web Search.

Exposes Model Context Protocol (MCP) tools for real-time web search
and query rephrasing with multi-tier provider support (Tavily, Brave, DDGS,
and intelligent contextual fallback).
"""

import os
import json
import re
import time
import inspect
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional
from mcp.server.mcpserver import MCPServer

try:
    from core.critic_checks import extract_task_keywords
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from core.critic_checks import extract_task_keywords

# Initialize MCP Server instance
search_mcp_server = MCPServer("search_server")


def _get_default_timeout() -> float:
    try:
        return float(os.environ.get("MCP_TOOL_TIMEOUT_SEC", "15.0"))
    except ValueError:
        return 15.0


def _is_rate_limit_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "ratelimit" in name or "429" in msg or "202" in msg or "rate limit" in msg or "too many requests" in msg:
        return True
    if hasattr(exc, "code") and getattr(exc, "code") in (429, 202):
        return True
    return False


def _is_timeout_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "timeout" in name or "timed out" in msg:
        return True
    return False


def filter_relevant_search_results(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Filter out obviously off-topic, spam, emulator, or non-English results."""
    if not results:
        return []

    # Extract substantive keywords from query using critic_checks logic
    keywords = extract_task_keywords(query)

    # Check if query is primarily English/Latin script
    query_has_cjk_or_cyrillic = bool(re.search(r"[\u4e00-\u9fff\u0400-\u04ff\u0600-\u06ff]", query))

    # Common off-topic spam terms (games/emulators/cheats/roms) to filter when query is not gaming-related
    query_lower = query.lower()
    gaming_intent = any(w in query_lower for w in ("game", "gaming", "rom", "emulator", "cheat", "playstation", "nintendo", "xbox"))
    spam_patterns = [r"\b(?:roms?|emulat(?:or|ion)|apkpure|game\s*download|free\s*roms)\b"]

    filtered = []
    for r in results:
        title = r.get("title", "")
        url = r.get("url", "")
        snippet = r.get("snippet", "")
        text = f"{title} {snippet}".lower()
        url_lower = url.lower()

        # 1. Non-English prefilter: if query is English but snippet/title has significant foreign script
        if not query_has_cjk_or_cyrillic:
            foreign_chars = len(re.findall(r"[\u4e00-\u9fff\u0400-\u04ff\u0600-\u06ff]", text))
            if foreign_chars > 8:
                continue

        # 2. Obvious spam/emulator filter if query has no gaming intent
        if not gaming_intent:
            if any(re.search(p, text) for p in spam_patterns) or any(re.search(p, url_lower) for p in spam_patterns):
                continue

        # 3. Keyword overlap requirement
        if keywords:
            matched_kw = [kw for kw in keywords if kw in text]
            if not matched_kw:
                continue
            if len(keywords) >= 3 and len(matched_kw) < 2 and not any(kw in title.lower() for kw in keywords):
                continue

        filtered.append(r)

    return filtered


def _search_tavily(query: str, max_results: int, api_key: str, timeout: float = 10.0) -> List[Dict[str, Any]]:
    """Execute search using Tavily API with timeout."""
    url = "https://api.tavily.com/search"
    payload = json.dumps({
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "ResearchAgentCrew/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        results = []
        for r in data.get("results", []):
            results.append({
                "title": r.get("title", "Untitled"),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")
            })
        return results


def _search_brave(query: str, max_results: int, api_key: str, timeout: float = 10.0) -> List[Dict[str, Any]]:
    """Execute search using Brave Search API with timeout."""
    params = urllib.parse.urlencode({"q": query, "count": max_results})
    url = f"https://api.search.brave.com/res/v1/web/search?{params}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
            "User-Agent": "ResearchAgentCrew/1.0"
        }
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        results = []
        for r in data.get("web", {}).get("results", []):
            results.append({
                "title": r.get("title", "Untitled"),
                "url": r.get("url", ""),
                "snippet": r.get("description", "")
            })
        return results


def _search_ddg(query: str, max_results: int, timeout: float = 6.0) -> List[Dict[str, Any]]:
    """Execute search using DuckDuckGo with backend restriction and timeout protection."""
    import warnings
    results = []
    start_t = time.perf_counter()
    backends = os.environ.get("DDGS_BACKENDS", "duckduckgo,wikipedia").strip()
    effective_timeout = min(timeout, float(os.environ.get("DDGS_TIMEOUT_SEC", "6.0")))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ddgs_cls = None
        try:
            from ddgs import DDGS
            ddgs_cls = DDGS
        except Exception:
            try:
                from duckduckgo_search import DDGS
                ddgs_cls = DDGS
            except Exception:
                pass

        if ddgs_cls:
            ddgs_inst = ddgs_cls(timeout=int(max(1, effective_timeout)))
            kwargs: Dict[str, Any] = {"max_results": max_results}
            try:
                sig = inspect.signature(ddgs_inst.text)
                if "backend" in sig.parameters:
                    kwargs["backend"] = backends
                elif hasattr(ddgs_inst, "_search_sync"):
                    kwargs["backend"] = backends
            except Exception:
                kwargs["backend"] = backends

            raw_gen = ddgs_inst.text(query, **kwargs)
            for item in raw_gen:
                results.append(item)
                if (time.perf_counter() - start_t) >= effective_timeout:
                    break

    formatted = []
    for r in results:
        title = r.get("title", "Untitled")
        url = r.get("href", r.get("link", ""))
        snippet = r.get("body", r.get("snippet", ""))
        if title and url:
            formatted.append({
                "title": title,
                "url": url,
                "snippet": snippet
            })
    return formatted


@search_mcp_server.tool()
def search_web(query: str, max_results: int = 5, timeout_sec: Optional[float] = None) -> str:
    """Search the web for up-to-date information, technical articles, and documentation.

    Args:
        query: Search query string.
        max_results: Maximum number of search results to return (default 5).
        timeout_sec: Timeout in seconds (defaults to MCP_TOOL_TIMEOUT_SEC).

    Returns:
        JSON string containing list of results with title, url, and snippet.
    """
    timeout = timeout_sec if timeout_sec is not None else _get_default_timeout()
    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
    brave_key = os.environ.get("BRAVE_API_KEY", "").strip()

    raw_results: List[Dict[str, Any]] = []
    engine_used = "none"
    last_error: Optional[str] = None
    last_status: Optional[str] = None
    provider_called_and_succeeded = False

    if tavily_key:
        try:
            raw_results = _search_tavily(query, max_results, tavily_key, timeout=timeout)
            engine_used = "tavily"
            provider_called_and_succeeded = True
        except Exception as exc:
            if _is_rate_limit_error(exc):
                last_status = "rate_limited"
            elif _is_timeout_error(exc):
                last_status = "timeout"
            else:
                last_status = "error"
            last_error = f"Tavily error: {str(exc)}"

    if not provider_called_and_succeeded and brave_key:
        try:
            raw_results = _search_brave(query, max_results, brave_key, timeout=timeout)
            engine_used = "brave"
            provider_called_and_succeeded = True
        except Exception as exc:
            if _is_rate_limit_error(exc):
                last_status = "rate_limited"
            elif _is_timeout_error(exc):
                last_status = "timeout"
            else:
                last_status = "error"
            last_error = f"Brave error: {str(exc)}"

    if not provider_called_and_succeeded:
        try:
            raw_results = _search_ddg(query, max_results, timeout=timeout)
            engine_used = "duckduckgo"
            provider_called_and_succeeded = True
        except Exception as exc:
            if _is_rate_limit_error(exc):
                last_status = "rate_limited"
            elif _is_timeout_error(exc):
                last_status = "timeout"
            else:
                last_status = "error"
            last_error = f"DDG error: {str(exc)}"

    # Apply relevance prefilter
    if raw_results:
        raw_results = filter_relevant_search_results(raw_results, query)

    # If results found, return success
    if raw_results:
        return json.dumps({
            "status": "success",
            "query": query,
            "engine": engine_used,
            "count": len(raw_results),
            "results": raw_results
        }, indent=2)

    # If rate-limited, return rate_limited status
    if last_status == "rate_limited":
        return json.dumps({
            "status": "rate_limited",
            "query": query,
            "engine": engine_used,
            "message": last_error or "Search rate limit exceeded (429/202)",
            "results": []
        })

    # If timed out, return timeout status
    if last_status == "timeout" or (last_error and ("timed out" in last_error.lower() or "timeout" in last_error.lower())):
        return json.dumps({
            "status": "timeout",
            "query": query,
            "engine": engine_used,
            "message": last_error or "Search request timed out",
            "results": []
        })

    # If error occurred
    if last_status == "error":
        return json.dumps({
            "status": "error",
            "query": query,
            "engine": engine_used,
            "message": last_error or "Search failed with an error",
            "results": []
        })

    # Zero results found cleanly
    return json.dumps({
        "status": "no_results",
        "query": query,
        "engine": engine_used,
        "message": f"No web search results found for query: '{query}'. Query rephrasing recommended.",
        "results": []
    })


@search_mcp_server.tool()
def rephrase_search_query(query: str, reason: str = "") -> str:
    """Rephrase a failed or zero-result search query into broader, keyword-focused terms.

    Args:
        query: The original search query that yielded poor or no results.
        reason: Why the query failed (e.g., 'too specific', 'zero results').

    Returns:
        JSON string with reformulated search queries.
    """
    cleaned = re.sub(r"[^\w\s-]", " ", query)
    tokens = [w for w in cleaned.split() if len(w) > 2]
    stop_words = {"what", "when", "where", "which", "who", "whom", "this", "that", "these", "those", "from", "with", "about"}
    keywords = [w for w in tokens if w.lower() not in stop_words]

    rephrased_queries = [
        " ".join(keywords[:5]),
        f"{' '.join(keywords[:4])} overview documentation",
        f"{' '.join(keywords[:3])} comparison analysis"
    ]

    return json.dumps({
        "original_query": query,
        "reason": reason,
        "suggested_queries": [q for q in rephrased_queries if q.strip()]
    }, indent=2)


if __name__ == "__main__":
    search_mcp_server.run("stdio")
