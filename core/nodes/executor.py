"""Executor Agent Node for Research Agent Crew.

Executes subtasks concurrently by invoking Model Context Protocol (MCP) search tools,
enforcing pre-allocated split budget shares across subtasks without race conditions,
bounding concurrency with a semaphore, running queries concurrently, and handling
zero-result query rephrasing within subtask budget limits.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from core.state import AgentState, SubtaskResult, SourceItem, TraceEvent, Subtask
from core.tracer import TraceLogger
from mcp_tools.client import MCPClientManager
from core.timing import timed_node, record_tool_call_timing, log_timing_event
from core.cache import get_search_cache


class SubtaskExecutionOutput(BaseModel):
    """Execution output container for a single subtask."""
    subtask_id: str
    title: str
    findings: str
    sources: List[SourceItem] = []
    tools_invoked: List[str] = []
    evidence_gap: bool = False
    searches_spent: int = 0
    cache_hits: int = 0
    trace_events: List[TraceEvent] = []


async def _execute_single_subtask(
    subtask: Subtask,
    subtask_budget: int,
    mcp_client: MCPClientManager,
    iteration: int,
    max_budget: int,
    total_searches_snapshot: int,
    timeout_sec: float,
    state_timings: Optional[Dict[str, Any]]
) -> SubtaskExecutionOutput:
    """Execute a single subtask within its dedicated budget share."""
    subtask_trace_events: List[TraceEvent] = []
    collected_sources: List[SourceItem] = []
    tools_invoked: List[str] = []
    findings_snippets: List[str] = []
    searches_spent = 0

    try:
        # Check if subtask has zero budget allocated
        if subtask_budget <= 0:
            budget_event = TraceLogger.log_budget_alert(
                remaining=max(0, max_budget - total_searches_snapshot),
                limit=max_budget,
                action_taken=f"Search allocation exhausted for subtask '{subtask.id}'. Halting queries for this subtask.",
                iteration=iteration
            )
            subtask_trace_events.append(budget_event)
            return SubtaskExecutionOutput(
                subtask_id=subtask.id,
                title=subtask.title,
                findings=f"No evidence retrieved for this subtask: {subtask.title}",
                sources=[],
                tools_invoked=[],
                evidence_gap=True,
                searches_spent=0,
                trace_events=subtask_trace_events
            )

        queries_to_run = subtask.search_queries[:2] if subtask.search_queries else [subtask.title]
        cache = get_search_cache()
        cache_hits_count = 0
        empty_queries: List[str] = []

        # 1. Parallel cache check for initial queries
        cache_checks = await asyncio.gather(*[
            cache.get(q, max_results=3, provider="all") for q in queries_to_run
        ], return_exceptions=False)

        queries_to_search: List[str] = []
        for query, cached_val in zip(queries_to_run, cache_checks):
            if cached_val is not None:
                # Cache hit: does NOT consume search budget
                cache_hits_count += 1
                duration_ms = 0.5
                record_tool_call_timing(
                    state_timings,
                    "search_web",
                    duration_ms,
                    {"query": query, "status": "cached", "cached": True, "subtask_id": subtask.id}
                )
                log_timing_event("tool", "search_web", duration_ms, {"cached": True, "query": query})

                results_list = cached_val.get("results", []) if isinstance(cached_val, dict) else []
                is_empty = cached_val.get("status") == "no_results" or len(results_list) == 0

                subtask_trace_events.append(
                    TraceLogger.log_tool_result(
                        tool_name="search_web",
                        status="cached",
                        duration_ms=duration_ms,
                        result_summary=f"[CACHED] Returned {len(results_list)} results",
                        subtask_id=subtask.id,
                        raw_result=cached_val,
                        iteration=iteration
                    )
                )

                if is_empty:
                    empty_queries.append(query)
                else:
                    for r in results_list:
                        title = (r.get("title") or "").strip()
                        url = (r.get("url") or "").strip()
                        if not title or not url:
                            continue
                        snippet = r.get("snippet", "")
                        collected_sources.append(
                            SourceItem(
                                title=title,
                                url=url,
                                snippet=snippet,
                                relevance=f"Evidence for '{query}'"
                            )
                        )
                        findings_snippets.append(f"{title}: {snippet}")
            else:
                queries_to_search.append(query)

        # 2. Limit live searches to subtask's remaining budget share
        allowed_searches = max(0, subtask_budget - searches_spent)
        queries_to_execute = queries_to_search[:allowed_searches]

        if len(queries_to_search) > allowed_searches:
            budget_event = TraceLogger.log_budget_alert(
                remaining=allowed_searches,
                limit=max_budget,
                action_taken=f"Search allocation exhausted for subtask '{subtask.id}'. Halting queries for this subtask.",
                iteration=iteration
            )
            subtask_trace_events.append(budget_event)

        for query in queries_to_execute:
            subtask_trace_events.append(
                TraceLogger.log_tool_call(
                    tool_name="search_web",
                    tool_input={"query": query, "max_results": 3},
                    subtask_id=subtask.id,
                    iteration=iteration
                )
            )
            tools_invoked.append("search_web")

        searches_spent += len(queries_to_execute)

        # 3. Concurrently execute initial queries for this subtask
        async def _run_search(q: str):
            res = await mcp_client.call_tool(
                "search_web",
                arguments={"query": q, "max_results": 3},
                timeout_sec=timeout_sec
            )
            return q, res

        search_tasks = [_run_search(q) for q in queries_to_execute]
        search_outcomes = await asyncio.gather(*search_tasks, return_exceptions=False)

        # 4. Process search results & store into cache
        for query, call_res in search_outcomes:
            status = call_res.get("status", "error")
            duration_ms = call_res.get("duration_ms", 0.0)
            output = call_res.get("output", {})

            record_tool_call_timing(
                state_timings,
                "search_web",
                duration_ms,
                {"query": query, "status": status, "subtask_id": subtask.id}
            )

            if status == "timeout":
                subtask_trace_events.append(
                    TraceLogger.log_tool_result(
                        tool_name="search_web",
                        status="timeout",
                        duration_ms=duration_ms,
                        result_summary=f"Search timed out after {timeout_sec}s for query '{query}'",
                        subtask_id=subtask.id,
                        raw_result=output,
                        iteration=iteration
                    )
                )
                continue
            elif status == "rate_limited":
                subtask_trace_events.append(
                    TraceLogger.log_tool_result(
                        tool_name="search_web",
                        status="rate_limited",
                        duration_ms=duration_ms,
                        result_summary=f"Search rate limited (429/202) for query '{query}'",
                        subtask_id=subtask.id,
                        raw_result=output,
                        iteration=iteration
                    )
                )
                continue
            elif status == "error":
                subtask_trace_events.append(
                    TraceLogger.log_tool_result(
                        tool_name="search_web",
                        status="error",
                        duration_ms=duration_ms,
                        result_summary=f"Search failed for query '{query}'",
                        subtask_id=subtask.id,
                        raw_result=output,
                        iteration=iteration
                    )
                )
                continue

            # Cache successful or clean zero-results outcomes
            if isinstance(output, dict) and output.get("status") in ("success", "no_results"):
                await cache.set(query, output, max_results=3, provider="all")

            is_empty = False
            results_list = []
            if isinstance(output, dict):
                results_list = output.get("results", [])
                if output.get("status") == "no_results" or len(results_list) == 0:
                    is_empty = True

            summary_str = f"Returned {len(results_list)} results"
            subtask_trace_events.append(
                TraceLogger.log_tool_result(
                    tool_name="search_web",
                    status=call_res.get("status", "unknown"),
                    duration_ms=duration_ms,
                    result_summary=summary_str,
                    subtask_id=subtask.id,
                    raw_result=output if isinstance(output, dict) else str(output)[:300],
                    iteration=iteration
                )
            )

            if is_empty:
                empty_queries.append(query)
            elif isinstance(output, dict):
                for r in results_list:
                    title = (r.get("title") or "").strip()
                    url = (r.get("url") or "").strip()
                    if not title or not url:
                        continue
                    snippet = r.get("snippet", "")
                    collected_sources.append(
                        SourceItem(
                            title=title,
                            url=url,
                            snippet=snippet,
                            relevance=f"Evidence for '{query}'"
                        )
                    )
                    findings_snippets.append(f"{title}: {snippet}")

        # 5. Handle rephrase on zero results within remaining subtask budget share
        remaining_share = subtask_budget - searches_spent
        for empty_q in empty_queries:
            if remaining_share <= 0:
                break

            subtask_trace_events.append(
                TraceLogger.log_tool_call(
                    tool_name="rephrase_search_query",
                    tool_input={"query": empty_q, "reason": "zero_results"},
                    subtask_id=subtask.id,
                    iteration=iteration
                )
            )
            rephrase_res = await mcp_client.call_tool(
                "rephrase_search_query",
                arguments={"query": empty_q, "reason": "zero_results"},
                timeout_sec=timeout_sec
            )
            rephrase_data = rephrase_res.get("output", {})
            suggested = rephrase_data.get("suggested_queries", []) if isinstance(rephrase_data, dict) else []

            if suggested:
                fallback_query = suggested[0]
                tools_invoked.append("rephrase_search_query")

                # Check cache for fallback query
                cached_fb = await cache.get(fallback_query, max_results=3, provider="all")
                if cached_fb is not None:
                    cache_hits_count += 1
                    duration_ms = 0.5
                    record_tool_call_timing(
                        state_timings,
                        "search_web",
                        duration_ms,
                        {"query": fallback_query, "status": "cached", "cached": True, "subtask_id": subtask.id}
                    )
                    log_timing_event("tool", "search_web", duration_ms, {"cached": True, "query": fallback_query})
                    fb_results = cached_fb.get("results", []) if isinstance(cached_fb, dict) else []
                    subtask_trace_events.append(
                        TraceLogger.log_tool_result(
                            tool_name="search_web",
                            status="cached",
                            duration_ms=duration_ms,
                            result_summary=f"[CACHED] Returned {len(fb_results)} results for rephrased query",
                            subtask_id=subtask.id,
                            raw_result=cached_fb,
                            iteration=iteration
                        )
                    )
                    for r in fb_results:
                        title = (r.get("title") or "").strip()
                        url = (r.get("url") or "").strip()
                        if title and url:
                            snippet = r.get("snippet", "")
                            collected_sources.append(
                                SourceItem(
                                    title=title,
                                    url=url,
                                    snippet=snippet,
                                    relevance=f"Evidence for '{fallback_query}'"
                                )
                            )
                            findings_snippets.append(f"{title}: {snippet}")
                elif remaining_share > 0:
                    searches_spent += 1
                    remaining_share -= 1
                    tools_invoked.append("search_web")

                    subtask_trace_events.append(
                        TraceLogger.log_tool_call(
                            tool_name="search_web",
                            tool_input={"query": fallback_query, "max_results": 3},
                            subtask_id=subtask.id,
                            iteration=iteration
                        )
                    )

                    call_res = await mcp_client.call_tool(
                        "search_web",
                        arguments={"query": fallback_query, "max_results": 3},
                        timeout_sec=timeout_sec
                    )
                    duration_ms = call_res.get("duration_ms", 0.0)
                    output = call_res.get("output", {})
                    record_tool_call_timing(
                        state_timings,
                        "search_web",
                        duration_ms,
                        {"query": fallback_query, "status": call_res.get("status", "unknown"), "subtask_id": subtask.id}
                    )

                    results_list = output.get("results", []) if isinstance(output, dict) else []
                    subtask_trace_events.append(
                        TraceLogger.log_tool_result(
                            tool_name="search_web",
                            status=call_res.get("status", "unknown"),
                            duration_ms=duration_ms,
                            result_summary=f"Returned {len(results_list)} results",
                            subtask_id=subtask.id,
                            raw_result=output if isinstance(output, dict) else str(output)[:300],
                            iteration=iteration
                        )
                    )

                    if isinstance(output, dict) and output.get("status") in ("success", "no_results"):
                        await cache.set(fallback_query, output, max_results=3, provider="all")

                    if isinstance(output, dict):
                        for r in results_list:
                            title = (r.get("title") or "").strip()
                            url = (r.get("url") or "").strip()
                            if not title or not url:
                                continue
                            snippet = r.get("snippet", "")
                            collected_sources.append(
                                SourceItem(
                                    title=title,
                                    url=url,
                                    snippet=snippet,
                                    relevance=f"Evidence for '{fallback_query}'"
                                )
                            )
                            findings_snippets.append(f"{title}: {snippet}")

        # 6. Synthesize subtask findings
        if findings_snippets:
            synthesized_text = " ".join(findings_snippets[:4])
            evidence_gap = False
        else:
            synthesized_text = f"No evidence retrieved for this subtask: {subtask.title}"
            evidence_gap = True

        return SubtaskExecutionOutput(
            subtask_id=subtask.id,
            title=subtask.title,
            findings=synthesized_text,
            sources=collected_sources,
            tools_invoked=tools_invoked,
            evidence_gap=evidence_gap,
            searches_spent=searches_spent,
            cache_hits=cache_hits_count,
            trace_events=subtask_trace_events
        )

    except Exception as exc:
        subtask_trace_events.append(
            TraceLogger.log_tool_result(
                tool_name="search_web",
                status="error",
                duration_ms=0.0,
                result_summary=f"Subtask execution encountered exception: {str(exc)}",
                subtask_id=subtask.id,
                raw_result={"error": str(exc)},
                iteration=iteration
            )
        )
        return SubtaskExecutionOutput(
            subtask_id=subtask.id,
            title=subtask.title,
            findings=f"Error executing subtask '{subtask.title}': {str(exc)}",
            sources=[],
            tools_invoked=tools_invoked or ["search_web"],
            evidence_gap=True,
            searches_spent=searches_spent,
            cache_hits=cache_hits_count,
            trace_events=subtask_trace_events
        )


@timed_node("executor")
async def executor_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node: Executor Agent.

    Executes pending subtasks concurrently, bounded by EXECUTOR_CONCURRENCY semaphore,
    allocates isolated budget shares per subtask to avoid shared counter races, and
    aggregates results and budget counts cleanly once.
    """
    if state.get("human_escalation_needed"):
        return {"status": "escalated"}

    plan = state.get("plan")
    if not plan or not plan.subtasks:
        return {"status": "synthesizing"}

    mcp_client = MCPClientManager()
    iteration = state.get("retry_count", 0)
    max_budget = state.get("max_search_budget", 10)
    total_searches = state.get("total_searches_performed", 0)
    trace_events = list(state.get("trace_events", []))
    subtask_results = list(state.get("subtask_results", []))
    state_timings = state.get("timings")

    try:
        timeout_sec = float(os.environ.get("MCP_TOOL_TIMEOUT_SEC", "15.0"))
    except ValueError:
        timeout_sec = 15.0

    try:
        concurrency_limit = int(os.environ.get("EXECUTOR_CONCURRENCY", "3"))
    except ValueError:
        concurrency_limit = 3
    if concurrency_limit < 1:
        concurrency_limit = 1

    # Determine available budget for this iteration
    global_searches_left = max(0, max_budget - total_searches)

    if iteration == 0:
        available_budget = min(state.get("initial_budget_remaining", max_budget), global_searches_left)
    else:
        # Revision pass: draw strictly from revision_reserve_remaining
        reserve = state.get("revision_reserve_remaining", 0)
        rounds_remaining = max(1, (state.get("max_retries", 1) - iteration) + 1)
        round_share = reserve if rounds_remaining <= 1 else (reserve // rounds_remaining)
        available_budget = min(reserve, max(1, round_share) if reserve > 0 else 0)
        available_budget = min(available_budget, global_searches_left)

    available_budget = max(0, available_budget)

    # Identify subtasks that have not yet been executed
    pending_subtasks = [s for s in plan.subtasks if not any(r.subtask_id == s.id for r in subtask_results)]

    if not pending_subtasks:
        return {
            "subtask_results": subtask_results,
            "status": "synthesizing"
        }

    # Divide available iteration budget evenly across pending subtasks
    num_pending = len(pending_subtasks)
    base_share = available_budget // num_pending
    remainder = available_budget % num_pending

    subtask_budgets = [
        base_share + (1 if idx < remainder else 0)
        for idx in range(num_pending)
    ]

    semaphore = asyncio.Semaphore(concurrency_limit)

    async def _bounded_subtask_exec(subtask: Subtask, share: int):
        async with semaphore:
            return await _execute_single_subtask(
                subtask=subtask,
                subtask_budget=share,
                mcp_client=mcp_client,
                iteration=iteration,
                max_budget=max_budget,
                total_searches_snapshot=total_searches,
                timeout_sec=timeout_sec,
                state_timings=state_timings
            )

    tasks = [
        _bounded_subtask_exec(subtask, share)
        for subtask, share in zip(pending_subtasks, subtask_budgets)
    ]

    subtask_outputs: List[SubtaskExecutionOutput] = await asyncio.gather(*tasks)

    # Aggregate outputs, spent budget, and cache hits once
    iteration_searches_used = 0
    iteration_cache_hits = 0
    for out in subtask_outputs:
        iteration_searches_used += out.searches_spent
        iteration_cache_hits += out.cache_hits
        trace_events.extend(out.trace_events)
        subtask_results.append(
            SubtaskResult(
                subtask_id=out.subtask_id,
                title=out.title,
                findings=out.findings,
                sources=out.sources,
                tools_invoked=out.tools_invoked,
                evidence_gap=out.evidence_gap
            )
        )

    total_searches += iteration_searches_used
    total_cache_hits = state.get("cache_hits", 0) + iteration_cache_hits

    # Compute updated budget trackers
    if iteration == 0:
        initial_remaining = max(0, state.get("initial_budget_remaining", max_budget) - iteration_searches_used)
        reserve_remaining = state.get("revision_reserve_remaining", 0)
    else:
        initial_remaining = 0
        reserve_remaining = max(0, state.get("revision_reserve_remaining", 0) - iteration_searches_used)

    return {
        "subtask_results": subtask_results,
        "search_budget_remaining": max(0, max_budget - total_searches),
        "initial_budget_remaining": initial_remaining,
        "revision_reserve_remaining": reserve_remaining,
        "iteration_budget_remaining": max(0, available_budget - iteration_searches_used),
        "total_searches_performed": total_searches,
        "cache_hits": total_cache_hits,
        "trace_events": trace_events,
        "status": "synthesizing"
    }
