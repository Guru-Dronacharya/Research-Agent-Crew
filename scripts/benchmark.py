"""Benchmark script for Research Agent Crew.

Runs a fixed research task through the real LangGraph multi-agent workflow
and outputs a detailed timing breakdown across nodes and tool executions.
"""

import sys
import os
import time
import argparse
import asyncio
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment if present
env_file = PROJECT_ROOT / ".env"
if env_file.is_file():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v

from core.graph import build_research_graph, create_initial_state
from core.llm import set_global_llm
from tests.stubs import StubLLM

DEFAULT_TASK = "Compare LangGraph vs AutoGen vs CrewAI for production agent architectures"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Research Agent Crew execution.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Task to benchmark")
    parser.add_argument("--budget", type=int, default=6, help="Max search budget")
    parser.add_argument("--retries", type=int, default=1, help="Max critic retries")
    parser.add_argument("--no-cache", action="store_true", help="Disable search cache")
    parser.add_argument("--mock-llm", action="store_true", help="Use StubLLM for deterministic offline benchmark")
    parser.add_argument("--output", type=str, default=None, help="Save output to file")
    return parser.parse_args()


async def run_benchmark(args):
    if args.no_cache:
        os.environ["CACHE_ENABLED"] = "false"

    has_live_llm = bool(
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    ) and not args.mock_llm

    if not has_live_llm or args.mock_llm:
        print("[INFO] Using deterministic StubLLM for benchmark execution.")
        set_global_llm(StubLLM())

    graph = build_research_graph()
    initial_state = create_initial_state(
        task=args.task,
        max_retries=args.retries,
        max_search_budget=args.budget
    )

    print("=" * 70)
    print("RESEARCH AGENT CREW BENCHMARK")
    print("=" * 70)
    print(f"Task: {args.task}")
    print(f"Max Search Budget: {args.budget}")
    print(f"Max Critic Retries: {args.retries}")
    print(f"Cache Enabled: {os.environ.get('CACHE_ENABLED', 'true')}")
    print("=" * 70)
    print("Running workflow...")

    wall_start = time.perf_counter()
    final_state = await graph.ainvoke(initial_state)
    total_wall_sec = time.perf_counter() - wall_start

    timings = final_state.get("timings") or {}
    node_timings = timings.get("nodes", {})
    tool_calls = timings.get("tool_calls", [])
    total_searches = final_state.get("total_searches_performed", 0)
    status = final_state.get("status", "unknown")
    retries_done = final_state.get("retry_count", 0)

    # Compute tool call stats
    search_calls = [t for t in tool_calls if t.get("tool") == "search_web"]
    total_search_time_ms = sum(t.get("duration_ms", 0.0) for t in search_calls)
    avg_search_time_ms = (total_search_time_ms / len(search_calls)) if search_calls else 0.0
    cached_searches = [t for t in search_calls if t.get("cached", False)]

    lines = []
    lines.append("=" * 70)
    lines.append("TIMING RESULTS SUMMARY")
    lines.append("=" * 70)
    lines.append(f"{'Metric':<35} | {'Value':<30}")
    lines.append("-" * 70)
    lines.append(f"{'Workflow Status':<35} | {status.upper():<30}")
    lines.append(f"{'Total Wall-Clock Time':<35} | {total_wall_sec:.2f} s ({total_wall_sec * 1000.0:.1f} ms)")
    lines.append(f"{'Reflection Loops Completed':<35} | {retries_done} / {args.retries}")
    lines.append(f"{'Total Searches Performed':<35} | {total_searches} / {args.budget}")
    lines.append(f"{'Search Cache Hits':<35} | {len(cached_searches)} / {len(search_calls)}")
    lines.append(f"{'Total Search Time':<35} | {total_search_time_ms:.1f} ms")
    lines.append(f"{'Average Search Latency':<35} | {avg_search_time_ms:.1f} ms")
    lines.append("-" * 70)
    lines.append("NODE BREAKDOWN")
    lines.append("-" * 70)
    lines.append(f"{'Node Name':<35} | {'Duration (ms)':<15} | {'Share (%)':<10}")
    lines.append("-" * 70)

    total_node_time = sum(node_timings.values()) if node_timings else 1.0
    for node in ["planner", "executor", "synthesizer", "critic", "finalizer"]:
        duration = node_timings.get(node, 0.0)
        pct = (duration / total_node_time * 100.0) if total_node_time > 0 else 0.0
        lines.append(f"{node:<35} | {duration:>10.2f} ms | {pct:>8.1f}%")

    lines.append("-" * 70)
    lines.append(f"{'Sum of Node Times':<35} | {total_node_time:>10.2f} ms | 100.0%")
    lines.append("=" * 70)

    output_str = "\n".join(lines)
    print(output_str)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_str, encoding="utf-8")
        print(f"\n[INFO] Benchmark output written to: {out_path}")

    return final_state


def main():
    args = parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
