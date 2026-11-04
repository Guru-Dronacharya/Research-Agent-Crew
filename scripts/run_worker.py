"""Worker script for executing research runs out-of-process.

Used for CLI execution, subprocess isolation, and crash resilience tests.
Supports deterministic test doubles (StubLLM, PassingCriticLLM, mock search)
and controlled pause points to allow parent processes to terminate with process.kill().
"""

import sys
import json
import argparse
import asyncio
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe stdout encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.db.migrator import apply_migrations
from core.db.repositories import ThreadRepo
from core.services.runs import start_run, execute_run
from mcp_tools.client import set_global_tool
from tests.stubs import StubLLM
from core.llm import set_global_llm
from core.state import (
    ResearchPlan,
    CriticEvaluation,
    CriticRubricScore,
    SubtaskResult,
    StructuredReport
)


class PassingCriticLLM(StubLLM):
    """Test double LLM whose critic always awards a passing rubric score."""

    async def evaluate_report(
        self,
        task: str,
        plan: ResearchPlan,
        report: str,
        subtask_results: list[SubtaskResult],
        iteration: int = 0,
        deterministic_metrics: dict = None,
        structured_report: StructuredReport = None
    ) -> CriticEvaluation:
        return CriticEvaluation(
            iteration=iteration,
            coverage=CriticRubricScore(
                dimension="Coverage",
                score=5,
                rationale="Comprehensive coverage across all subtasks and dimensions."
            ),
            evidence=CriticRubricScore(
                dimension="Evidence",
                score=5,
                rationale="Findings are supported by verified documentation sources."
            ),
            coherence=CriticRubricScore(
                dimension="Coherence",
                score=5,
                rationale="Logical progression with clear executive summary and structure."
            ),
            total_score=15,
            verdict="PASS",
            overall_feedback="Excellent draft meeting all quality rubrics.",
            next_step_instructions=[]
        )


def setup_mock_search():
    """Register deterministic offline search tool."""
    def mock_search(query: str, max_results: int = 3, **kwargs):
        print(f"  [Worker Mock Search] Query: '{query}'", flush=True)
        return json.dumps({
            "status": "success",
            "results": [
                {
                    "title": f"Doc for {query[:30]}",
                    "url": "https://example.com/doc",
                    "snippet": f"Evidence snippet for {query}"
                }
            ]
        })

    set_global_tool("search_web", mock_search)


async def main():
    parser = argparse.ArgumentParser(description="Research Agent Crew Worker")
    parser.add_argument("--run-id", type=str, help="Existing run ID to execute")
    parser.add_argument("--thread-id", type=str, help="Thread ID (if creating new run)")
    parser.add_argument("--task", type=str, help="Task description (if creating new run)")
    parser.add_argument("--config", type=str, default="{}", help="JSON config for run")
    parser.add_argument(
        "--kill-point",
        type=str,
        choices=["none", "executor"],
        default="none",
        help="Where to pause and signal parent for process.kill()"
    )
    parser.add_argument("--signal-file", type=str, help="File to write when kill point is reached")
    parser.add_argument("--critic-pass", action="store_true", help="Critic test double always returns PASS")
    args = parser.parse_args()

    # Ensure DB schema is applied
    await apply_migrations()

    # Setup LLM test doubles
    if args.critic_pass:
        set_global_llm(PassingCriticLLM())
    else:
        set_global_llm(StubLLM())

    # Setup search test double
    setup_mock_search()

    thread_repo = ThreadRepo()

    if args.run_id:
        target_run_id = args.run_id
    else:
        if not args.thread_id:
            thread = await thread_repo.create_thread(title="Worker Thread")
            thread_id = thread.id
        else:
            thread_id = args.thread_id

        task = args.task or "Analyze agent memory architectures"
        cfg = json.loads(args.config)
        run = await start_run(thread_id=thread_id, task=task, config=cfg)
        target_run_id = run.id
        print(f"RUN_ID:{target_run_id}", flush=True)

    # If pause point requested, hook into graph
    if args.kill_point == "executor":
        import core.graph as graph_module
        orig_synth = graph_module.synthesizer_node

        async def pausing_synthesizer(state):
            print("MID_EXECUTOR_SIGNAL", flush=True)
            if args.signal_file:
                try:
                    with open(args.signal_file, "w", encoding="utf-8") as f:
                        f.write("ready\n")
                except Exception as e:
                    print(f"Error writing signal file: {e}", flush=True)
            # Pause to give parent process time to issue process.kill()
            await asyncio.sleep(60.0)
            return await orig_synth(state)

        graph_module.synthesizer_node = pausing_synthesizer

    print(f"WORKER_STARTING:{target_run_id}", flush=True)
    finished_run = await execute_run(target_run_id)
    print(f"WORKER_FINISHED:{finished_run.status}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
