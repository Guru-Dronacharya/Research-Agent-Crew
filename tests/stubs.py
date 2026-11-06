"""Test doubles and stubs for unit and integration testing.

Contains StubLLM (test double) used exclusively under pytest to simulate
agent reasoning deterministically without external API calls.
"""

import re
from typing import Optional, List, Dict, Any, Tuple
from core.state import (
    ResearchPlan, Subtask, CriticEvaluation, CriticRubricScore, SubtaskResult,
    Claim, Section, ComparisonRow, StructuredReport
)
from core.report_render import validate_structured_report, render_structured_report_to_markdown
from core.llm import BaseAgentLLM
from core.ratelimit import get_llm_rate_limiter


class StubLLM(BaseAgentLLM):
    """Deterministic test double for agent reasoning in automated test suites."""

    def __init__(self):
        self.scripted_streaming_deltas: Optional[List[str]] = None
        self.simulate_pre_stream_quota_error: bool = False
        self.simulate_mid_stream_quota_error: bool = False
        self.stream_delay_seconds: float = 0.0
        self.stream_calls_count: int = 0

    async def astream_chat(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None
    ):
        """Deterministic streaming chat generator for tests."""
        import asyncio
        from core.ratelimit import QuotaExceededError

        self.stream_calls_count += 1
        await get_llm_rate_limiter().acquire(model=model_name or "stub-model")

        if self.simulate_pre_stream_quota_error:
            raise QuotaExceededError(
                "Google Gemini daily quota exceeded for model 'stub-model'. Resets at midnight Pacific Time (PT).",
                reset_time="midnight Pacific Time (PT)"
            )

        if self.scripted_streaming_deltas is not None:
            deltas = list(self.scripted_streaming_deltas)
        else:
            last_msg = messages[-1].get("content", "") if messages else ""
            deltas = [
                f"Answer for '{last_msg[:30]}': ",
                "Supported by evidence [chunk-1] ",
                "and further verified by [chunk-2]."
            ]

        if self.simulate_mid_stream_quota_error:
            # Yield partial tokens before quota error
            if deltas:
                yield deltas[0]
            if len(deltas) > 1:
                yield deltas[1]
            raise QuotaExceededError(
                "Google Gemini daily quota exceeded mid-stream for model 'stub-model'. Resets at midnight Pacific Time (PT).",
                reset_time="midnight Pacific Time (PT)"
            )

        for d in deltas:
            if self.stream_delay_seconds > 0:
                await asyncio.sleep(self.stream_delay_seconds)
            yield d

    async def generate_plan(
        self,
        task: str,
        critic_evaluation: Optional[CriticEvaluation] = None,
        iteration: int = 0,
        subtask_results: Optional[List[SubtaskResult]] = None
    ) -> ResearchPlan:
        await get_llm_rate_limiter().acquire(model="stub-model")
        words = [w for w in re.findall(r"\w+", task) if len(w) > 3]
        topic = " ".join(words[:4]) if words else "Target Domain"

        if critic_evaluation and critic_evaluation.verdict == "FAIL":
            subtasks = [
                Subtask(
                    id="subtask_revised_1",
                    title=f"Deep Empirical Verification for {topic}",
                    description="Address critic directive: investigate benchmark metrics, empirical numbers, and performance latency.",
                    search_queries=[
                        f"{topic} performance benchmark metrics",
                        f"{topic} empirical evaluation 2024 2025"
                    ],
                    rationale="Directly fulfills Critic's evidence and empirical grounding requirements."
                ),
                Subtask(
                    id="subtask_revised_2",
                    title=f"Production Trade-offs & Edge Cases in {topic}",
                    description="Analyze operational complexity, fault-tolerance, and scaling bottlenecks.",
                    search_queries=[
                        f"{topic} production failure modes trade-offs",
                        f"{topic} best practices enterprise architecture"
                    ],
                    rationale="Enhances coverage and coherence by addressing production constraints."
                )
            ]
            return ResearchPlan(
                task_summary=f"Revised Research Strategy for: '{task}' (incorporating Critic feedback)",
                subtasks=subtasks,
                key_hypotheses=[
                    "Concrete benchmark data will satisfy the Critic's Evidence threshold.",
                    "Operational trade-off analysis will elevate Coverage to >= 4/5."
                ],
                estimated_searches=4,
                revision_notes=f"Planner revised scope following Iteration {iteration - 1} Critic evaluation: {critic_evaluation.overall_feedback}"
            )
        else:
            subtasks = [
                Subtask(
                    id="subtask_1",
                    title=f"Core Foundations and Architecture of {topic}",
                    description="Analyze fundamental design principles, architectural components, and core state mechanics.",
                    search_queries=[
                        f"{topic} architecture overview",
                        f"{topic} core components state management"
                    ],
                    rationale="Establishes structural baseline and key technical taxonomy."
                ),
                Subtask(
                    id="subtask_2",
                    title="Comparative Analysis & Production Frameworks",
                    description="Compare leading paradigms, performance trade-offs, and ecosystem integrations.",
                    search_queries=[
                        f"{topic} comparison vs alternatives",
                        f"{topic} production pros cons"
                    ],
                    rationale="Provides multidimensional comparison essential for decision makers."
                ),
                Subtask(
                    id="subtask_3",
                    title="Real-World Deployment & Future Trajectory",
                    description="Examine enterprise adoption case studies, current limitations, and upcoming trends.",
                    search_queries=[
                        f"{topic} case studies deployment",
                        f"{topic} roadmap 2025 2026"
                    ],
                    rationale="Ensures comprehensive forward-looking coverage of the research query."
                )
            ]
            return ResearchPlan(
                task_summary=f"Systematic technical investigation into '{task}'",
                subtasks=subtasks,
                key_hypotheses=[
                    f"Architecture decisions dictate scalability and failure-recovery in {topic}.",
                    "Multi-agent frameworks differ significantly in state persistence and human-in-the-loop controls."
                ],
                estimated_searches=5,
                revision_notes="Initial baseline decomposition generated by Planner."
            )

    async def synthesize_findings(
        self,
        task: str,
        plan: ResearchPlan,
        subtask_results: List[SubtaskResult],
        critic_evaluation: Optional[CriticEvaluation] = None,
        iteration: int = 0
    ) -> Tuple[str, StructuredReport]:
        await get_llm_rate_limiter().acquire(model="stub-model")

        sections = []
        open_gaps = []
        for res in subtask_results:
            claims = []
            if res.sources:
                for s in res.sources[:3]:
                    claims.append(
                        Claim(
                            text=f"{res.findings[:120].strip() or res.title} backed by evidence.",
                            source_urls=[s.url],
                            confidence=0.9
                        )
                    )
            else:
                open_gaps.append(f"{res.title}: No verified evidence retrieved.")

            sections.append(
                Section(
                    subtask_id=res.subtask_id,
                    heading=res.title,
                    summary=res.findings,
                    claims=claims
                )
            )

        comparison = None
        if "vs" in task.lower() or "compare" in task.lower():
            comparison = [
                ComparisonRow(dimension="Architecture", values={"LangGraph": "Stateful Graph", "AutoGen": "Conversational"}),
                ComparisonRow(dimension="Human-in-the-Loop", values={"LangGraph": "Built-in Breakpoints", "AutoGen": "UserProxyAgent"})
            ]

        structured = StructuredReport(
            title=f"Research Report: {task}",
            tldr=[
                f"Synthesizes key findings for {task} across {len(subtask_results)} subtasks.",
                "Primary architectural trade-offs evaluated against retrieved evidence."
            ][:5],
            sections=sections,
            comparison=comparison,
            recommendation=f"Adopt the framework best matching operational requirements for {task}.",
            open_gaps=open_gaps,
            limitations=["Analysis based on retrieved public documentation and technical whitepapers."]
        )

        all_sources = [s for r in subtask_results for s in r.sources]
        validated = validate_structured_report(structured, all_sources)
        rendered_md = render_structured_report_to_markdown(validated, all_sources)

        return rendered_md, validated

    async def evaluate_report(
        self,
        task: str,
        plan: ResearchPlan,
        report: str,
        subtask_results: List[SubtaskResult],
        iteration: int = 0,
        deterministic_metrics: Optional[Dict[str, Any]] = None,
        structured_report: Optional[StructuredReport] = None
    ) -> CriticEvaluation:
        await get_llm_rate_limiter().acquire(model="stub-model")
        if iteration == 0:
            coverage = CriticRubricScore(
                dimension="Coverage",
                score=3,
                rationale="The draft covers baseline architecture concepts but lacks empirical benchmark citations and specific production failure mode details."
            )
            evidence = CriticRubricScore(
                dimension="Evidence",
                score=3,
                rationale="While sources were discovered, numerical latency figures and concrete trade-off comparisons need stronger verification."
            )
            coherence = CriticRubricScore(
                dimension="Coherence",
                score=4,
                rationale="Report structure is clean and logically ordered, with clear section headers and an executive summary."
            )
            return CriticEvaluation(
                iteration=iteration,
                coverage=coverage,
                evidence=evidence,
                coherence=coherence,
                total_score=10,
                verdict="FAIL",
                overall_feedback="Good initial draft structure, but misses empirical benchmark data and concrete operational trade-off metrics.",
                next_step_instructions=[
                    "Conduct targeted search for empirical benchmarks, throughput, and latency metrics.",
                    "Add an in-depth operational trade-off matrix detailing production failure modes.",
                    "Strengthen citation links to primary documentation sources."
                ]
            )
        else:
            coverage = CriticRubricScore(
                dimension="Coverage",
                score=5,
                rationale="All core questions and subtasks thoroughly answered, including empirical trade-offs and edge cases."
            )
            evidence = CriticRubricScore(
                dimension="Evidence",
                score=5,
                rationale="Findings are thoroughly corroborated by verified URLs, comparative matrices, and concrete data points."
            )
            coherence = CriticRubricScore(
                dimension="Coherence",
                score=5,
                rationale="Exceptional narrative clarity, professional executive summary, and actionable recommendations."
            )
            return CriticEvaluation(
                iteration=iteration,
                coverage=coverage,
                evidence=evidence,
                coherence=coherence,
                total_score=15,
                verdict="PASS",
                overall_feedback="Report meets all quality criteria with high rigor, verified citations, and excellent coherence.",
                next_step_instructions=[]
            )


# Alias for backward compatibility in tests
FakeAgentLLM = StubLLM
