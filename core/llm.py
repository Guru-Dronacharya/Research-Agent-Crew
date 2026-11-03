"""LLM Provider and structured reasoning engine for Research Agent Crew.

Provides production integration for Google Gemini with structured outputs,
explicit configuration enforcement, retry policies with exponential backoff,
and dependency injection for testing.
"""
import os
import re
import time
from typing import Optional, List, Dict, Any, Tuple
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)
from tenacity.wait import wait_base

from core.state import (
    ResearchPlan, CriticEvaluation, SubtaskResult, StructuredReport
)
from core.report_render import validate_structured_report, render_structured_report_to_markdown
from core.timing import record_llm_call_timing
from core.ratelimit import get_llm_rate_limiter


class ConfigurationError(RuntimeError):
    """Raised when required environment configuration or API keys are missing."""
    pass


def _is_transient_error(exc: BaseException) -> bool:
    """Detect transient 429 / rate limit / network / server errors."""
    msg = str(exc).lower()
    transient_indicators = [
        "429", "rate limit", "resource_exhausted", "resourceexhausted",
        "quota", "503", "500", "overloaded", "temporary", "timeout",
        "connection reset", "econnreset", "try again"
    ]
    return any(indicator in msg for indicator in transient_indicators)


def _extract_retry_after(exc: BaseException) -> Optional[float]:
    """Extract retry-after or retryDelay seconds from error if present."""
    msg = str(exc)
    response = getattr(exc, "response", None)
    if response and hasattr(response, "headers"):
        retry_header = response.headers.get("retry-after")
        if retry_header and retry_header.isdigit():
            return float(retry_header)

    m = re.search(r"retry[-_]?after[:=\s]+(\d+(?:\.\d+)?)", msg, re.IGNORECASE)
    if m:
        return float(m.group(1))

    m2 = re.search(r"retryDelay['\":\s]+(\d+(?:\.\d+)?)s?", msg, re.IGNORECASE)
    if m2:
        return float(m2.group(1))

    m3 = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
    if m3:
        return float(m3.group(1))

    return None


class wait_with_retry_after(wait_base):
    """Tenacity wait strategy that respects Retry-After with fallback to exponential backoff."""
    def __init__(self, fallback_strategy, max_retry_after: float = 30.0):
        self.fallback = fallback_strategy
        self.max_retry_after = max_retry_after

    def __call__(self, retry_state):
        if retry_state.outcome and retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            delay = _extract_retry_after(exc)
            if delay is not None and delay > 0:
                return min(delay, self.max_retry_after)
        return self.fallback(retry_state)


llm_retry = retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_with_retry_after(wait_exponential(multiplier=1, min=2, max=10)),
    retry=retry_if_exception(_is_transient_error)
)


class BaseAgentLLM:
    """Interface for agent LLM reasoning."""

    async def generate_plan(
        self,
        task: str,
        critic_evaluation: Optional[CriticEvaluation] = None,
        iteration: int = 0,
        subtask_results: Optional[List[SubtaskResult]] = None
    ) -> ResearchPlan:
        raise NotImplementedError

    async def synthesize_findings(
        self,
        task: str,
        plan: ResearchPlan,
        subtask_results: List[SubtaskResult],
        critic_evaluation: Optional[CriticEvaluation] = None,
        iteration: int = 0
    ) -> Any:
        raise NotImplementedError

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
        raise NotImplementedError


class GeminiAgentLLM(BaseAgentLLM):
    """Production LLM using Google Gemini with structured outputs."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2
    ):
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )
        if not self.api_key:
            raise ConfigurationError(
                "GEMINI_API_KEY is required. Please set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or .env file."
            )

        resolved_model = (model_name or os.environ.get("GEMINI_MODEL", "")).strip()
        if not resolved_model:
            raise ConfigurationError(
                "GEMINI_MODEL environment variable is not set. Please specify a valid Gemini model in your .env file."
            )

        self.model_name = resolved_model
        self.temperature = temperature
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.temperature
        )

    async def _invoke_structured(self, schema_cls: Any, prompt: str) -> Any:
        """Invoke LLM with structured output, handling parsing_error with single retry."""
        await get_llm_rate_limiter().acquire()
        structured_llm = self.llm.with_structured_output(schema_cls, include_raw=True)

        @llm_retry
        async def _call_once():
            return await structured_llm.ainvoke(prompt)

        result_dict = await _call_once()

        # Handle include_raw=True dict: {"raw": ..., "parsed": ..., "parsing_error": ...}
        if isinstance(result_dict, dict) and "parsed" in result_dict:
            parsing_err = result_dict.get("parsing_error")
            parsed_obj = result_dict.get("parsed")
            if parsing_err is not None or parsed_obj is None:
                # Retry once as per specification
                retry_dict = await _call_once()
                if retry_dict.get("parsing_error") is not None or retry_dict.get("parsed") is None:
                    raise ValueError(
                        f"Structured output parsing failed after retry: {retry_dict.get('parsing_error')}"
                    )
                return retry_dict["parsed"]
            return parsed_obj
        return result_dict

    async def generate_plan(
        self,
        task: str,
        critic_evaluation: Optional[CriticEvaluation] = None,
        iteration: int = 0,
        subtask_results: Optional[List[SubtaskResult]] = None
    ) -> ResearchPlan:
        prompt = f"""You are the Lead Research Planner in a multi-agent research crew.
Your goal is to break down the following research task into 3 to 4 sequential, focused subtasks.
Each subtask must include targeted search queries and clear rationale.

Master Task: "{task}"
"""
        if critic_evaluation and critic_evaluation.verdict == "FAIL":
            gaps_info = ""
            satisfied_info = ""
            if subtask_results:
                gap_subtasks = [r for r in subtask_results if getattr(r, "evidence_gap", False) or not r.sources]
                satisfied_subtasks = [r for r in subtask_results if not getattr(r, "evidence_gap", False) and r.sources]
                if gap_subtasks:
                    gaps_info = "\nSubtasks with Identified Evidence Gaps:\n" + "\n".join(
                        f"- [{r.subtask_id}] {r.title}: {r.findings}" for r in gap_subtasks
                    )
                if satisfied_subtasks:
                    satisfied_info = "\nAlready Satisfied Subtasks (DO NOT REPEAT):\n" + "\n".join(
                        f"- [{r.subtask_id}] {r.title} ({len(r.sources)} sources retrieved)" for r in satisfied_subtasks
                    )

            prompt += f"""
CRITIC FEEDBACK FROM PREVIOUS ITERATION (Iteration {iteration - 1}):
- Overall Feedback: {critic_evaluation.overall_feedback}
- Coverage Score: {critic_evaluation.coverage.score}/5 ({critic_evaluation.coverage.rationale})
- Evidence Score: {critic_evaluation.evidence.score}/5 ({critic_evaluation.evidence.rationale})
- Coherence Score: {critic_evaluation.coherence.score}/5 ({critic_evaluation.coherence.rationale})
- Required Next Steps:
{chr(10).join(f'  * {step}' for step in critic_evaluation.next_step_instructions)}
{gaps_info}
{satisfied_info}

CRITICAL: Revise the research plan to address these specific gaps and missing evidence.
Produce focused subtasks ONLY for identified gaps. Do not repeat already-satisfied subtasks.
Explain your changes in the revision_notes field.
"""
        t0 = time.perf_counter()
        plan = await self._invoke_structured(ResearchPlan, prompt)
        record_llm_call_timing("generate_plan", self.model_name, (time.perf_counter() - t0) * 1000.0)
        return plan

    async def synthesize_findings(
        self,
        task: str,
        plan: ResearchPlan,
        subtask_results: List[SubtaskResult],
        critic_evaluation: Optional[CriticEvaluation] = None,
        iteration: int = 0
    ) -> Tuple[str, StructuredReport]:
        t0 = time.perf_counter()

        try:
            max_sources = int(os.environ.get("SYNTH_MAX_SOURCES_PER_SUBTASK", "4"))
        except ValueError:
            max_sources = 4

        try:
            max_snippet_chars = int(os.environ.get("SYNTH_SNIPPET_MAX_CHARS", "300"))
        except ValueError:
            max_snippet_chars = 300

        # Build evidence text respecting limits
        evidence_blocks = []
        for r in subtask_results:
            top_sources = r.sources[:max_sources]
            source_lines = []
            for s in top_sources:
                snip = (s.snippet or "")[:max_snippet_chars].strip()
                source_lines.append(f"  * [{s.title}]({s.url}): {snip}")
            src_str = "\n".join(source_lines) if source_lines else "  * (No retrieved sources - evidence gap)"
            evidence_blocks.append(
                f"### Subtask: {r.title} (ID: {r.subtask_id})\nFindings: {r.findings}\nRetrieved Sources:\n{src_str}"
            )
        evidence_text = "\n\n".join(evidence_blocks)

        critic_context = ""
        if critic_evaluation and critic_evaluation.verdict == "FAIL":
            critic_context = f"""
PREVIOUS CRITIC FEEDBACK (Iteration {iteration - 1}):
- Overall: {critic_evaluation.overall_feedback}
- Directives:
{chr(10).join(f'  * {inst}' for inst in critic_evaluation.next_step_instructions)}
"""

        synth_prompt = f"""You are the Lead Research Synthesizer. Synthesize the findings and verified evidence into a StructuredReport.

Master Research Objective: "{task}"
Plan Summary: "{plan.task_summary if plan else task}"
{critic_context}

Available Evidence by Subtask:
{evidence_text}

CRITICAL INSTRUCTIONS:
1. Grounding: Use ONLY the provided evidence and sources above.
2. Citations: Every claim in `claims` must cite provided URLs in `source_urls` (minimum 1 URL per claim). DO NOT hallucinate or alter URLs.
3. Comparison: Include the `comparison` matrix ONLY if the master research objective compares different items, options, or technologies. Otherwise leave `comparison` as null.
4. Open Gaps: Any unsupported topics, missing data, or items lacking verifiable evidence MUST be placed in `open_gaps`.
5. Executive Takeaways: `tldr` must contain at most 5 concise, high-impact takeaway bullet points.
6. Provide accurate confidence scores (0.0 to 1.0) for each claim based on source authority and clarity.
"""
        raw_structured: StructuredReport = await self._invoke_structured(StructuredReport, synth_prompt)

        all_retrieved_sources = [s for r in subtask_results for s in r.sources]
        validated_structured = validate_structured_report(raw_structured, all_retrieved_sources)
        rendered_md = render_structured_report_to_markdown(validated_structured, all_retrieved_sources)

        record_llm_call_timing("synthesize_findings", self.model_name, (time.perf_counter() - t0) * 1000.0)
        return rendered_md, validated_structured

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
        # LLM judge prompt for Part B
        metrics_summary = ""
        if deterministic_metrics:
            claim_info = ""
            if "claim_count" in deterministic_metrics:
                claim_info = f"""- Total Claims: {deterministic_metrics.get('claim_count', 0)}
- Claims per Subtask: {deterministic_metrics.get('claims_per_subtask', {})}
- Low-Confidence Claim Ratio: {deterministic_metrics.get('low_confidence_ratio', 0.0):.2f} ({deterministic_metrics.get('low_confidence_claims_count', 0)} claims < 0.70 confidence)"""

            metrics_summary = f"""
Deterministic Audit Metrics (Pre-computed from State):
- Citation Validity: {deterministic_metrics.get('citation_validity', 0.0):.2f} (1.0 = all report URLs exist in retrieved sources)
- Source Relevance: {deterministic_metrics.get('source_relevance', 0.0):.2f}
- Evidence Gap Rate: {deterministic_metrics.get('evidence_gap_rate', 0.0):.2f} ({deterministic_metrics.get('gap_subtasks_count', 0)} subtasks with evidence_gap)
- Min Sources Count: {deterministic_metrics.get('min_sources', 0)} unique relevant sources
- Coverage Map: {deterministic_metrics.get('coverage_map', {})}
- Budget Respected: {deterministic_metrics.get('budget_respected', True)}
{claim_info}
"""

        structured_summary = ""
        if structured_report:
            structured_summary = f"""
Structured Report Overview:
- Title: {structured_report.title}
- Executive Takeaways: {len(structured_report.tldr)} items
- Finding Sections: {len(structured_report.sections)}
- Comparison Included: {bool(structured_report.comparison)}
- Open Gaps: {len(structured_report.open_gaps)}
- Limitations: {len(structured_report.limitations)}
"""

        prompt = f"""You are a rigorous, independent LLM Judge evaluating a research report.
Your role is anti-sycophantic: PASS is rare and reserved only for reports with verified facts, zero hallucinations, and thorough coverage.
Penalize padded, generic, or boilerplate text severely. You must quote specific report lines supporting each score.

Master Research Task: "{task}"

Plan Subtasks:
{chr(10).join(f'- [{s.id}] {s.title}: {s.description}' for s in plan.subtasks)}

{metrics_summary}
{structured_summary}

Draft Report to Evaluate:
{report}

Rubric Scoring Guidelines:
Score each dimension from 0 to 5 using the following anchors:
1. Coverage (0-5):
   - 1: Major subtasks ignored; superficial coverage; obvious questions unaddressed.
   - 3: Most subtasks mentioned, but significant gaps exist or treated shallowly.
   - 5: Every subtask comprehensively answered with specific architectural details and trade-offs.
2. Evidence (0-5):
   - Base Evidence on checkable facts rather than tone.
   - 1: Unsubstantiated claims, hallucinated links, or reliance on unsupported assertions.
   - 3: Some citations present, but weak empirical data or missing primary documentation.
   - 5: Every claim backed by verified citations matching retrieved sources with concrete technical data.
3. Coherence (0-5):
   - 1: Disjointed, redundant filler, inconsistent structure, or templated boilerplate.
   - 3: Legible but contains generic padding or awkward transitions.
   - 5: Flawless narrative structure, logical synthesis, and crystal-clear presentation.

Provide concise justifications (at most 2 sentences per dimension) along with the concrete quoted line from the report supporting each score.
If any score is below 4, provide 2 to 3 actionable next_step_instructions specifying exactly what evidence or subtasks must be addressed.
"""
        t0 = time.perf_counter()
        evaluation = await self._invoke_structured(CriticEvaluation, prompt)
        record_llm_call_timing("evaluate_report", self.model_name, (time.perf_counter() - t0) * 1000.0)
        evaluation.iteration = iteration
        evaluation.total_score = (
            evaluation.coverage.score + evaluation.evidence.score + evaluation.coherence.score
        )
        return evaluation


# Global injectable override for test suites
_GLOBAL_LLM_OVERRIDE: Optional[BaseAgentLLM] = None


def set_global_llm(llm: Optional[BaseAgentLLM]) -> None:
    """Inject an LLM instance (such as StubLLM in test suites)."""
    global _GLOBAL_LLM_OVERRIDE
    _GLOBAL_LLM_OVERRIDE = llm


def resolve_model_name(role_env_var: str, explicit_override: Optional[str] = None) -> str:
    """Resolve model name for a specific agent role following priority:
    1. Explicit runtime override (if provided)
    2. Role-specific env var (PLANNER_MODEL, SYNTH_MODEL, CRITIC_MODEL)
    3. Global GEMINI_MODEL
    """
    if explicit_override and explicit_override.strip():
        return explicit_override.strip()

    role_model = os.environ.get(role_env_var, "").strip()
    if role_model:
        return role_model

    gemini_model = os.environ.get("GEMINI_MODEL", "").strip()
    if not gemini_model:
        raise ConfigurationError(
            f"GEMINI_MODEL environment variable is not set. Please set GEMINI_MODEL (or {role_env_var}) in your environment or .env file."
        )
    return gemini_model


def get_planner_llm(model_name: Optional[str] = None) -> BaseAgentLLM:
    """Factory to acquire the Planner agent LLM (PLANNER_MODEL or GEMINI_MODEL)."""
    if _GLOBAL_LLM_OVERRIDE is not None:
        return _GLOBAL_LLM_OVERRIDE

    target_model = resolve_model_name("PLANNER_MODEL", model_name)
    gemini_key = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if not gemini_key:
        raise ConfigurationError(
            "GEMINI_API_KEY is required. Please set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or .env file."
        )

    return GeminiAgentLLM(model_name=target_model, api_key=gemini_key, temperature=0.2)


def get_synth_llm(model_name: Optional[str] = None) -> BaseAgentLLM:
    """Factory to acquire the Synthesizer agent LLM (SYNTH_MODEL or GEMINI_MODEL)."""
    if _GLOBAL_LLM_OVERRIDE is not None:
        return _GLOBAL_LLM_OVERRIDE

    target_model = resolve_model_name("SYNTH_MODEL", model_name)
    gemini_key = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if not gemini_key:
        raise ConfigurationError(
            "GEMINI_API_KEY is required. Please set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or .env file."
        )

    return GeminiAgentLLM(model_name=target_model, api_key=gemini_key, temperature=0.2)


def get_agent_llm(model_name: Optional[str] = None) -> BaseAgentLLM:
    """Factory to acquire the primary agent LLM.

    Returns the Gemini implementation or raises ConfigurationError if keys or model are missing.
    Can be overridden via set_global_llm for testing.
    """
    if _GLOBAL_LLM_OVERRIDE is not None:
        return _GLOBAL_LLM_OVERRIDE

    target_model = resolve_model_name("GEMINI_MODEL", model_name)
    gemini_key = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if not gemini_key:
        raise ConfigurationError(
            "GEMINI_API_KEY is required. Please set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or .env file."
        )

    return GeminiAgentLLM(model_name=target_model, api_key=gemini_key, temperature=0.2)


def get_critic_llm(model_name: Optional[str] = None) -> BaseAgentLLM:
    """Factory to acquire the Critic LLM judge (CRITIC_MODEL or GEMINI_MODEL, temperature 0.0)."""
    if _GLOBAL_LLM_OVERRIDE is not None:
        return _GLOBAL_LLM_OVERRIDE

    target_model = resolve_model_name("CRITIC_MODEL", model_name)
    gemini_key = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )
    if not gemini_key:
        raise ConfigurationError(
            "GEMINI_API_KEY is required. Please set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or .env file."
        )

    return GeminiAgentLLM(model_name=target_model, api_key=gemini_key, temperature=0.0)
