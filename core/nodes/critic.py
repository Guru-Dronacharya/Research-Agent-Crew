"""Critic Agent Node for Research Agent Crew.

Implements hybrid evaluation combining:
1. Deterministic quality checks (citation validity, source relevance, evidence gap rate,
   coverage map, budget verification).
2. LLM rubric judge (Coverage, Evidence, Coherence).
3. Strict code-enforced scoring caps and verdict determination.
4. Actionable revision directives and budget-aware escalation.
"""

from typing import Dict, Any, List
from core.state import AgentState, CriticEvaluation, CriticRubricScore
from core.llm import get_critic_llm
from core.tracer import TraceLogger
from core.critic_checks import run_deterministic_critic_checks
from core.timing import timed_node
from core.ratelimit import QuotaExceededError
from core.cancellation import is_cancellation_requested, RunCancelledError


@timed_node("critic")
async def critic_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node: Critic Agent.

    Conducts deterministic audits and LLM judge scoring, applies hard scoring caps,
    and produces actionable next-step directives for the Planner.
    """
    if state.get("human_escalation_needed"):
        return {"status": "escalated"}

    run_id = state.get("run_id")
    if await is_cancellation_requested(run_id):
        raise RunCancelledError(f"Run {run_id} cancelled by user")

    task = state["task"]
    plan = state["plan"]
    draft_report = state.get("draft_report", "") or ""
    subtask_results = state.get("subtask_results", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    trace_events = list(state.get("trace_events", []))
    critic_evals = list(state.get("critic_evaluations", []))
    total_searches = state.get("total_searches_performed", 0)
    max_budget = state.get("max_search_budget", 10)
    reserve_remaining = state.get("revision_reserve_remaining", 0)

    # 1. Run pure deterministic audits from state
    deterministic_metrics = run_deterministic_critic_checks(state)
    citation_validity = deterministic_metrics["citation_validity"]
    source_relevance = deterministic_metrics["source_relevance"]
    evidence_gap_rate = deterministic_metrics["evidence_gap_rate"]
    budget_respected = deterministic_metrics["budget_respected"]

    # 2. Check if a hard-fail condition is already certain
    hard_fail_certain = (
        citation_validity < 1.0
        or evidence_gap_rate > 0.34
        or not budget_respected
    )

    caps_fired: List[str] = []

    if hard_fail_certain:
        # Build FAIL evaluation from deterministic checks only; skip LLM judge
        if citation_validity < 1.0:
            caps_fired.append(f"Evidence capped at 2 (citation_validity {citation_validity:.2f} < 1.0)")
        if evidence_gap_rate > 0.34:
            caps_fired.append(f"Coverage capped at 2 (evidence_gap_rate {evidence_gap_rate:.2f} > 0.34)")
        if source_relevance < 0.6:
            caps_fired.append(f"Evidence capped at 3 (source_relevance {source_relevance:.2f} < 0.6)")

        cov_score = 2 if evidence_gap_rate > 0.34 else 3
        evi_score = 2 if citation_validity < 1.0 else (3 if source_relevance < 0.6 else 3)
        coh_score = 3
        total_score = cov_score + evi_score + coh_score

        cov_rationale = (
            f"[Hard Cap Applied: Coverage capped at 2 because evidence gap rate ({evidence_gap_rate:.2f}) > 0.34]"
            if evidence_gap_rate > 0.34
            else "Evaluated from deterministic audit metrics."
        )

        evi_rationale = (
            f"[Hard Cap Applied: Evidence capped at 2 because citation validity ({citation_validity:.2f}) < 1.0]"
            if citation_validity < 1.0
            else (
                f"[Hard Cap Applied: Evidence capped at 3 because source relevance ({source_relevance:.2f}) < 0.6]"
                if source_relevance < 0.6
                else "Evaluated from deterministic audit metrics."
            )
        )

        reasons = []
        if citation_validity < 1.0:
            reasons.append(f"Citation validity ({citation_validity:.2f}) < 1.0")
        if evidence_gap_rate > 0.34:
            reasons.append(f"Evidence gap rate ({evidence_gap_rate:.2f}) > 0.34")
        if not budget_respected:
            reasons.append("Search budget ceiling violated")

        feedback = f"Deterministic audit hard-fail: {', '.join(reasons)}. LLM judge skipped."

        deterministic_insts = deterministic_metrics.get("deterministic_instructions", [])
        combined_instructions: List[str] = []
        for inst in deterministic_insts:
            if inst and inst not in combined_instructions:
                combined_instructions.append(inst)

        if not combined_instructions:
            combined_instructions.append("Conduct targeted web search to gather missing empirical evidence for open subtasks.")
            combined_instructions.append("Verify citations against retrieved sources and replace unsupported claims.")

        evaluation = CriticEvaluation(
            iteration=retry_count,
            coverage=CriticRubricScore(
                dimension="Coverage",
                score=cov_score,
                rationale=cov_rationale
            ),
            evidence=CriticRubricScore(
                dimension="Evidence",
                score=evi_score,
                rationale=evi_rationale
            ),
            coherence=CriticRubricScore(
                dimension="Coherence",
                score=coh_score,
                rationale="Draft report structure evaluated from deterministic audit metrics."
            ),
            total_score=total_score,
            verdict="FAIL",
            overall_feedback=feedback,
            next_step_instructions=combined_instructions[:3],
            judge_skipped=True
        )
    else:
        # Deterministic checks did not force a hard-fail: invoke LLM Judge
        model_name = state.get("model_name")
        llm = get_critic_llm(model_name=model_name)
        try:
            evaluation: CriticEvaluation = await llm.evaluate_report(
                task=task,
                plan=plan,
                report=draft_report,
                subtask_results=subtask_results,
                iteration=retry_count,
                deterministic_metrics=deterministic_metrics,
                structured_report=state.get("draft_report_structured")
            )
            evaluation.judge_skipped = False
        except QuotaExceededError as exc:
            if state.get("run_id"):
                raise
            escalation_reason = f"Daily LLM quota exceeded: {str(exc)}"
            trace_events.append(
                TraceLogger.log_escalation(
                    reason=escalation_reason,
                    retry_count=retry_count,
                    iteration=retry_count
                )
            )
            return {
                "status": "escalated",
                "human_escalation_needed": True,
                "escalation_reason": escalation_reason,
                "trace_events": trace_events
            }

        # Apply Hard Scoring Caps in code
        if citation_validity < 1.0:
            if evaluation.evidence.score > 2:
                evaluation.evidence.score = 2
                cap_note = f"[Hard Cap Applied: Evidence capped at 2 because citation validity ({citation_validity:.2f}) < 1.0]"
                evaluation.evidence.rationale = f"{evaluation.evidence.rationale} {cap_note}".strip()
                caps_fired.append(f"Evidence capped at 2 (citation_validity {citation_validity:.2f} < 1.0)")

        if evidence_gap_rate > 0.34:
            if evaluation.coverage.score > 2:
                evaluation.coverage.score = 2
                cap_note = f"[Hard Cap Applied: Coverage capped at 2 because evidence gap rate ({evidence_gap_rate:.2f}) > 0.34]"
                evaluation.coverage.rationale = f"{evaluation.coverage.rationale} {cap_note}".strip()
                caps_fired.append(f"Coverage capped at 2 (evidence_gap_rate {evidence_gap_rate:.2f} > 0.34)")

        if source_relevance < 0.6:
            if evaluation.evidence.score > 3:
                evaluation.evidence.score = 3
                cap_note = f"[Hard Cap Applied: Evidence capped at 3 because source relevance ({source_relevance:.2f}) < 0.6]"
                evaluation.evidence.rationale = f"{evaluation.evidence.rationale} {cap_note}".strip()
                caps_fired.append(f"Evidence capped at 3 (source_relevance {source_relevance:.2f} < 0.6)")

        evaluation.total_score = (
            evaluation.coverage.score + evaluation.evidence.score + evaluation.coherence.score
        )

    # 4. Strict Pass/Fail Determination
    # PASS only if every dimension >= 4, total >= 12, citation_validity == 1.0, and budget_respected
    is_pass = (
        evaluation.coverage.score >= 4
        and evaluation.evidence.score >= 4
        and evaluation.coherence.score >= 4
        and evaluation.total_score >= 12
        and citation_validity == 1.0
        and budget_respected
    )

    if is_pass:
        evaluation.verdict = "PASS"
        status = "completed"
        human_needed = False
        escalation_reason = None
    else:
        evaluation.verdict = "FAIL"

        # Combine deterministic instructions with judge directives, de-duplicated
        deterministic_insts = deterministic_metrics.get("deterministic_instructions", [])
        judge_insts = evaluation.next_step_instructions or []
        combined_instructions: List[str] = []
        for inst in deterministic_insts + judge_insts:
            if inst and inst not in combined_instructions:
                combined_instructions.append(inst)

        # Ensure we have at least 2-3 concrete actionable instructions
        if not combined_instructions:
            combined_instructions.append("Conduct targeted web search to gather missing empirical evidence for open subtasks.")
            combined_instructions.append("Verify citations against retrieved sources and replace unsupported claims.")

        evaluation.next_step_instructions = combined_instructions[:3]

    # 5. Log structured observability trace (rubric_evaluation MUST precede escalation)
    critic_evals.append(evaluation)
    trace_events.append(
        TraceLogger.log_critic_evaluation(
            evaluation=evaluation,
            iteration=evaluation.iteration,
            deterministic_metrics=deterministic_metrics,
            caps_fired=caps_fired
        )
    )

    if not is_pass:
        # Check revision reserve and retry counts
        if reserve_remaining <= 0:
            # Out of revision search budget
            status = "escalated"
            human_needed = True
            escalation_reason = (
                f"Search budget exhausted ({total_searches}/{max_budget} searches performed, "
                f"0 revision reserve remaining). Unable to perform further revision loops."
            )
            trace_events.append(
                TraceLogger.log_escalation(
                    reason=escalation_reason,
                    retry_count=retry_count,
                    iteration=retry_count
                )
            )
        elif retry_count < max_retries:
            status = "revising"
            retry_count += 1
            human_needed = False
            escalation_reason = None
        else:
            # Exceeded maximum allowed retries
            status = "escalated"
            human_needed = True
            escalation_reason = (
                f"Evaluation threshold unmet after {retry_count} retries "
                f"(Final Score: {evaluation.total_score}/15). "
                f"This task may require human guidance or additional primary sources."
            )
            trace_events.append(
                TraceLogger.log_escalation(
                    reason=escalation_reason,
                    retry_count=retry_count,
                    iteration=retry_count
                )
            )

    return {
        "critic_evaluations": critic_evals,
        "retry_count": retry_count,
        "status": status,
        "human_escalation_needed": human_needed,
        "escalation_reason": escalation_reason,
        "trace_events": trace_events
    }
