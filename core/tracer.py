"""Observability and trace telemetry for Research Agent Crew.

Provides real-time event recording, structured trace formatting,
and logging for the multi-agent graph execution.
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from core.state import TraceEvent, CriticEvaluation, Subtask


class TraceLogger:
    """Collects and standardizes observability telemetry events."""

    @staticmethod
    def now_iso() -> str:
        """Returns current UTC timestamp in ISO-8601 format."""
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def create_event(
        cls,
        step: str,
        event_type: str,
        summary: str,
        details: Dict[str, Any],
        iteration: int = 0
    ) -> TraceEvent:
        """Factory for building a TraceEvent instance."""
        return TraceEvent(
            timestamp=cls.now_iso(),
            step=step,
            event_type=event_type,
            iteration=iteration,
            summary=summary,
            details=details
        )

    @classmethod
    def log_planner_decomposition(
        cls,
        task: str,
        subtasks: List[Subtask],
        hypotheses: List[str],
        iteration: int = 0
    ) -> TraceEvent:
        """Log initial task breakdown into subtasks."""
        return cls.create_event(
            step="planner",
            event_type="decomposition",
            summary=f"Decomposed task into {len(subtasks)} subtasks",
            details={
                "task": task,
                "subtask_count": len(subtasks),
                "subtasks": [s.model_dump() for s in subtasks],
                "hypotheses": hypotheses,
            },
            iteration=iteration
        )

    @classmethod
    def log_planner_revision(
        cls,
        task: str,
        subtasks: List[Subtask],
        revision_notes: str,
        critic_guidance: List[str],
        iteration: int
    ) -> TraceEvent:
        """Log plan revision based on Critic feedback."""
        return cls.create_event(
            step="planner",
            event_type="reflection_revision",
            summary=f"Revised plan (Iteration {iteration}): {len(subtasks)} targeted subtasks",
            details={
                "task": task,
                "revision_notes": revision_notes,
                "critic_guidance": critic_guidance,
                "new_subtasks": [s.model_dump() for s in subtasks],
            },
            iteration=iteration
        )

    @classmethod
    def log_tool_call(
        cls,
        tool_name: str,
        tool_input: Dict[str, Any],
        subtask_id: str,
        iteration: int = 0
    ) -> TraceEvent:
        """Log invocation of an MCP tool."""
        return cls.create_event(
            step="executor",
            event_type="tool_call",
            summary=f"MCP Tool Call: {tool_name} (Subtask: {subtask_id})",
            details={
                "tool_name": tool_name,
                "tool_input": tool_input,
                "subtask_id": subtask_id
            },
            iteration=iteration
        )

    @classmethod
    def log_tool_result(
        cls,
        tool_name: str,
        status: str,
        duration_ms: float,
        result_summary: str,
        subtask_id: str,
        raw_result: Any = None,
        iteration: int = 0
    ) -> TraceEvent:
        """Log the result of an MCP tool execution."""
        return cls.create_event(
            step="executor",
            event_type="tool_result",
            summary=f"MCP Tool Result: {tool_name} [{status}] ({duration_ms:.1f}ms)",
            details={
                "tool_name": tool_name,
                "status": status,
                "duration_ms": duration_ms,
                "result_summary": result_summary,
                "subtask_id": subtask_id,
                "raw_result": raw_result
            },
            iteration=iteration
        )

    @classmethod
    def log_critic_evaluation(
        cls,
        evaluation: CriticEvaluation,
        iteration: int,
        deterministic_metrics: Optional[Dict[str, Any]] = None,
        caps_fired: Optional[List[str]] = None
    ) -> TraceEvent:
        """Log Critic multi-rubric evaluation including deterministic metrics and hard caps."""
        return cls.create_event(
            step="critic",
            event_type="rubric_evaluation",
            summary=f"Critic Verdict: {evaluation.verdict} (Score: {evaluation.total_score}/15)",
            details={
                "verdict": evaluation.verdict,
                "total_score": evaluation.total_score,
                "coverage_score": evaluation.coverage.score,
                "coverage_rationale": evaluation.coverage.rationale,
                "evidence_score": evaluation.evidence.score,
                "evidence_rationale": evaluation.evidence.rationale,
                "coherence_score": evaluation.coherence.score,
                "coherence_rationale": evaluation.coherence.rationale,
                "overall_feedback": evaluation.overall_feedback,
                "next_step_instructions": evaluation.next_step_instructions,
                "deterministic_metrics": deterministic_metrics or {},
                "caps_fired": caps_fired or []
            },
            iteration=iteration
        )

    @classmethod
    def log_budget_alert(
        cls,
        remaining: int,
        limit: int,
        action_taken: str,
        iteration: int
    ) -> TraceEvent:
        """Log budget constraint warnings or halts."""
        return cls.create_event(
            step="executor",
            event_type="budget_alert",
            summary=f"Search Budget Notice: {remaining}/{limit} calls remaining ({action_taken})",
            details={
                "remaining_budget": remaining,
                "total_budget": limit,
                "action": action_taken
            },
            iteration=iteration
        )

    @classmethod
    def log_escalation(
        cls,
        reason: str,
        retry_count: int,
        iteration: int
    ) -> TraceEvent:
        """Log graceful termination and human escalation."""
        return cls.create_event(
            step="critic",
            event_type="escalation",
            summary=f"Human Escalation Triggered: Max retries ({retry_count}) exceeded",
            details={
                "reason": reason,
                "retry_count": retry_count
            },
            iteration=iteration
        )
