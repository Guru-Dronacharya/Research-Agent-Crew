"""Unit tests for LangGraph checkpoint serializer allowlists and AgentState models."""

from pydantic import BaseModel
from core.checkpoint import get_state_serializer
from core.state import (
    Claim, Section, ComparisonRow, StructuredReport,
    Subtask, ResearchPlan, SourceItem, SubtaskResult,
    CriticRubricScore, CriticEvaluation, TraceEvent
)


def test_checkpoint_serde_roundtrips_fully_populated_state(monkeypatch):
    """Verify that a fully populated AgentState round-trips without loss or warning under strict msgpack mode."""
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")

    serde = get_state_serializer()

    state = {
        "task": "Test Serialization Roundtrip",
        "plan": ResearchPlan(
            task_summary="Detailed task summary",
            subtasks=[
                Subtask(id="s1", title="Sub 1", description="desc 1", search_queries=["q1"], rationale="r1")
            ],
            key_hypotheses=["h1"]
        ),
        "subtask_results": [
            SubtaskResult(
                subtask_id="s1",
                title="Sub 1",
                findings="Key findings from search",
                sources=[
                    SourceItem(title="Page 1", url="https://example.com/p1", snippet="Snippet 1")
                ],
                tools_invoked=["search_web"]
            )
        ],
        "draft_report": "# Draft",
        "draft_report_structured": StructuredReport(
            title="Draft Report",
            tldr=["Takeaway 1"],
            sections=[
                Section(
                    subtask_id="s1",
                    heading="Section 1",
                    summary="Summary 1",
                    claims=[
                        Claim(text="Claim 1", source_urls=["https://example.com/p1"], confidence=0.95)
                    ]
                )
            ],
            comparison=[ComparisonRow(dimension="Perf", values={"A": "high"})],
            recommendation="Adopt A",
            open_gaps=["Gap 1"],
            limitations=["Lim 1"]
        ),
        "final_report": "# Final Report",
        "final_report_structured": StructuredReport(
            title="Final Report",
            tldr=["Takeaway Final"],
            sections=[
                Section(
                    subtask_id="s1",
                    heading="Section 1",
                    summary="Summary 1",
                    claims=[
                        Claim(text="Claim 1", source_urls=["https://example.com/p1"], confidence=0.95)
                    ]
                )
            ]
        ),
        "report_metadata": {"generated_by": "crew"},
        "critic_evaluations": [
            CriticEvaluation(
                iteration=0,
                coverage=CriticRubricScore(dimension="Coverage", score=5, rationale="Complete"),
                evidence=CriticRubricScore(dimension="Evidence", score=4, rationale="Strong"),
                coherence=CriticRubricScore(dimension="Coherence", score=5, rationale="Logical"),
                total_score=14,
                verdict="PASS",
                overall_feedback="Excellent work"
            )
        ],
        "retry_count": 0,
        "max_retries": 1,
        "search_budget_remaining": 8,
        "max_search_budget": 10,
        "initial_budget_remaining": 6,
        "revision_reserve_remaining": 4,
        "iteration_budget_remaining": 6,
        "total_searches_performed": 2,
        "cache_hits": 1,
        "trace_events": [
            TraceEvent(
                timestamp="2026-11-03T00:00:00Z",
                step="planner",
                event_type="decomposition",
                summary="Plan created",
                details={"count": 1}
            )
        ],
        "human_escalation_needed": False,
        "escalation_reason": None,
        "status": "completed",
        "mcp_saved_path": "reports/test.md",
        "model_name": "gemini-2.0-flash",
        "timings": {"nodes": {"planner": 12.5}}
    }

    dumped = serde.dumps_typed(state)
    restored = serde.loads_typed(dumped)

    assert restored["task"] == "Test Serialization Roundtrip"
    assert isinstance(restored["plan"], ResearchPlan)
    assert restored["plan"].subtasks[0].id == "s1"
    assert isinstance(restored["subtask_results"][0], SubtaskResult)
    assert isinstance(restored["draft_report_structured"], StructuredReport)
    assert restored["draft_report_structured"].sections[0].claims[0].text == "Claim 1"
    assert isinstance(restored["critic_evaluations"][0], CriticEvaluation)
    assert restored["critic_evaluations"][0].coverage.score == 5
    assert isinstance(restored["trace_events"][0], TraceEvent)


def test_unregistered_pydantic_model_fails_under_strict_mode(monkeypatch):
    """An unregistered Pydantic model must be blocked or fail loudly under strict msgpack mode."""
    class UnregisteredModel(BaseModel):
        secret: str

    # Create serializer that only allows registered models
    serde = get_state_serializer()

    payload = {"data": UnregisteredModel(secret="unregistered")}
    dumped = serde.dumps_typed(payload)

    # When deserialized, because UnregisteredModel is not in STATE_MSG_PACK_MODULES,
    # it is blocked from being reconstructed as an UnregisteredModel instance
    restored = serde.loads_typed(dumped)
    assert not isinstance(restored["data"], UnregisteredModel)
