"""Unit tests for Critic Agent hybrid scoring, deterministic checks, hard caps, and routing."""

import pytest
from core.state import (
    AgentState, ResearchPlan, Subtask, CriticEvaluation, CriticRubricScore, SubtaskResult, SourceItem
)
from core.nodes.critic import critic_node
from core.graph import should_continue
from core.critic_checks import (
    compute_citation_validity,
    compute_source_relevance,
    compute_evidence_gap_rate,
    compute_min_sources,
    compute_coverage_map,
    compute_budget_respected
)
from tests.stubs import StubLLM
from core.llm import set_global_llm


# ---------------------------------------------------------------------------
# Deterministic Checks Unit Tests (Pure Functions)
# ---------------------------------------------------------------------------

@pytest.mark.critical
def test_citation_validity_all_valid():
    """Verify citation validity is 1.0 when all cited URLs exist in retrieved sources."""
    results = [
        SubtaskResult(
            subtask_id="s1",
            title="LangGraph Overview",
            findings="Found state graph docs",
            sources=[SourceItem(title="Docs", url="https://langchain.com/langgraph", snippet="StateGraph architecture")]
        )
    ]
    report = "Detailed review citing [LangGraph](https://langchain.com/langgraph)."
    validity, invalid = compute_citation_validity(report, results)
    assert validity == 1.0
    assert len(invalid) == 0


@pytest.mark.critical
def test_citation_validity_with_hallucinated_url():
    """Verify citation validity is < 1.0 and flags hallucinated URL not in retrieved sources."""
    results = [
        SubtaskResult(
            subtask_id="s1",
            title="LangGraph Overview",
            findings="Found state graph docs",
            sources=[SourceItem(title="Docs", url="https://langchain.com/langgraph", snippet="StateGraph")]
        )
    ]
    report = (
        "Citing real source [LangGraph](https://langchain.com/langgraph) "
        "and fake citation [Fake Source](https://hallucinated-citation-example.com/fake)."
    )
    validity, invalid = compute_citation_validity(report, results)
    assert validity == 0.5
    assert "https://hallucinated-citation-example.com/fake" in invalid


def test_source_relevance_keyword_matching():
    """Verify source relevance computes fraction of sources with >= 2 task keywords."""
    task = "Analyze LangGraph architecture and multi-agent state persistence"
    results = [
        SubtaskResult(
            subtask_id="s1",
            title="LangGraph Core",
            findings="Architecture overview",
            sources=[
                SourceItem(
                    title="LangGraph Architecture",
                    url="https://langchain.com/docs",
                    snippet="Deep dive into multi-agent StateGraph architecture and state persistence."
                ),
                SourceItem(
                    title="Cooking Recipes",
                    url="https://recipes.com/pasta",
                    snippet="Delicious pasta with garlic and tomato sauce."
                )
            ]
        )
    ]
    relevance = compute_source_relevance(task, results)
    assert relevance["total_sources"] == 2
    assert relevance["relevant_count"] == 1
    assert relevance["relevance_ratio"] == 0.5


def test_evidence_gap_rate_calculation():
    """Verify evidence gap rate correctly identifies subtasks flagged evidence_gap."""
    results = [
        SubtaskResult(
            subtask_id="s1",
            title="Subtask 1",
            findings="Grounded findings",
            sources=[SourceItem(title="S1", url="https://s1.org", snippet="data")],
            evidence_gap=False
        ),
        SubtaskResult(
            subtask_id="s2",
            title="Subtask 2",
            findings="No evidence retrieved",
            sources=[],
            evidence_gap=True
        ),
        SubtaskResult(
            subtask_id="s3",
            title="Subtask 3",
            findings="No evidence retrieved",
            sources=[],
            evidence_gap=True
        )
    ]
    rate, gap_ids = compute_evidence_gap_rate(results)
    assert rate == round(2 / 3, 4)
    assert "s2" in gap_ids
    assert "s3" in gap_ids


def test_min_sources_unique_count():
    """Verify min_sources counts distinct URLs across subtasks."""
    results = [
        SubtaskResult(
            subtask_id="s1",
            title="S1",
            findings="",
            sources=[
                SourceItem(title="A", url="https://example.com/a", snippet=""),
                SourceItem(title="A Duplicate", url="https://example.com/a/", snippet="")
            ]
        ),
        SubtaskResult(
            subtask_id="s2",
            title="S2",
            findings="",
            sources=[
                SourceItem(title="B", url="https://example.com/b", snippet="")
            ]
        )
    ]
    unique_count = compute_min_sources(results)
    assert unique_count == 2


def test_coverage_map_and_budget_respected():
    """Verify coverage map identifies addressed subtasks and budget respected check."""
    plan = ResearchPlan(
        task_summary="Test Plan",
        subtasks=[
            Subtask(id="s1", title="StateGraph Architecture", description="core", search_queries=[], rationale=""),
            Subtask(id="s2", title="Uncovered Topic", description="uncovered", search_queries=[], rationale="")
        ]
    )
    results = [
        SubtaskResult(
            subtask_id="s1",
            title="StateGraph Architecture",
            findings="Found architecture details",
            sources=[SourceItem(title="Docs", url="https://docs.langchain.com", snippet="")]
        ),
        SubtaskResult(
            subtask_id="s2",
            title="Uncovered Topic",
            findings="No data",
            sources=[],
            evidence_gap=True
        )
    ]
    report = "# Report\nDiscussing StateGraph Architecture with [Docs](https://docs.langchain.com)."
    coverage = compute_coverage_map(plan, report, results)
    assert coverage["s1"] is True
    assert coverage["s2"] is False

    assert compute_budget_respected(5, 10) is True
    assert compute_budget_respected(11, 10) is False


# ---------------------------------------------------------------------------
# Hybrid Critic Node & Hard Scoring Caps Tests
# ---------------------------------------------------------------------------

class FakeCapLLM(StubLLM):
    """Custom double that returns perfect 5/5/5 scores to test hard caps."""
    async def evaluate_report(self, *args, **kwargs) -> CriticEvaluation:
        return CriticEvaluation(
            iteration=0,
            coverage=CriticRubricScore(dimension="Coverage", score=5, rationale="Superb coverage"),
            evidence=CriticRubricScore(dimension="Evidence", score=5, rationale="Superb evidence"),
            coherence=CriticRubricScore(dimension="Coherence", score=5, rationale="Superb coherence"),
            total_score=15,
            verdict="PASS",
            overall_feedback="Excellent",
            next_step_instructions=[]
        )


@pytest.mark.asyncio
async def test_critic_hard_cap_citation_validity(monkeypatch):
    """Verify that citation_validity < 1.0 caps Evidence <= 2 and changes verdict to FAIL."""
    set_global_llm(FakeCapLLM())

    state: AgentState = {
        "task": "Analyze LangGraph architecture",
        "plan": ResearchPlan(
            task_summary="Analyze LangGraph",
            subtasks=[Subtask(id="s1", title="StateGraph", description="core", search_queries=[], rationale="core")],
            key_hypotheses=[]
        ),
        # Report cites a hallucinated URL not in retrieved sources
        "draft_report": "# LangGraph Report\nCiting [Fake](https://hallucinated-citation.org).",
        "subtask_results": [
            SubtaskResult(
                subtask_id="s1",
                title="StateGraph",
                findings="Details",
                sources=[SourceItem(title="Real", url="https://langchain.com/real", snippet="architecture LangGraph")]
            )
        ],
        "final_report": None,
        "report_metadata": None,
        "critic_evaluations": [],
        "retry_count": 0,
        "max_retries": 2,
        "search_budget_remaining": 5,
        "max_search_budget": 10,
        "initial_budget_remaining": 6,
        "revision_reserve_remaining": 4,
        "iteration_budget_remaining": 6,
        "total_searches_performed": 2,
        "trace_events": [],
        "human_escalation_needed": False,
        "escalation_reason": None,
        "status": "evaluating",
        "mcp_saved_path": None,
        "model_name": "gemini-2.0-flash"
    }

    result = await critic_node(state)
    eval_res: CriticEvaluation = result["critic_evaluations"][0]

    # Hard cap must cap Evidence at <= 2
    assert eval_res.evidence.score <= 2
    assert "Hard Cap Applied" in eval_res.evidence.rationale
    assert eval_res.verdict == "FAIL"
    assert result["status"] == "revising"


@pytest.mark.asyncio
async def test_critic_hard_cap_evidence_gap_rate():
    """Verify that evidence_gap_rate > 0.34 caps Coverage <= 2."""
    set_global_llm(FakeCapLLM())

    state: AgentState = {
        "task": "Analyze LangGraph architecture",
        "plan": ResearchPlan(
            task_summary="Analyze LangGraph",
            subtasks=[
                Subtask(id="s1", title="Subtask 1", description="", search_queries=[], rationale=""),
                Subtask(id="s2", title="Subtask 2", description="", search_queries=[], rationale="")
            ],
            key_hypotheses=[]
        ),
        "draft_report": "# Report\nCiting [Real](https://langchain.com/real).",
        "subtask_results": [
            # 1 of 2 subtasks has evidence_gap=True -> gap_rate = 0.5 > 0.34
            SubtaskResult(
                subtask_id="s1",
                title="Subtask 1",
                findings="Ok",
                sources=[SourceItem(title="Real", url="https://langchain.com/real", snippet="LangGraph architecture")],
                evidence_gap=False
            ),
            SubtaskResult(
                subtask_id="s2",
                title="Subtask 2",
                findings="Missing",
                sources=[],
                evidence_gap=True
            )
        ],
        "final_report": None,
        "report_metadata": None,
        "critic_evaluations": [],
        "retry_count": 0,
        "max_retries": 2,
        "search_budget_remaining": 5,
        "max_search_budget": 10,
        "initial_budget_remaining": 6,
        "revision_reserve_remaining": 4,
        "iteration_budget_remaining": 6,
        "total_searches_performed": 2,
        "trace_events": [],
        "human_escalation_needed": False,
        "escalation_reason": None,
        "status": "evaluating",
        "mcp_saved_path": None,
        "model_name": "gemini-2.0-flash"
    }

    result = await critic_node(state)
    eval_res: CriticEvaluation = result["critic_evaluations"][0]

    # Coverage must be capped at <= 2
    assert eval_res.coverage.score <= 2
    assert "Hard Cap Applied" in eval_res.coverage.rationale
    assert eval_res.verdict == "FAIL"


@pytest.mark.asyncio
async def test_critic_evaluation_rubric_pass():
    """Verify that a passing evaluation with valid citations routes to finalizer."""
    # StubLLM returns 5/5/5 on iteration >= 1
    state: AgentState = {
        "task": "Analyze LangGraph architecture",
        "plan": ResearchPlan(
            task_summary="Analyze LangGraph",
            subtasks=[Subtask(id="s1", title="StateGraph", description="core", search_queries=[], rationale="core")],
            key_hypotheses=[]
        ),
        "draft_report": "# LangGraph Report\nThorough coverage with [LangGraph](https://langchain.com/docs).",
        "subtask_results": [
            SubtaskResult(
                subtask_id="s1",
                title="StateGraph",
                findings="Architecture details",
                sources=[SourceItem(title="Docs", url="https://langchain.com/docs", snippet="LangGraph architecture")]
            )
        ],
        "final_report": None,
        "report_metadata": None,
        "critic_evaluations": [],
        "retry_count": 1,
        "max_retries": 2,
        "search_budget_remaining": 5,
        "max_search_budget": 10,
        "initial_budget_remaining": 6,
        "revision_reserve_remaining": 4,
        "iteration_budget_remaining": 4,
        "total_searches_performed": 2,
        "trace_events": [],
        "human_escalation_needed": False,
        "escalation_reason": None,
        "status": "evaluating",
        "mcp_saved_path": None,
        "model_name": "gemini-2.0-flash"
    }

    result = await critic_node(state)
    assert len(result["critic_evaluations"]) == 1
    eval_res: CriticEvaluation = result["critic_evaluations"][0]
    assert eval_res.verdict == "PASS"
    assert eval_res.total_score >= 12
    assert result["status"] == "completed"

    next_node = should_continue(result)
    assert next_node == "finalizer"


@pytest.mark.asyncio
async def test_critic_evaluation_rubric_fail_loops_to_planner():
    """Verify that a failing evaluation within retry limits loops back to planner with concrete next steps."""
    state: AgentState = {
        "task": "Analyze LangGraph architecture",
        "plan": ResearchPlan(
            task_summary="Analyze LangGraph",
            subtasks=[Subtask(id="s1", title="StateGraph", description="core", search_queries=[], rationale="core")],
            key_hypotheses=[]
        ),
        "draft_report": "# Incomplete Draft",
        "subtask_results": [
            SubtaskResult(
                subtask_id="s1",
                title="StateGraph",
                findings="No evidence",
                sources=[],
                evidence_gap=True
            )
        ],
        "final_report": None,
        "report_metadata": None,
        "critic_evaluations": [],
        "retry_count": 0,
        "max_retries": 2,
        "search_budget_remaining": 5,
        "max_search_budget": 10,
        "initial_budget_remaining": 6,
        "revision_reserve_remaining": 4,
        "iteration_budget_remaining": 6,
        "total_searches_performed": 2,
        "trace_events": [],
        "human_escalation_needed": False,
        "escalation_reason": None,
        "status": "evaluating",
        "mcp_saved_path": None,
        "model_name": "gemini-2.0-flash"
    }

    result = await critic_node(state)
    assert len(result["critic_evaluations"]) == 1
    eval_res: CriticEvaluation = result["critic_evaluations"][0]
    assert eval_res.verdict == "FAIL"
    assert len(eval_res.next_step_instructions) > 0
    # Must reference specific failing subtask
    assert any("s1" in step for step in eval_res.next_step_instructions)
    assert result["status"] == "revising"
    assert result["retry_count"] == 1

    next_node = should_continue(result)
    assert next_node == "planner"


@pytest.mark.asyncio
async def test_critic_escalation_on_max_retries():
    """Verify that exceeding max retries triggers human escalation rather than infinite loop."""
    state: AgentState = {
        "task": "Analyze LangGraph architecture",
        "plan": ResearchPlan(
            task_summary="Analyze LangGraph",
            subtasks=[Subtask(id="s1", title="StateGraph", description="core", search_queries=[], rationale="core")],
            key_hypotheses=[]
        ),
        "draft_report": "# Incomplete Draft",
        "subtask_results": [],
        "final_report": None,
        "report_metadata": None,
        "critic_evaluations": [],
        "retry_count": 0,
        "max_retries": 0,  # Zero retries allowed
        "search_budget_remaining": 5,
        "max_search_budget": 10,
        "initial_budget_remaining": 6,
        "revision_reserve_remaining": 4,
        "iteration_budget_remaining": 6,
        "total_searches_performed": 2,
        "trace_events": [],
        "human_escalation_needed": False,
        "escalation_reason": None,
        "status": "evaluating",
        "mcp_saved_path": None,
        "model_name": "gemini-2.0-flash"
    }

    result = await critic_node(state)
    assert result["human_escalation_needed"] is True
    assert "Evaluation threshold unmet" in result["escalation_reason"] or "retries" in result["escalation_reason"]
    assert result["status"] == "escalated"

    next_node = should_continue(result)
    assert next_node == "finalizer"


@pytest.mark.asyncio
async def test_critic_escalation_on_zero_reserve():
    """Verify that when a Critic FAIL occurs with zero reserve remaining, it escalates with budget reason."""
    state: AgentState = {
        "task": "Analyze LangGraph architecture",
        "plan": ResearchPlan(
            task_summary="Analyze LangGraph",
            subtasks=[Subtask(id="s1", title="StateGraph", description="core", search_queries=[], rationale="core")],
            key_hypotheses=[]
        ),
        "draft_report": "# Incomplete Draft",
        "subtask_results": [],
        "final_report": None,
        "report_metadata": None,
        "critic_evaluations": [],
        "retry_count": 0,
        "max_retries": 2,  # Retries theoretically allowed, but budget reserve is 0
        "search_budget_remaining": 0,
        "max_search_budget": 10,
        "initial_budget_remaining": 0,
        "revision_reserve_remaining": 0,  # Zero reserve left!
        "iteration_budget_remaining": 0,
        "total_searches_performed": 10,
        "trace_events": [],
        "human_escalation_needed": False,
        "escalation_reason": None,
        "status": "evaluating",
        "mcp_saved_path": None,
        "model_name": "gemini-2.0-flash"
    }

    result = await critic_node(state)
    assert result["human_escalation_needed"] is True
    assert "budget" in result["escalation_reason"].lower() or "exhausted" in result["escalation_reason"].lower()
    assert result["status"] == "escalated"

    next_node = should_continue(result)
    assert next_node == "finalizer"
