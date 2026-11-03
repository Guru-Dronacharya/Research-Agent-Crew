"""Unit tests for cheaper, smarter critic skip and parallel synthesis."""

import pytest
from core.state import (
    AgentState, ResearchPlan, Subtask, CriticEvaluation, CriticRubricScore, SubtaskResult, SourceItem
)
from core.nodes.critic import critic_node
from core.nodes.synthesizer import synthesizer_node
from core.critic_checks import compute_citation_validity
from core.llm import BaseAgentLLM, set_global_llm


class SpyJudgeLLM(BaseAgentLLM):
    def __init__(self):
        self.evaluate_called = False

    async def evaluate_report(self, *args, **kwargs) -> CriticEvaluation:
        self.evaluate_called = True
        return CriticEvaluation(
            iteration=0,
            coverage=CriticRubricScore(dimension="Coverage", score=5, rationale="Perfect coverage."),
            evidence=CriticRubricScore(dimension="Evidence", score=5, rationale="Verified claims."),
            coherence=CriticRubricScore(dimension="Coherence", score=5, rationale="Coherent structure."),
            total_score=15,
            verdict="PASS",
            overall_feedback="Excellent report.",
            next_step_instructions=[]
        )


@pytest.mark.asyncio
async def test_critic_skips_judge_on_citation_hallucination():
    """When citation_validity < 1.0, LLM judge is SKIPPED and evaluation marked judge_skipped=True."""
    spy_llm = SpyJudgeLLM()
    set_global_llm(spy_llm)

    state: AgentState = {
        "task": "Compare LangGraph vs AutoGen",
        "plan": ResearchPlan(
            task_summary="Compare agents",
            subtasks=[Subtask(id="s1", title="LangGraph", description="core", search_queries=[], rationale="r")],
            key_hypotheses=[]
        ),
        "draft_report": "Report citing [Hallucinated](https://hallucinated-citation.org/fake)",
        "subtask_results": [
            SubtaskResult(
                subtask_id="s1",
                title="LangGraph",
                findings="Details",
                sources=[SourceItem(title="Real", url="https://langchain.com/real", snippet="LangGraph architecture")]
            )
        ],
        "final_report": None,
        "report_metadata": None,
        "critic_evaluations": [],
        "retry_count": 0,
        "max_retries": 1,
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
        "model_name": ""
    }

    result = await critic_node(state)
    eval_res: CriticEvaluation = result["critic_evaluations"][0]

    assert not spy_llm.evaluate_called  # Judge was SKIPPED!
    assert eval_res.judge_skipped is True
    assert eval_res.verdict == "FAIL"
    assert eval_res.evidence.score <= 2


@pytest.mark.asyncio
async def test_critic_calls_judge_when_deterministic_checks_pass():
    """When all deterministic checks pass, LLM judge IS called."""
    spy_llm = SpyJudgeLLM()
    set_global_llm(spy_llm)

    state: AgentState = {
        "task": "Compare LangGraph vs AutoGen",
        "plan": ResearchPlan(
            task_summary="Compare agents",
            subtasks=[Subtask(id="s1", title="LangGraph", description="core", search_queries=[], rationale="r")],
            key_hypotheses=[]
        ),
        "draft_report": "Report citing [Real](https://langchain.com/real)",
        "subtask_results": [
            SubtaskResult(
                subtask_id="s1",
                title="LangGraph",
                findings="Details",
                sources=[SourceItem(title="Real", url="https://langchain.com/real", snippet="LangGraph vs AutoGen comparison")]
            )
        ],
        "final_report": None,
        "report_metadata": None,
        "critic_evaluations": [],
        "retry_count": 0,
        "max_retries": 1,
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
        "model_name": ""
    }

    result = await critic_node(state)
    eval_res: CriticEvaluation = result["critic_evaluations"][0]

    assert spy_llm.evaluate_called  # Judge WAS invoked!
    assert eval_res.judge_skipped is False
    assert eval_res.verdict == "PASS"


@pytest.mark.asyncio
async def test_synthesizer_parallel_generation_valid_citations():
    """Synthesizer produces a report whose citations all match retrieved sources (validity == 1.0)."""
    subtasks = [
        SubtaskResult(
            subtask_id="s1",
            title="LangGraph Architecture",
            findings="StateGraph provides cyclically connected agents.",
            sources=[SourceItem(title="LG Docs", url="https://langchain.com/docs/lg", snippet="StateGraph features")]
        ),
        SubtaskResult(
            subtask_id="s2",
            title="AutoGen Multi-Agent",
            findings="ConversableAgent models multi-agent conversations.",
            sources=[SourceItem(title="AG Docs", url="https://microsoft.com/autogen", snippet="ConversableAgent")]
        )
    ]

    state: AgentState = {
        "task": "Compare LangGraph vs AutoGen",
        "plan": ResearchPlan(
            task_summary="Compare architectures",
            subtasks=[
                Subtask(id="s1", title="LangGraph Architecture", description="d", search_queries=[], rationale="r"),
                Subtask(id="s2", title="AutoGen Multi-Agent", description="d", search_queries=[], rationale="r")
            ],
            key_hypotheses=[]
        ),
        "subtask_results": subtasks,
        "draft_report": None,
        "final_report": None,
        "report_metadata": None,
        "critic_evaluations": [],
        "retry_count": 0,
        "max_retries": 1,
        "search_budget_remaining": 5,
        "max_search_budget": 10,
        "initial_budget_remaining": 6,
        "revision_reserve_remaining": 4,
        "iteration_budget_remaining": 6,
        "total_searches_performed": 2,
        "trace_events": [],
        "human_escalation_needed": False,
        "escalation_reason": None,
        "status": "synthesizing",
        "mcp_saved_path": None,
        "model_name": ""
    }

    result = await synthesizer_node(state)
    draft = result["draft_report"]

    assert draft is not None
    assert "# " in draft
    assert "## Executive Summary" in draft
    validity, invalid = compute_citation_validity(draft, subtasks)
    assert validity == 1.0
    assert len(invalid) == 0
