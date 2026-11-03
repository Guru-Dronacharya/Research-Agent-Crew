"""State and data models for Research Agent Crew.

Defines the TypedDict AgentState used across the LangGraph orchestration,
as well as Pydantic models for structured planning, execution, rubric evaluations,
and trace logging.
"""

from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator


class Claim(BaseModel):
    """A granular factual claim grounded in retrieved source evidence."""
    text: str = Field(description="The factual claim statement or insight")
    source_urls: List[str] = Field(min_length=1, description="List of source URLs supporting this claim (minimum 1)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")

    @field_validator("confidence", mode="before")
    @classmethod
    def parse_confidence(cls, v: Any) -> float:
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))
        if isinstance(v, str):
            clean = v.strip().lower()
            if clean in ("high", "very high"):
                return 0.95
            if clean in ("medium", "moderate"):
                return 0.70
            if clean in ("low", "very low"):
                return 0.40
            try:
                val = float(clean)
                return max(0.0, min(1.0, val))
            except ValueError:
                return 1.0
        return 1.0


class Section(BaseModel):
    """Findings section for a single research subtask."""
    subtask_id: str = Field(description="Associated subtask ID, e.g. 'subtask_1'")
    heading: str = Field(description="Section heading or title")
    summary: str = Field(description="High-level narrative summary of findings for this section")
    claims: List[Claim] = Field(default_factory=list, description="Specific factual claims with citations")


class ComparisonRow(BaseModel):
    """A single row in a comparative analysis matrix."""
    dimension: str = Field(description="Feature, dimension, or metric being compared")
    values: Dict[str, Any] = Field(default_factory=dict, description="Mapping of entity name to evaluated attribute")


class StructuredReport(BaseModel):
    """Strictly structured research report schema."""
    title: str = Field(description="Clear, professional research report title")
    tldr: List[str] = Field(default_factory=list, max_length=5, description="Executive takeaways (maximum 5 bullet points)")
    sections: List[Section] = Field(default_factory=list, description="Subtask-specific finding sections")
    comparison: Optional[List[ComparisonRow]] = Field(default=None, description="Comparison matrix if comparing items")
    recommendation: Optional[str] = Field(default=None, description="Actionable recommendation if applicable")
    open_gaps: List[str] = Field(default_factory=list, description="Identified evidence gaps and unverified items")
    limitations: List[str] = Field(default_factory=list, description="Methodological limitations or caveats")


class Subtask(BaseModel):
    """A granular research subtask decomposed by the Planner."""
    id: str = Field(description="Unique identifier for the subtask, e.g., 'subtask_1'")
    title: str = Field(description="Clear title of the subtask")
    description: str = Field(description="Actionable description of the research goal")
    search_queries: List[str] = Field(
        default_factory=list,
        description="Targeted search queries to gather evidence for this subtask"
    )
    rationale: str = Field(description="Why this subtask is essential to answering the master question")


class ResearchPlan(BaseModel):
    """Structured research plan generated and revised by the Planner agent."""
    task_summary: str = Field(description="High-level understanding of the research objective")
    subtasks: List[Subtask] = Field(
        default_factory=list,
        description="Ordered list of subtasks to execute"
    )
    key_hypotheses: List[str] = Field(
        default_factory=list,
        description="Initial hypotheses or questions to validate"
    )
    estimated_searches: int = Field(
        default=5,
        description="Estimated number of tool calls needed"
    )
    revision_notes: Optional[str] = Field(
        default=None,
        description="Reflection notes explaining what was changed based on Critic feedback"
    )


class SourceItem(BaseModel):
    """A verified source discovered by the Executor via search MCP."""
    title: str = Field(description="Page or document title")
    url: str = Field(description="Canonical URL or reference location")
    snippet: str = Field(description="Relevant text excerpt extracted from the source")
    relevance: str = Field(default="relevant", description="Relevance assessment")


class SubtaskResult(BaseModel):
    """Execution output for a single research subtask."""
    subtask_id: str = Field(description="ID of the subtask executed")
    title: str = Field(description="Title of the subtask")
    findings: str = Field(description="Synthesized facts and insights gathered for this subtask")
    sources: List[SourceItem] = Field(
        default_factory=list,
        description="Citations and URLs discovered during execution"
    )
    tools_invoked: List[str] = Field(
        default_factory=list,
        description="Names of MCP tools called during this subtask"
    )
    evidence_gap: bool = Field(
        default=False,
        description="True if no usable evidence could be retrieved for this subtask"
    )


class CriticRubricScore(BaseModel):
    """Evaluation score and justification for a single rubric dimension."""
    dimension: str = Field(description="Dimension name: 'Coverage', 'Evidence', or 'Coherence'")
    score: int = Field(ge=0, le=5, description="Numerical score from 0 (poor) to 5 (excellent)")
    rationale: str = Field(description="Thorough explanation justifying the assigned score")


class CriticEvaluation(BaseModel):
    """Multidimensional evaluation performed by the Critic agent."""
    iteration: int = Field(description="Current loop iteration count")
    coverage: CriticRubricScore = Field(description="Assessment of whether all sub-questions are answered")
    evidence: CriticRubricScore = Field(description="Assessment of whether claims are backed by credible sources/links")
    coherence: CriticRubricScore = Field(description="Assessment of logical flow and report organization")
    total_score: int = Field(ge=0, le=15, description="Sum of dimension scores (out of 15)")
    verdict: str = Field(description="'PASS' or 'FAIL'")
    overall_feedback: str = Field(description="Executive critique summarizing strengths and weaknesses")
    next_step_instructions: List[str] = Field(
        default_factory=list,
        description="Concrete, actionable revision instructions for the Planner if verdict is FAIL"
    )
    judge_skipped: bool = Field(
        default=False,
        description="True if LLM judge call was skipped due to certain deterministic hard-fail"
    )


class TraceEvent(BaseModel):
    """Structured telemetry event for system observability and UI trace inspection."""
    timestamp: str = Field(description="ISO-8601 formatted timestamp of the event")
    step: str = Field(description="Current agent role or graph node (planner, executor, synthesizer, critic, finalizer)")
    event_type: str = Field(description="Event category (decomposition, tool_call, tool_result, rubric_evaluation, reflection_revision, budget_alert, escalation)")
    iteration: int = Field(default=0, description="Loop iteration counter")
    summary: str = Field(description="Short human-readable summary of the event")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full payload (prompts, tool parameters, scores, diffs, outputs) for drill-down inspection"
    )


class AgentState(TypedDict):
    """Complete LangGraph state shared across all nodes."""
    task: str
    plan: Optional[ResearchPlan]
    subtask_results: List[SubtaskResult]
    draft_report: Optional[str]
    draft_report_structured: Optional[StructuredReport]
    final_report: Optional[str]
    final_report_structured: Optional[StructuredReport]
    report_metadata: Optional[Dict[str, Any]]
    critic_evaluations: List[CriticEvaluation]
    retry_count: int
    max_retries: int
    search_budget_remaining: int
    max_search_budget: int
    initial_budget_remaining: int
    revision_reserve_remaining: int
    iteration_budget_remaining: int
    total_searches_performed: int
    cache_hits: int
    trace_events: List[TraceEvent]
    human_escalation_needed: bool
    escalation_reason: Optional[str]
    status: str
    mcp_saved_path: Optional[str]
    model_name: str
    timings: Optional[Dict[str, Any]]
