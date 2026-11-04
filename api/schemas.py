"""Pydantic schemas for the Research Agent Crew FastAPI service."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """Payload to initiate a multi-agent research run (legacy synchronous or background)."""
    task: str = Field(..., description="The research question or topic to investigate", min_length=5)
    max_retries: int = Field(default=1, ge=0, le=5, description="Maximum critic reflection loops")
    max_search_budget: int = Field(default=10, ge=1, le=30, description="Hard cap on MCP search calls")
    model_name: Optional[str] = Field(default=None, description="LLM model to use (Google Gemini)")


class SubtaskSummarySchema(BaseModel):
    id: str
    title: str
    sources_count: int


class CriticSummarySchema(BaseModel):
    verdict: str
    total_score: int
    coverage: int
    evidence: int
    coherence: int
    feedback: str


class TraceEventSummarySchema(BaseModel):
    timestamp: str
    step: str
    event_type: str
    iteration: int
    summary: str


class ResearchResponse(BaseModel):
    """Full execution response for a research query."""
    task_id: str
    task: str
    status: str
    iterations: int
    searches_performed: int
    cache_hits: Optional[int] = 0
    human_escalation_needed: bool
    escalation_reason: Optional[str] = None
    mcp_saved_path: Optional[str] = None
    report_markdown: Optional[str] = None
    report_metadata: Optional[Dict[str, Any]] = None
    structured_report: Optional[Dict[str, Any]] = None
    critic_summary: Optional[CriticSummarySchema] = None
    subtasks: List[SubtaskSummarySchema] = []
    traces: List[TraceEventSummarySchema] = []


# ==============================================================================
# Persistent Thread & Resumable Run Schemas
# ==============================================================================

class ThreadCreateRequest(BaseModel):
    """Payload to create a new research conversation thread."""
    title: Optional[str] = Field(default=None, description="Optional custom title; auto-derived from prompt if omitted")
    prompt: Optional[str] = Field(default=None, description="Optional initial prompt to auto-derive title from")
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom thread settings/parameters")


class ThreadUpdateRequest(BaseModel):
    """Payload to rename, archive, or adjust settings of a thread."""
    title: Optional[str] = None
    archived: Optional[bool] = None
    settings: Optional[Dict[str, Any]] = None


class ThreadResponse(BaseModel):
    """Representation of a persistent research thread."""
    id: str
    title: str
    created_at: str
    updated_at: str
    archived: bool
    settings: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None


class MessageResponse(BaseModel):
    """Representation of a message in a research thread."""
    id: str
    thread_id: str
    role: str
    content: str
    created_at: str
    run_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RunCreateRequest(BaseModel):
    """Payload to launch a research run within an existing thread."""
    task: str = Field(..., description="The research prompt or question", min_length=3)
    max_retries: int = Field(default=1, ge=0, le=5)
    max_search_budget: int = Field(default=10, ge=1, le=30)
    model_name: Optional[str] = None


class ResumeRunRequest(BaseModel):
    """Payload to resume an interrupted or quota-paused run."""
    force: bool = Field(default=False, description="Force resume even if before quota reset time")


class RunResponse(BaseModel):
    """Representation of an execution run."""
    id: str
    thread_id: str
    task: str
    status: str
    started_at: str
    finished_at: Optional[str] = None
    error: Optional[str] = None
    resume_after: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    report_md: Optional[str] = None
    report_json: Optional[Any] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    cancel_requested: bool = False


class RunEventResponse(BaseModel):
    """Representation of an individual run telemetry event."""
    seq: int
    run_id: str
    ts: str
    type: str
    node: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class ThreadExportResponse(BaseModel):
    """Complete bundle export of a thread including messages, runs, and events."""
    thread: ThreadResponse
    messages: List[MessageResponse]
    runs: List[RunResponse]
    events: List[RunEventResponse]
