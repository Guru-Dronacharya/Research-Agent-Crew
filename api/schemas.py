"""Pydantic schemas for the Research Agent Crew FastAPI service."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """Payload to initiate a multi-agent research run."""
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
