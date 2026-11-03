"""Unit tests for StructuredReport schema, validation, deterministic rendering, and critic metrics."""

import pytest
from core.state import (
    Claim, Section, ComparisonRow, StructuredReport, SubtaskResult, SourceItem,
    ResearchPlan, Subtask, AgentState
)
from core.report_render import (
    validate_structured_report, render_structured_report_to_markdown
)
from core.critic_checks import (
    compute_citation_validity, compute_structured_claim_metrics, run_deterministic_critic_checks
)
from core.nodes.synthesizer import synthesizer_node
from core.nodes.finalizer import finalizer_node


def test_invalid_urls_are_stripped():
    """Verify that URLs not present in retrieved sources are stripped from claim citations."""
    retrieved_sources = [
        SourceItem(title="Official Docs", url="https://langchain.com/docs", snippet="LangChain docs")
    ]

    structured = StructuredReport(
        title="Test Report",
        tldr=["Takeaway 1"],
        sections=[
            Section(
                subtask_id="s1",
                heading="Architecture",
                summary="Summary text",
                claims=[
                    Claim(
                        text="Valid and invalid cited claim",
                        source_urls=["https://langchain.com/docs", "https://unverified-domain.com/bogus"],
                        confidence=0.9
                    )
                ]
            )
        ]
    )

    validated = validate_structured_report(structured, retrieved_sources)

    # Valid URL retained, ungrounded URL stripped
    assert len(validated.sections) == 1
    assert len(validated.sections[0].claims) == 1
    assert validated.sections[0].claims[0].source_urls == ["https://langchain.com/docs"]


def test_claims_without_sources_are_dropped_and_moved_to_open_gaps():
    """Claims whose only citations are invalid must be dropped, and their topic moved to open_gaps."""
    retrieved_sources = [
        SourceItem(title="Official Docs", url="https://langchain.com/docs", snippet="LangChain docs")
    ]

    structured = StructuredReport(
        title="Test Report",
        tldr=["Takeaway 1"],
        sections=[
            Section(
                subtask_id="s1",
                heading="Scalability",
                summary="Scalability analysis summary",
                claims=[
                    Claim(
                        text="Completely unsupported claim",
                        source_urls=["https://unverified-hallucination.com/fake"],
                        confidence=0.5
                    ),
                    Claim(
                        text="Well supported claim",
                        source_urls=["https://langchain.com/docs"],
                        confidence=0.95
                    )
                ]
            )
        ]
    )

    validated = validate_structured_report(structured, retrieved_sources)

    # The unsupported claim was dropped
    assert len(validated.sections[0].claims) == 1
    assert validated.sections[0].claims[0].text == "Well supported claim"

    # Dropped claim was transferred to open gaps
    assert any("Completely unsupported claim" in gap for gap in validated.open_gaps)


def test_comparison_table_renders_only_when_present():
    """Comparison table markdown is rendered only when comparison rows exist, and omitted when None."""
    retrieved = [SourceItem(title="Docs", url="https://example.com/item", snippet="Snippet")]

    # Case A: Without comparison
    rep_without_comp = StructuredReport(
        title="Single Entity Evaluation",
        tldr=["Takeaway"],
        sections=[
            Section(
                subtask_id="s1",
                heading="Overview",
                summary="Overview summary",
                claims=[Claim(text="Claim A", source_urls=["https://example.com/item"], confidence=0.9)]
            )
        ],
        comparison=None
    )
    md_no_comp = render_structured_report_to_markdown(rep_without_comp, retrieved)
    assert "Comparative Analysis" not in md_no_comp
    assert "| Dimension |" not in md_no_comp

    # Case B: With comparison
    rep_with_comp = StructuredReport(
        title="Comparative Evaluation",
        tldr=["Takeaway"],
        sections=[
            Section(
                subtask_id="s1",
                heading="Overview",
                summary="Overview summary",
                claims=[Claim(text="Claim A", source_urls=["https://example.com/item"], confidence=0.9)]
            )
        ],
        comparison=[
            ComparisonRow(dimension="Latency", values={"FrameworkA": "10ms", "FrameworkB": "25ms"}),
            ComparisonRow(dimension="Memory", values={"FrameworkA": "50MB", "FrameworkB": "120MB"})
        ]
    )
    md_with_comp = render_structured_report_to_markdown(rep_with_comp, retrieved)
    assert "## Comparative Analysis" in md_with_comp
    assert "| Dimension | FrameworkA | FrameworkB |" in md_with_comp
    assert "| Latency | 10ms | 25ms |" in md_with_comp
    assert "| Memory | 50MB | 120MB |" in md_with_comp


def test_rendering_is_deterministic():
    """Calling render_structured_report_to_markdown multiple times produces byte-identical Markdown."""
    retrieved = [
        SourceItem(title="Source 1", url="https://example.com/1", snippet="Snippet 1"),
        SourceItem(title="Source 2", url="https://example.com/2", snippet="Snippet 2")
    ]

    report = StructuredReport(
        title="Deterministic Benchmark Report",
        tldr=["Takeaway 1", "Takeaway 2"],
        sections=[
            Section(
                subtask_id="s1",
                heading="Architecture Core",
                summary="Core findings narrative",
                claims=[
                    Claim(text="Claim Alpha", source_urls=["https://example.com/1"], confidence=0.95),
                    Claim(text="Claim Beta", source_urls=["https://example.com/2", "https://example.com/1"], confidence=0.85)
                ]
            )
        ],
        comparison=[
            ComparisonRow(dimension="Throughput", values={"OptionX": "High", "OptionY": "Medium"}),
            ComparisonRow(dimension="Cost", values={"OptionX": "$10", "OptionY": "$5"})
        ],
        recommendation="Adopt OptionX for throughput-critical workloads.",
        open_gaps=["Long-term longevity under load"],
        limitations=["Evaluated using simulated benchmark numbers"]
    )

    outputs = [render_structured_report_to_markdown(report, retrieved) for _ in range(10)]
    for out in outputs[1:]:
        assert out == outputs[0]


def test_rendered_report_passes_citation_validity():
    """A report validated against retrieved sources produces Markdown with 1.0 citation validity."""
    retrieved = [
        SourceItem(title="Python Asyncio", url="https://docs.python.org/3/library/asyncio.html", snippet="Async docs"),
        SourceItem(title="Tenacity Retries", url="https://tenacity.readthedocs.io/en/latest/", snippet="Retry docs")
    ]

    raw_report = StructuredReport(
        title="Async Reliability Analysis",
        tldr=["Async event loops require robust retry backoff."],
        sections=[
            Section(
                subtask_id="s1",
                heading="Async Engine",
                summary="Event loop details.",
                claims=[
                    Claim(
                        text="Asyncio handles non-blocking I/O efficiently.",
                        source_urls=["https://docs.python.org/3/library/asyncio.html"],
                        confidence=0.98
                    ),
                    Claim(
                        text="Tenacity handles exponential backoff.",
                        source_urls=["https://tenacity.readthedocs.io/en/latest/", "https://unverified-bogus-domain.com"],
                        confidence=0.90
                    )
                ]
            )
        ]
    )

    subtask_results = [
        SubtaskResult(
            subtask_id="s1",
            title="Async Engine",
            findings="Asyncio and Tenacity findings",
            sources=retrieved
        )
    ]

    validated = validate_structured_report(raw_report, retrieved)
    rendered_md = render_structured_report_to_markdown(validated, retrieved)

    validity_ratio, invalid_urls = compute_citation_validity(rendered_md, subtask_results)
    assert validity_ratio == 1.0
    assert invalid_urls == []


def test_structured_claim_metrics():
    """Deterministic claim metrics calculation computes claim counts and low confidence ratio correctly."""
    report = StructuredReport(
        title="Metrics Test",
        tldr=["Takeaway"],
        sections=[
            Section(
                subtask_id="s1",
                heading="Sec 1",
                summary="Sum",
                claims=[
                    Claim(text="High conf claim", source_urls=["https://ex.com/1"], confidence=0.9),
                    Claim(text="Low conf claim", source_urls=["https://ex.com/1"], confidence=0.5)
                ]
            ),
            Section(
                subtask_id="s2",
                heading="Sec 2",
                summary="Sum",
                claims=[
                    Claim(text="Mid conf claim", source_urls=["https://ex.com/2"], confidence=0.8)
                ]
            )
        ]
    )

    metrics = compute_structured_claim_metrics(report)
    assert metrics["claim_count"] == 3
    assert metrics["claims_per_subtask"] == {"s1": 2, "s2": 1}
    assert metrics["low_confidence_claims_count"] == 1
    assert metrics["low_confidence_ratio"] == round(1 / 3, 3)


@pytest.mark.asyncio
async def test_synthesizer_and_finalizer_preserve_structured_report():
    """Synthesizer stores draft_report_structured and finalizer saves structured JSON artifact."""
    sources = [SourceItem(title="Doc", url="https://langchain.com/lg", snippet="StateGraph")]
    subtasks = [
        SubtaskResult(
            subtask_id="s1",
            title="LangGraph Core",
            findings="Cyclic graph framework",
            sources=sources
        )
    ]

    initial_state: AgentState = {
        "task": "Evaluate LangGraph",
        "plan": ResearchPlan(
            task_summary="Analyze LangGraph",
            subtasks=[Subtask(id="s1", title="LangGraph Core", description="d", search_queries=[], rationale="r")],
            key_hypotheses=[]
        ),
        "subtask_results": subtasks,
        "draft_report": None,
        "draft_report_structured": None,
        "final_report": None,
        "final_report_structured": None,
        "report_metadata": None,
        "critic_evaluations": [],
        "retry_count": 0,
        "max_retries": 1,
        "search_budget_remaining": 5,
        "max_search_budget": 10,
        "initial_budget_remaining": 6,
        "revision_reserve_remaining": 4,
        "iteration_budget_remaining": 6,
        "total_searches_performed": 1,
        "cache_hits": 0,
        "trace_events": [],
        "human_escalation_needed": False,
        "escalation_reason": None,
        "status": "synthesizing",
        "mcp_saved_path": None,
        "model_name": "",
        "timings": {}
    }

    # Run synthesizer
    synth_result = await synthesizer_node(initial_state)
    assert synth_result["draft_report"] is not None
    assert synth_result["draft_report_structured"] is not None
    assert isinstance(synth_result["draft_report_structured"], StructuredReport)

    # Pass to finalizer
    initial_state.update(synth_result)
    final_result = await finalizer_node(initial_state)
    assert final_result["final_report_structured"] is not None
    assert final_result["status"] == "finished"
    assert any(e.event_type == "artifact_persistence" for e in final_result["trace_events"])


def test_run_deterministic_critic_checks_incorporates_structured_claims():
    """Verify that run_deterministic_critic_checks captures structured report claims."""
    state: AgentState = {
        "task": "Evaluate LangGraph",
        "plan": None,
        "draft_report": "Some report citing https://langchain.com",
        "draft_report_structured": StructuredReport(
            title="LangGraph Report",
            tldr=["Key takeaway"],
            sections=[
                Section(
                    subtask_id="s1",
                    heading="Core",
                    summary="Sum",
                    claims=[
                        Claim(text="Claim 1", source_urls=["https://langchain.com"], confidence=0.95),
                        Claim(text="Claim 2", source_urls=["https://langchain.com"], confidence=0.50)
                    ]
                )
            ]
        ),
        "subtask_results": [
            SubtaskResult(
                subtask_id="s1",
                title="Core",
                findings="Findings",
                sources=[SourceItem(title="LC", url="https://langchain.com", snippet="Docs")]
            )
        ],
        "total_searches_performed": 1,
        "max_search_budget": 5
    }
    metrics = run_deterministic_critic_checks(state)
    assert metrics["claim_count"] == 2
    assert metrics["claims_per_subtask"] == {"s1": 2}
    assert metrics["low_confidence_ratio"] == 0.5
    assert metrics["low_confidence_claims_count"] == 1

