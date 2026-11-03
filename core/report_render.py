"""Deterministic validation and Markdown rendering for StructuredReport.

Ensures citation grounding by stripping invalid URLs, dropping ungrounded claims,
and rendering clean, predictable Markdown without filler text.
"""

from typing import List, Optional, Dict
from core.state import StructuredReport, Section, Claim, SourceItem
from core.critic_checks import normalize_url


def validate_structured_report(
    structured: StructuredReport,
    retrieved_sources: List[SourceItem]
) -> StructuredReport:
    """Validate structured report against retrieved sources.

    - Strips URLs not present in retrieved sources.
    - Drops claims left with zero valid URLs.
    - Moves dropped claim topics into open_gaps.
    - Truncates TL;DR to at most 5 items.
    - Omits empty sections.
    """
    valid_url_set = {
        normalize_url(s.url)
        for s in retrieved_sources
        if s.url
    }

    open_gaps = list(structured.open_gaps)
    valid_sections: List[Section] = []

    for section in structured.sections:
        kept_claims: List[Claim] = []
        for claim in section.claims:
            grounded_urls = [
                u for u in claim.source_urls
                if normalize_url(u) in valid_url_set
            ]
            if grounded_urls:
                kept_claims.append(
                    Claim(
                        text=claim.text.strip(),
                        source_urls=grounded_urls,
                        confidence=claim.confidence
                    )
                )
            else:
                gap_note = f"Unverified claim dropped from '{section.heading}': {claim.text.strip()}"
                if gap_note not in open_gaps:
                    open_gaps.append(gap_note)

        summary_clean = section.summary.strip() if section.summary else ""
        if kept_claims or summary_clean:
            valid_sections.append(
                Section(
                    subtask_id=section.subtask_id,
                    heading=section.heading.strip(),
                    summary=summary_clean,
                    claims=kept_claims
                )
            )

    # Deduplicate open gaps while preserving order
    deduped_gaps = list(dict.fromkeys(open_gaps))

    return StructuredReport(
        title=structured.title.strip(),
        tldr=structured.tldr[:5],
        sections=valid_sections,
        comparison=structured.comparison,
        recommendation=structured.recommendation.strip() if structured.recommendation else None,
        open_gaps=deduped_gaps,
        limitations=list(dict.fromkeys(structured.limitations))
    )


def render_structured_report_to_markdown(
    report: StructuredReport,
    retrieved_sources: Optional[List[SourceItem]] = None
) -> str:
    """Render a StructuredReport deterministically to clean Markdown.

    Order:
    1. Title
    2. Executive Summary (TL;DR)
    3. Sections (with claims + numbered source links + confidence)
    4. Comparative Analysis table (if present)
    5. Recommendation (if present)
    6. Open Gaps (if present)
    7. Limitations (if present)
    8. Sources list
    """
    lines: List[str] = []

    # Map retrieved URLs to source titles for nice display
    url_to_title: Dict[str, str] = {}
    if retrieved_sources:
        for s in retrieved_sources:
            if s.url:
                norm = normalize_url(s.url)
                if norm not in url_to_title and s.title:
                    clean_title = s.title.replace("|", " ").replace("\n", " ").strip()
                    url_to_title[norm] = clean_title[:80]

    # 1. Title
    lines.append(f"# {report.title.strip()}")
    lines.append("")

    # 2. TL;DR / Executive Summary
    if report.tldr:
        lines.append("## Executive Summary")
        for item in report.tldr:
            if item.strip():
                lines.append(f"- {item.strip()}")
        lines.append("")

    # 3. Sections with claims, numbered links, and confidence
    url_to_num: Dict[str, int] = {}
    num_to_url: Dict[int, str] = {}

    for section in report.sections:
        heading = section.heading.strip()
        summary = section.summary.strip() if section.summary else ""
        if not heading and not summary and not section.claims:
            continue

        lines.append(f"## {heading}")
        if summary:
            lines.append(summary)
            lines.append("")

        for claim in section.claims:
            text = claim.text.strip()
            links_parts: List[str] = []
            for u in claim.source_urls:
                norm = normalize_url(u)
                if norm not in url_to_num:
                    idx = len(url_to_num) + 1
                    url_to_num[norm] = idx
                    num_to_url[idx] = u
                idx = url_to_num[norm]
                canonical_url = num_to_url[idx]
                links_parts.append(f"[{idx}]({canonical_url})")

            links_str = " ".join(links_parts)
            conf_str = f"(Confidence: {claim.confidence:.2f})"
            if links_str:
                lines.append(f"- {text} {links_str} {conf_str}")
            else:
                lines.append(f"- {text} {conf_str}")
        lines.append("")

    # 4. Comparative Analysis Table (optional)
    if report.comparison:
        all_entities = sorted({
            k for row in report.comparison for k in row.values.keys()
        })
        if all_entities:
            lines.append("## Comparative Analysis")
            header_cols = ["Dimension"] + all_entities
            lines.append("| " + " | ".join(header_cols) + " |")
            lines.append("| " + " | ".join([":---"] * len(header_cols)) + " |")
            for row in report.comparison:
                dim = row.dimension.replace("|", " ").strip()
                row_vals = [
                    str(row.values.get(ent, "-")).replace("|", " ").strip()
                    for ent in all_entities
                ]
                lines.append("| " + " | ".join([dim] + row_vals) + " |")
            lines.append("")

    # 5. Recommendation (optional)
    if report.recommendation and report.recommendation.strip():
        lines.append("## Recommendation")
        lines.append(report.recommendation.strip())
        lines.append("")

    # 6. Open Gaps (optional)
    if report.open_gaps:
        lines.append("## Open Gaps")
        for gap in report.open_gaps:
            if gap.strip():
                lines.append(f"- {gap.strip()}")
        lines.append("")

    # 7. Limitations (optional)
    if report.limitations:
        lines.append("## Limitations")
        for lim in report.limitations:
            if lim.strip():
                lines.append(f"- {lim.strip()}")
        lines.append("")

    # 8. Sources List
    if num_to_url:
        lines.append("## Verified Citations & References")
        for idx in sorted(num_to_url.keys()):
            raw_url = num_to_url[idx]
            norm = normalize_url(raw_url)
            title = url_to_title.get(norm, raw_url)
            lines.append(f"{idx}. [{title}]({raw_url})")
        lines.append("")

    # Clean up trailing empty lines
    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines)
