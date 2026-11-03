"""Deterministic Quality and Grounding Checks for the Critic Agent.

Provides pure, fully unit-testable verification functions that evaluate:
- Citation validity (every report URL must exist in retrieved sources)
- Source relevance (keywords extracted from task matched in snippets)
- Evidence gap rate (fraction of subtasks with evidence gaps)
- Minimum source volume & near-duplicate detection
- Subtask coverage map (identifying which subtasks are addressed)
- Budget constraint verification
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from urllib.parse import urlparse
from core.state import AgentState, SubtaskResult, ResearchPlan, SourceItem, StructuredReport


STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't",
    "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers",
    "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they",
    "they'd", "they'll", "they're", "they've", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
    "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves", "vs", "versus", "compare", "analysis", "investigate",
    "research", "report", "detailed", "deep", "dive"
}


def normalize_url(url: str) -> str:
    """Normalize URL by stripping trailing slashes, fragments, and standardizing scheme/netloc."""
    if not url:
        return ""
    clean = url.strip()
    # Remove trailing punctuation often captured by regex
    clean = clean.rstrip(").,;:\"'")
    try:
        parsed = urlparse(clean)
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/")
        normalized = f"{parsed.scheme.lower()}://{netloc}{path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        return normalized
    except Exception:
        return clean.lower()


def extract_report_urls(report: str) -> List[str]:
    """Extract all HTTP/HTTPS URLs present in the report markdown."""
    if not report:
        return []
    # Markdown links [title](url)
    md_urls = re.findall(r"\[.*?\]\((https?://[^\s\)]+)\)", report)
    # Bare URLs (exclude markdown brackets and common delimiters)
    bare_urls = re.findall(r"(?<!\()(https?://[^\s\)\"\'><\]\[]+)", report)
    combined = []
    seen = set()
    for u in md_urls + bare_urls:
        norm = normalize_url(u)
        if norm and norm not in seen:
            seen.add(norm)
            combined.append(norm)
    return combined


def extract_task_keywords(task: str) -> List[str]:
    """Extract substantive keywords from the task statement, filtering stop words."""
    cleaned = re.sub(r"[^\w\s-]", " ", task.lower())
    tokens = [w for w in cleaned.split() if len(w) > 2]
    keywords = [w for w in tokens if w not in STOP_WORDS]
    return list(dict.fromkeys(keywords))  # Preserves order, removes duplicates


def compute_citation_validity(report: str, subtask_results: List[SubtaskResult]) -> Tuple[float, List[str]]:
    """Compute the fraction of URLs cited in the draft report that exist in retrieved sources.

    Returns:
        (validity_ratio, list_of_invalid_urls)
    """
    report_urls = extract_report_urls(report)
    retrieved_urls = {
        normalize_url(s.url)
        for r in subtask_results
        for s in r.sources
        if s.url
    }

    if not report_urls:
        # If the report cites no URLs:
        # If no sources were retrieved anyway, validity is 1.0 (no false citations)
        # If sources WERE retrieved but report cited none of them, validity is 0.0
        return (1.0 if not retrieved_urls else 0.0), []

    invalid_urls = [u for u in report_urls if u not in retrieved_urls]
    valid_count = len(report_urls) - len(invalid_urls)
    validity = valid_count / len(report_urls)
    return round(validity, 4), invalid_urls


def compute_source_relevance(
    task: str,
    subtask_results: List[SubtaskResult]
) -> Dict[str, Any]:
    """Assess whether retrieved sources are relevant to the master task keywords.

    A source is considered relevant if its combined title and snippet contain at least
    2 keywords derived from the task statement. Also flags duplicate and near-duplicate URLs.
    """
    keywords = extract_task_keywords(task)
    all_sources: List[SourceItem] = [
        s for r in subtask_results for s in r.sources if s.url
    ]

    if not all_sources:
        return {
            "relevance_ratio": 0.0,
            "relevant_count": 0,
            "total_sources": 0,
            "near_duplicates": [],
            "keywords_checked": keywords
        }

    relevant_count = 0
    seen_domains: Dict[str, List[str]] = {}
    near_duplicates: List[str] = []

    for src in all_sources:
        text = f"{src.title} {src.snippet}".lower()
        matched_kw = [kw for kw in keywords if kw in text]
        if len(keywords) < 2:
            # If the task itself only has 0 or 1 keyword, matching 1 keyword is sufficient
            is_relevant = len(matched_kw) >= max(1, len(keywords))
        else:
            is_relevant = len(matched_kw) >= 2

        if is_relevant:
            relevant_count += 1

        norm = normalize_url(src.url)
        try:
            parsed = urlparse(norm)
            path_key = f"{parsed.netloc}{parsed.path}"
            if path_key in seen_domains:
                near_duplicates.append(src.url)
            else:
                seen_domains[path_key] = [src.url]
        except Exception:
            pass

    ratio = relevant_count / len(all_sources)
    return {
        "relevance_ratio": round(ratio, 4),
        "relevant_count": relevant_count,
        "total_sources": len(all_sources),
        "near_duplicates": list(set(near_duplicates)),
        "keywords_checked": keywords
    }


def compute_evidence_gap_rate(subtask_results: List[SubtaskResult]) -> Tuple[float, List[str]]:
    """Compute the fraction of subtasks that failed to retrieve usable evidence.

    Returns:
        (gap_rate, list_of_gap_subtask_ids)
    """
    if not subtask_results:
        return 1.0, []

    gap_ids = [
        r.subtask_id
        for r in subtask_results
        if getattr(r, "evidence_gap", False) or not r.sources
    ]
    rate = len(gap_ids) / len(subtask_results)
    return round(rate, 4), gap_ids


def compute_min_sources(subtask_results: List[SubtaskResult]) -> int:
    """Return the number of unique relevant URLs retrieved across all subtasks."""
    unique_urls = {
        normalize_url(s.url)
        for r in subtask_results
        for s in r.sources
        if s.url
    }
    return len(unique_urls)


def compute_coverage_map(
    plan: ResearchPlan,
    report: str,
    subtask_results: List[SubtaskResult]
) -> Dict[str, bool]:
    """Map each planned subtask to whether the draft report incorporates evidence for it.

    A subtask is covered if:
    1. A source retrieved for that subtask is referenced in the report, OR
    2. Key subtask title phrases appear in the report alongside discussion.
    """
    coverage: Dict[str, bool] = {}
    report_lower = report.lower() if report else ""

    subtask_res_map = {r.subtask_id: r for r in subtask_results}

    for subtask in plan.subtasks:
        res = subtask_res_map.get(subtask.id)
        is_covered = False

        if res and res.sources:
            for s in res.sources:
                norm = normalize_url(s.url)
                if norm and (norm in report_lower or s.title.lower() in report_lower):
                    is_covered = True
                    break

        if not is_covered and subtask.title.lower() in report_lower:
            is_covered = True

        coverage[subtask.id] = is_covered

    return coverage


def compute_budget_respected(total_searches: int, max_budget: int) -> bool:
    """Verify that total searches performed is strictly within the allocated budget."""
    return total_searches <= max_budget


def compute_structured_claim_metrics(structured: Optional[StructuredReport]) -> Dict[str, Any]:
    """Compute deterministic metrics across structured claims:
    - total claim count
    - claims per subtask (dict)
    - low-confidence ratio (claims with confidence < 0.70)
    """
    if not structured or not structured.sections:
        return {
            "claim_count": 0,
            "claims_per_subtask": {},
            "low_confidence_claims_count": 0,
            "low_confidence_ratio": 0.0
        }

    total_claims = 0
    claims_per_subtask: Dict[str, int] = {}
    low_confidence_count = 0

    for sec in structured.sections:
        c_count = len(sec.claims)
        claims_per_subtask[sec.subtask_id] = c_count
        total_claims += c_count
        for c in sec.claims:
            if c.confidence < 0.70:
                low_confidence_count += 1

    low_conf_ratio = (low_confidence_count / total_claims) if total_claims > 0 else 0.0

    return {
        "claim_count": total_claims,
        "claims_per_subtask": claims_per_subtask,
        "low_confidence_claims_count": low_confidence_count,
        "low_confidence_ratio": round(low_conf_ratio, 3)
    }


def run_deterministic_critic_checks(state: AgentState) -> Dict[str, Any]:
    """Run the complete suite of deterministic quality checks against state.

    Returns a structured dictionary of metrics, caps indicators, and deterministic revision suggestions.
    """
    task = state.get("task", "")
    plan = state.get("plan")
    draft = state.get("draft_report", "") or ""
    subtask_results = state.get("subtask_results", [])
    total_searches = state.get("total_searches_performed", 0)
    max_budget = state.get("max_search_budget", 10)
    structured_report = state.get("draft_report_structured")

    # 1. Citation Validity
    citation_validity, invalid_urls = compute_citation_validity(draft, subtask_results)

    # 2. Source Relevance
    relevance_data = compute_source_relevance(task, subtask_results)
    source_relevance = relevance_data["relevance_ratio"]

    # 3. Evidence Gap Rate
    evidence_gap_rate, gap_subtask_ids = compute_evidence_gap_rate(subtask_results)

    # 4. Min Sources
    min_sources = compute_min_sources(subtask_results)

    # 5. Coverage Map
    coverage_map = {}
    if plan and plan.subtasks:
        coverage_map = compute_coverage_map(plan, draft, subtask_results)

    # 6. Budget Respected
    budget_respected = compute_budget_respected(total_searches, max_budget)

    # 7. Structured Claims Metrics
    claim_metrics = compute_structured_claim_metrics(structured_report)

    # Generate deterministic revision instructions for failing checks
    instructions = []
    if citation_validity < 1.0:
        if invalid_urls:
            instructions.append(
                f"Remove unverified or hallucinated citations not found in retrieved evidence: {', '.join(invalid_urls[:3])}"
            )
        else:
            instructions.append("Substantiate claims by citing verified URLs discovered in retrieved evidence.")

    if gap_subtask_ids:
        for sid in gap_subtask_ids[:2]:
            instructions.append(f"Subtask '{sid}' returned 0 usable sources; execute targeted query to retrieve primary evidence.")

    if source_relevance < 0.6:
        instructions.append(
            f"Improve source relevance: align search queries more closely with master task keywords ({', '.join(relevance_data['keywords_checked'][:4])})."
        )

    if structured_report and claim_metrics["low_confidence_ratio"] > 0.5:
        instructions.append(
            f"High ratio of low-confidence claims ({claim_metrics['low_confidence_ratio']:.0%}); retrieve authoritative sources to verify key statements."
        )

    return {
        "citation_validity": citation_validity,
        "invalid_urls": invalid_urls,
        "source_relevance": source_relevance,
        "relevance_details": relevance_data,
        "evidence_gap_rate": evidence_gap_rate,
        "gap_subtask_ids": gap_subtask_ids,
        "min_sources": min_sources,
        "coverage_map": coverage_map,
        "budget_respected": budget_respected,
        "claim_count": claim_metrics["claim_count"],
        "claims_per_subtask": claim_metrics["claims_per_subtask"],
        "low_confidence_ratio": claim_metrics["low_confidence_ratio"],
        "low_confidence_claims_count": claim_metrics["low_confidence_claims_count"],
        "structured_claim_metrics": claim_metrics,
        "deterministic_instructions": instructions
    }
