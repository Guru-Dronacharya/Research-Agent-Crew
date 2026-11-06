"""Zero-LLM heuristic router for incoming chat queries.

Classifies queries into one of four deterministic outcomes:
1. "canned": Conversational greetings, farewells, or expressions of thanks (0 LLM calls).
2. "thread": Query can be grounded in the thread's existing indexed documents/report (>= CHAT_MIN_SCORE).
3. "quick": Short factual or definition question unsupported by thread; routed to single-search quick path.
4. "research": Deep-dive, comparative, multi-entity, opinion/best-of, or analytical query requiring full crew.
"""

import os
import re
from typing import Optional
from pydantic import BaseModel
from core.db.repositories import ChunkRepo, DocumentRepo, RunRepo
from core.rag import query_hybrid_rag, get_embedder

CHAT_MIN_SCORE = float(os.environ.get("CHAT_MIN_SCORE", "0.35"))

GREETINGS_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|greetings|good\s+morning|good\s+afternoon|good\s+evening|howdy|what'?s\s+up)\b[!?.]*\s*$",
    re.IGNORECASE
)

THANKS_FAREWELL_PATTERN = re.compile(
    r"^\s*(thanks|thank\s+you|thx|many\s+thanks|cheers|bye|goodbye|see\s+ya|have\s+a\s+good\s+day)\b[!?.]*\s*$",
    re.IGNORECASE
)

OPINION_PATTERN = re.compile(
    r"\b(best|top|recommended|favorite|favourite|worst|opinion|ideal|finest)\b",
    re.IGNORECASE
)

COMPARISON_PATTERN = re.compile(
    r"\b(vs|versus|compare|comparison|difference|differences|diff|trade-offs|tradeoffs|pros\s+and\s+cons)\b",
    re.IGNORECASE
)

RESEARCH_KEYWORDS_PATTERN = re.compile(
    r"\b(analyze|analysis|deep\s+dive|deepdive|survey|overview|comprehensive|evaluate|evaluation|benchmark|benchmarks|landscape|in-depth|roadmap|exhaustive|multi-agent)\b",
    re.IGNORECASE
)


class RouterDecision(BaseModel):
    """Container for router classification decision."""
    mode: str  # "canned", "thread", "quick", "research"
    reason: str
    canned_response: Optional[str] = None


def is_canned_query(query: str) -> Optional[RouterDecision]:
    """Detect if query is a greeting or thanks, returning canned decision if so."""
    q = query.strip()
    if GREETINGS_PATTERN.match(q):
        return RouterDecision(
            mode="canned",
            reason="Conversational greeting detected",
            canned_response="Hello! How can I assist you with this thread's knowledge or research today?"
        )
    if THANKS_FAREWELL_PATTERN.match(q):
        return RouterDecision(
            mode="canned",
            reason="Gratitude or farewell detected",
            canned_response="You're welcome! Let me know whenever you need further answers or analysis."
        )
    return None


def is_research_query(query: str) -> Optional[str]:
    """Detect if query requires full research crew based on complexity heuristics.
    
    Returns reason string if research, or None if potentially quick/thread.
    """
    words = query.strip().split()
    word_count = len(words)

    # 1. Length check: queries longer than ~20 words denote detailed requirements/goals
    if word_count > 20:
        return f"Query length ({word_count} words) exceeds 20 words limit for quick factual lookups"

    # 2. Opinion questions ("best vector database", "top practices") must route to research
    if OPINION_PATTERN.search(query):
        return "Opinion, ranking, or qualitative recommendation query ('best', 'top', etc.)"

    # 3. Comparison questions ("vs", "compare X and Y")
    if COMPARISON_PATTERN.search(query):
        return "Comparative or trade-off analysis query ('compare', 'vs', etc.)"

    # 4. Deep-dive, survey, benchmark, or analytical keywords
    if RESEARCH_KEYWORDS_PATTERN.search(query):
        return "In-depth analytical, evaluation, survey, or benchmark query"

    # 5. Multiple entity conjunctions (e.g. "X and Y and Z")
    if len(re.findall(r"\band\b", query, re.IGNORECASE)) >= 2:
        return "Multiple entity conjunctions requiring cross-topic research"

    return None


async def classify_query(
    query: str,
    thread_id: Optional[str] = None,
    db_path: Optional[str] = None,
    min_score: float = CHAT_MIN_SCORE
) -> RouterDecision:
    """Classify incoming query using zero-LLM heuristics into canned, thread, quick, or research."""
    # Step 1: Canned greeting / gratitude
    canned = is_canned_query(query)
    if canned:
        return canned

    # Step 2: Research complexity check
    research_reason = is_research_query(query)
    if research_reason:
        return RouterDecision(
            mode="research",
            reason=research_reason
        )

    # Step 3: Check if thread can support the question
    if thread_id:
        try:
            chunk_repo = ChunkRepo(db_path=db_path)
            doc_repo = DocumentRepo(db_path=db_path)
            run_repo = RunRepo(db_path=db_path)

            has_chunks = await chunk_repo.count_chunks_by_thread(thread_id) > 0
            runs = await run_repo.list_runs_by_thread(thread_id)
            has_report = any(r.report_md or r.report_json for r in runs if r.status in ("completed", "finished"))

            if has_chunks or has_report:
                embedder = get_embedder()
                retrieved, gap = await query_hybrid_rag(
                    thread_id=thread_id,
                    query=query,
                    embedder=embedder,
                    chunk_repo=chunk_repo,
                    document_repo=doc_repo,
                    top_k=3,
                    min_score=min_score
                )
                if not gap and retrieved and retrieved[0].score >= min_score:
                    return RouterDecision(
                        mode="thread",
                        reason=f"Thread has matching evidence (score {retrieved[0].score:.2f} >= {min_score})"
                    )
        except Exception:
            pass

    # Step 4: Short factual question unsupported by thread -> quick path
    return RouterDecision(
        mode="quick",
        reason="Short factual or definition query without matching thread evidence"
    )
