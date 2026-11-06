"""Base interfaces, retry decorators, and token helpers for LLM providers."""

import os
import re
from typing import Optional, List, Dict, Any, Tuple, AsyncIterator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)
from tenacity.wait import wait_base

from core.state import ResearchPlan, CriticEvaluation, SubtaskResult, StructuredReport
from core.ratelimit import QuotaExceededError, ConfigurationError


def estimate_tokens(text: str, chars_per_token: Optional[int] = None) -> int:
    """Estimate token count for a text using chars_per_token (default 4)."""
    if not text:
        return 0
    try:
        cpt = chars_per_token or int(os.environ.get("TOKEN_CHARS_PER_TOKEN", "4"))
    except ValueError:
        cpt = 4
    cpt = max(1, cpt)
    return max(1, len(text) // cpt)


def strip_reasoning_tags(text: str) -> Tuple[str, int]:
    """Strip <think>...</think> and <thought>...</thought> reasoning wrappers.

    Returns:
        (clean_text, stripped_token_estimate)
    """
    if not text:
        return "", 0

    pattern = r"<think>(.*?)</think>|<thought>(.*?)</thought>"
    matches = re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    stripped_chars = 0
    for m in matches:
        content = m[0] or m[1] or ""
        stripped_chars += len(content)

    clean_text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    return clean_text, max(0, stripped_chars // 4)


def _parse_duration_string(val: str) -> Optional[float]:
    """Parse duration strings like '1.2s', '250ms', '6m30s' into seconds."""
    if not val:
        return None
    val = val.strip()
    try:
        return float(val)
    except ValueError:
        pass
    pattern = r"(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)"
    matches = re.findall(pattern, val, re.IGNORECASE)
    if not matches:
        return None
    multipliers = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    total = 0.0
    for num, unit in matches:
        total += float(num) * multipliers[unit.lower()]
    return total


def _extract_retry_after(exc: BaseException) -> Optional[float]:
    """Extract retry-after seconds from exception attributes or response headers."""
    # 1. Check HTTP response headers (groq SDK and Google genai SDK)
    response = getattr(exc, "response", None)
    if response and hasattr(response, "headers"):
        headers = response.headers

        # retry-after-ms (highest precision)
        ms_val = headers.get("retry-after-ms")
        if ms_val:
            try:
                return float(ms_val) / 1000.0
            except ValueError:
                pass

        # retry-after
        ra_val = headers.get("retry-after")
        if ra_val:
            try:
                return float(ra_val)
            except ValueError:
                pass

        # Groq reset headers
        req_reset = headers.get("x-ratelimit-reset-requests")
        if req_reset:
            dur = _parse_duration_string(req_reset)
            if dur is not None:
                return dur

        tok_reset = headers.get("x-ratelimit-reset-tokens")
        if tok_reset:
            dur = _parse_duration_string(tok_reset)
            if dur is not None:
                return dur

    msg = str(exc)

    # 2. Regex fallback for message bodies
    m = re.search(r"retry[-_]?after[:=\s]+(\d+(?:\.\d+)?)", msg, re.IGNORECASE)
    if m:
        return float(m.group(1))

    m2 = re.search(r"retryDelay['\":\s]+(\d+(?:\.\d+)?)s?", msg, re.IGNORECASE)
    if m2:
        return float(m2.group(1))

    m3 = re.search(r"retry in (\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
    if m3:
        return float(m3.group(1))

    return None


def _is_transient_error(exc: BaseException) -> bool:
    """Detect transient 429 / rate limit / network / server errors.

    Returns False for QuotaExceededError or long-delay daily quota 429s (> 60s).
    """
    if isinstance(exc, QuotaExceededError):
        return False
    if isinstance(exc, ConfigurationError):
        return False

    delay = _extract_retry_after(exc)
    if delay is not None and delay > 60.0:
        return False

    msg = str(exc).lower()
    if "per day" in msg or ("daily" in msg and "quota" in msg):
        return False

    transient_indicators = [
        "429", "rate limit", "resource_exhausted", "resourceexhausted",
        "quota", "503", "500", "overloaded", "temporary", "timeout",
        "connection reset", "econnreset", "try again"
    ]
    return any(indicator in msg for indicator in transient_indicators)


class wait_with_retry_after(wait_base):
    """Tenacity wait strategy that respects Retry-After with fallback to exponential backoff."""
    def __init__(self, fallback_strategy, max_retry_after: float = 30.0):
        self.fallback = fallback_strategy
        self.max_retry_after = max_retry_after

    def __call__(self, retry_state):
        if retry_state.outcome and retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            delay = _extract_retry_after(exc)
            if delay is not None and delay > 0:
                return min(delay, self.max_retry_after)
        return self.fallback(retry_state)


llm_retry = retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_with_retry_after(wait_exponential(multiplier=1, min=2, max=10)),
    retry=retry_if_exception(_is_transient_error)
)


class BaseAgentLLM:
    """Abstract interface for agent LLM reasoning."""

    async def generate_plan(
        self,
        task: str,
        critic_evaluation: Optional[CriticEvaluation] = None,
        iteration: int = 0,
        subtask_results: Optional[List[SubtaskResult]] = None
    ) -> ResearchPlan:
        raise NotImplementedError

    async def synthesize_findings(
        self,
        task: str,
        plan: ResearchPlan,
        subtask_results: List[SubtaskResult],
        critic_evaluation: Optional[CriticEvaluation] = None,
        iteration: int = 0
    ) -> Tuple[str, StructuredReport]:
        raise NotImplementedError

    async def evaluate_report(
        self,
        task: str,
        plan: ResearchPlan,
        report: str,
        subtask_results: List[SubtaskResult],
        iteration: int = 0,
        deterministic_metrics: Optional[Dict[str, Any]] = None,
        structured_report: Optional[StructuredReport] = None
    ) -> CriticEvaluation:
        raise NotImplementedError

    async def astream_chat(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None
    ) -> AsyncIterator[str]:
        raise NotImplementedError
        yield ""
