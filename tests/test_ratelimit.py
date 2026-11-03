"""Unit tests for rate limiters, daily quotas, Retry-After parsing, and graceful escalation."""

import time
import pytest
from core.ratelimit import AsyncSlidingWindowLimiter, QuotaExceededError
from core.llm import _extract_retry_after
from core.graph import build_research_graph, create_initial_state
from tests.stubs import StubLLM
from core.llm import set_global_llm


@pytest.mark.asyncio
async def test_sliding_window_limiter_spacing():
    """Verify that limiter paces requests when RPM threshold is met."""
    # 60 RPM = 1 req/sec average, but test with high rate so test is fast:
    # 120 RPM = 2 req/sec. Let's test 120 RPM: capacity 2 per second
    limiter = AsyncSlidingWindowLimiter(rpm=2, name="TestLimiter")

    t0 = time.monotonic()
    await limiter.acquire()
    await limiter.acquire()
    # 3rd acquire must wait ~60s if window was 60s. For unit test, test window logic directly:
    # Set minute window to past times:
    limiter._minute_window[0] = time.monotonic() - 59.8
    # Now acquiring should take ~0.2s, not 60s
    await limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.15


@pytest.mark.asyncio
async def test_daily_quota_exceeded_raises():
    """Verify that reaching LLM_RPD raises QuotaExceededError."""
    limiter = AsyncSlidingWindowLimiter(rpm=100, rpd=3, name="DailyTest")

    await limiter.acquire()
    await limiter.acquire()
    await limiter.acquire()

    with pytest.raises(QuotaExceededError) as exc_info:
        await limiter.acquire()
    assert "Daily quota exceeded" in str(exc_info.value)


def test_retry_after_extraction():
    """Verify Retry-After is correctly extracted from various error representations."""
    class MockError(Exception):
        pass

    # From retryDelay
    err1 = MockError("{'error': {'status': 'RESOURCE_EXHAUSTED', 'retryDelay': '12s'}}")
    assert _extract_retry_after(err1) == 12.0

    # From retry in X s
    err2 = MockError("Please retry in 5s.")
    assert _extract_retry_after(err2) == 5.0

    # From retry-after
    err3 = MockError("Rate limited. Retry-After: 7")
    assert _extract_retry_after(err3) == 7.0

    # From HTTP response headers
    class MockResponse:
        headers = {"retry-after": "4"}

    err4 = MockError("HTTP 429")
    err4.response = MockResponse()
    assert _extract_retry_after(err4) == 4.0


@pytest.mark.asyncio
async def test_quota_exceeded_graph_graceful_escalation(monkeypatch):
    """Verify QuotaExceededError causes graph to escalate gracefully without crashing."""
    class QuotaFailLLM(StubLLM):
        async def generate_plan(self, *args, **kwargs):
            raise QuotaExceededError("Simulated daily quota exhausted")

    set_global_llm(QuotaFailLLM())

    graph = build_research_graph()
    initial_state = create_initial_state(
        task="Test Quota Escalation",
        max_retries=1,
        max_search_budget=4
    )

    final_state = await graph.ainvoke(initial_state)

    assert final_state["human_escalation_needed"] is True
    assert "Daily LLM quota exceeded" in (final_state.get("escalation_reason") or "")
    assert final_state["status"] in ("escalated", "finished", "completed")
    # Verify finalizer still produced scorecard with human review warning
    assert "Human Review Recommended" in (final_state.get("final_report") or "")


@pytest.mark.asyncio
async def test_concurrent_increments_never_exceed_limit(tmp_path):
    """Concurrent increment calls must never exceed LLM_RPD limit."""
    import asyncio
    from core.ratelimit import DailyModelQuotaTracker

    db_file = str(tmp_path / "concurrent_quota.db")
    tracker = DailyModelQuotaTracker(db_path=db_file)
    rpd_limit = 5
    num_callers = 15

    succeeded = 0
    failed = 0

    async def _try_increment():
        nonlocal succeeded, failed
        try:
            await asyncio.to_thread(tracker.check_and_increment, "concurrent-model", rpd_limit)
            succeeded += 1
        except QuotaExceededError:
            failed += 1

    tasks = [_try_increment() for _ in range(num_callers)]
    await asyncio.gather(*tasks)

    assert succeeded == rpd_limit
    assert failed == (num_callers - rpd_limit)
    assert tracker.get_count("concurrent-model") == rpd_limit


def test_counter_survives_a_restart(tmp_path):
    """Per-model count in SQLite must persist across process / instance restarts."""
    from core.ratelimit import DailyModelQuotaTracker

    db_file = str(tmp_path / "persistent_quota.db")
    tracker1 = DailyModelQuotaTracker(db_path=db_file)
    tracker1.check_and_increment("model-alpha", rpd_limit=10)
    tracker1.check_and_increment("model-alpha", rpd_limit=10)
    tracker1.check_and_increment("model-alpha", rpd_limit=10)
    assert tracker1.get_count("model-alpha") == 3

    # Simulate restart by creating brand new instance pointing to same DB file
    tracker2 = DailyModelQuotaTracker(db_path=db_file)
    assert tracker2.get_count("model-alpha") == 3
    new_count = tracker2.check_and_increment("model-alpha", rpd_limit=10)
    assert new_count == 4
    assert tracker2.get_count("model-alpha") == 4


def test_429_with_long_delay_is_not_retried():
    """Errors with retryDelay > 60s (daily quota exhaustion) are classified as non-transient."""
    from core.llm import _is_transient_error

    class MockLongDelayError(Exception):
        pass

    # 120s delay (> 60s) -> Daily quota exhaustion, NOT transient
    long_delay_err = MockLongDelayError("RESOURCE_EXHAUSTED: retryDelay: 120s")
    assert _is_transient_error(long_delay_err) is False

    # 15s delay (<= 60s) -> Transient rate limit, retryable
    short_delay_err = MockLongDelayError("RESOURCE_EXHAUSTED: retryDelay: 15s")
    assert _is_transient_error(short_delay_err) is True


def test_daily_quota_429_sets_counter_to_limit(tmp_path):
    """When a daily quota 429 occurs, local counter is set to limit so subsequent calls fail fast."""
    from core.ratelimit import DailyModelQuotaTracker

    db_file = str(tmp_path / "fail_fast_quota.db")
    tracker = DailyModelQuotaTracker(db_path=db_file)
    model = "gemini-pro-test"

    # Counter initially 1
    tracker.check_and_increment(model, rpd_limit=50)
    assert tracker.get_count(model) == 1

    # Simulate daily 429 trigger: set quota exhausted to 50
    tracker.set_quota_exhausted(model, limit=50)
    assert tracker.get_count(model) >= 50

    # Next attempt fails fast locally without making an external call
    with pytest.raises(QuotaExceededError) as exc_info:
        tracker.check_and_increment(model, rpd_limit=50)
    assert "Daily quota exceeded" in str(exc_info.value)
    assert tracker.get_reset_time_description() in str(exc_info.value)

