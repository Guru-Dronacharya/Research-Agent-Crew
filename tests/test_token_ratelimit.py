"""Unit tests for token rate limiting, reservation lifecycle, SQLite persistence, and limits file."""

import json
import asyncio
import sqlite3
import pytest

from core.ratelimit import (
    DailyModelQuotaTracker,
    QuotaExceededError,
    ConfigurationError,
    get_model_limits,
    compute_reset_timestamps
)


def test_quota_db_schema_and_version(tmp_path):
    """Verify quota DB initializes schema version and tables."""
    db_file = str(tmp_path / "test_schema.db")
    DailyModelQuotaTracker(db_path=db_file)

    with sqlite3.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("SELECT MAX(version) FROM quota_db_schema_version;")
        ver = cur.fetchone()[0]
        assert ver >= 2

        # Check llm_call_events table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='llm_call_events';")
        assert cur.fetchone() is not None

        # Check legacy model_daily_quota table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='model_daily_quota';")
        assert cur.fetchone() is not None


def test_limits_file_resolution(tmp_path, monkeypatch):
    """Verify LLM_LIMITS_FILE correctly parses per-model limits with fallback to env."""
    limits_data = {
        "special-model": {
            "rpm": 30,
            "rpd": 500,
            "tpm": 6000,
            "tpd": 100000,
            "day_reset": "utc_midnight"
        }
    }
    limits_path = tmp_path / "model_limits.json"
    limits_path.write_text(json.dumps(limits_data), encoding="utf-8")

    monkeypatch.setenv("LLM_LIMITS_FILE", str(limits_path))
    monkeypatch.setenv("LLM_RPM", "10")
    monkeypatch.setenv("LLM_TPM", "2000")
    monkeypatch.setenv("LLM_DAY_RESET", "pacific_midnight")

    # Listed model
    special_limits = get_model_limits("special-model")
    assert special_limits["rpm"] == 30.0
    assert special_limits["rpd"] == 500
    assert special_limits["tpm"] == 6000
    assert special_limits["tpd"] == 100000
    assert special_limits["day_reset"] == "utc_midnight"

    # Unlisted model falls back to env
    other_limits = get_model_limits("other-model")
    assert other_limits["rpm"] == 10.0
    assert other_limits["tpm"] == 2000
    assert other_limits["day_reset"] == "pacific_midnight"


def test_fail_fast_single_call_exceeds_tpm(tmp_path, monkeypatch):
    """When a single call's estimated tokens exceed TPM, fail fast with ConfigurationError."""
    db_file = str(tmp_path / "tpm_fail.db")
    tracker = DailyModelQuotaTracker(db_path=db_file)

    monkeypatch.setenv("LLM_TPM", "1000")
    # Call requiring 1500 tokens must immediately fail fast
    with pytest.raises(ConfigurationError) as exc:
        tracker.reserve("groq", "test-model", estimated_tokens=1500)
    assert "exceeds total per-minute limit (TPM=1000)" in str(exc.value)
    assert "Reduce RAG_MAX_CONTEXT_CHARS" in str(exc.value)


def test_fail_fast_daily_tpd_exceeded(tmp_path, monkeypatch):
    """When daily token capacity is exceeded, fail fast with QuotaExceededError."""
    db_file = str(tmp_path / "tpd_fail.db")
    tracker = DailyModelQuotaTracker(db_path=db_file)

    monkeypatch.setenv("LLM_TPD", "5000")
    # Reserve 4000 tokens
    ev1 = tracker.reserve("groq", "tpd-model", estimated_tokens=4000)
    tracker.complete(ev1, actual_tokens=4000)

    # Next call requiring 1500 tokens exceeds 5000 limit (4000 + 1500 > 5000)
    with pytest.raises(QuotaExceededError) as exc:
        tracker.reserve("groq", "tpd-model", estimated_tokens=1500)
    assert "Daily token quota exceeded" in str(exc.value)
    assert exc.value.reset_time is not None


def test_reservation_lifecycle_complete_release_ambiguous(tmp_path):
    """Verify complete, release, and mark_ambiguous token reconciliation in SQLite."""
    db_file = str(tmp_path / "lifecycle.db")
    tracker = DailyModelQuotaTracker(db_path=db_file)

    # 1. Complete reconciles actual usage
    ev1 = tracker.reserve("groq", "model-a", estimated_tokens=1000)
    tracker.complete(ev1, actual_tokens=650)  # Reconciled down to 650

    with sqlite3.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("SELECT tokens_reserved, tokens_actual, status FROM llm_call_events WHERE id = ?", (ev1,))
        r1, a1, s1 = cur.fetchone()
        assert r1 == 1000
        assert a1 == 650
        assert s1 == "completed"

    # 2. Release zeroes reserved tokens (for local error)
    ev2 = tracker.reserve("groq", "model-a", estimated_tokens=800)
    tracker.release(ev2, reason="ValidationError")

    with sqlite3.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("SELECT tokens_reserved, tokens_actual, status, error_kind FROM llm_call_events WHERE id = ?", (ev2,))
        r2, a2, s2, e2 = cur.fetchone()
        assert r2 == 0
        assert a2 == 0
        assert s2 == "released"
        assert e2 == "ValidationError"

    # 3. Mark ambiguous keeps tokens reserved (for timeout)
    ev3 = tracker.reserve("groq", "model-a", estimated_tokens=500)
    tracker.mark_ambiguous(ev3, reason="TimeoutError")

    with sqlite3.connect(db_file) as conn:
        cur = conn.cursor()
        cur.execute("SELECT tokens_reserved, tokens_actual, status, error_kind FROM llm_call_events WHERE id = ?", (ev3,))
        r3, a3, s3, e3 = cur.fetchone()
        assert r3 == 500
        assert a3 is None
        assert s3 == "ambiguous"
        assert e3 == "TimeoutError"


@pytest.mark.asyncio
@pytest.mark.critical
async def test_concurrent_reservations_safety(tmp_path, monkeypatch):
    """Multiple concurrent async tasks safely acquire reservations under RPD limit."""
    db_file = str(tmp_path / "concurrent_tokens.db")
    tracker = DailyModelQuotaTracker(db_path=db_file)
    monkeypatch.setenv("LLM_RPD", "5")

    succeeded = 0
    failed = 0

    async def _try_reserve():
        nonlocal succeeded, failed
        try:
            ev = await asyncio.to_thread(tracker.reserve, "gemini", "conc-model", 100)
            await asyncio.to_thread(tracker.complete, ev, 100)
            succeeded += 1
        except QuotaExceededError:
            failed += 1

    tasks = [_try_reserve() for _ in range(12)]
    await asyncio.gather(*tasks)

    assert succeeded == 5
    assert failed == 7


def test_counter_survives_process_restart(tmp_path):
    """Call events and counts survive new instance instantiation."""
    db_file = str(tmp_path / "restart.db")
    tracker1 = DailyModelQuotaTracker(db_path=db_file)
    ev = tracker1.reserve("groq", "model-p", 300)
    tracker1.complete(ev, 300)

    # Re-instantiate
    tracker2 = DailyModelQuotaTracker(db_path=db_file)
    used_reqs, used_tokens = tracker2.get_sliding_window_usage("groq", "model-p", 60.0)
    assert used_reqs == 1
    assert used_tokens == 300


def test_reset_timestamp_modes():
    """Verify compute_reset_timestamps for pacific_midnight, utc_midnight, rolling_24h."""
    _, desc_pt, iso_pt = compute_reset_timestamps("pacific_midnight")
    assert "PT" in desc_pt or "Pacific" in desc_pt or "00:00:00" in desc_pt
    assert iso_pt.endswith("+00:00") or "Z" in iso_pt or "T" in iso_pt

    _, desc_utc, iso_utc = compute_reset_timestamps("utc_midnight")
    assert "UTC" in desc_utc

    _, desc_roll, _ = compute_reset_timestamps("rolling_24h")
    assert "24-hour" in desc_roll
