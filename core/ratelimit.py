"""Async Rate Limiter and Quota Management for Research Agent Crew.

Provides:
- Token-bucket / sliding-window rate limiting for LLM (LLM_RPM) and Search requests.
- Persistent SQLite daily request tracker per model (LLM_RPD) evaluated per Pacific Time calendar day.
- Atomic check-and-increment preventing concurrent over-quota calls.
- Fast-failure daily quota exhaustion tracking with human-readable reset timestamps.
"""

import os
import time
import sqlite3
import asyncio
import threading
from pathlib import Path
from typing import Optional
from collections import deque
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")


class QuotaExceededError(RuntimeError):
    """Raised when daily request quota (LLM_RPD) is exceeded."""
    def __init__(
        self,
        message: str,
        reset_time: Optional[str] = None,
        resume_after_iso: Optional[str] = None
    ):
        super().__init__(message)
        self.reset_time = reset_time
        self.resume_after_iso = resume_after_iso


class DailyModelQuotaTracker:
    """Thread-safe, SQLite-backed persistent daily quota counter per model.

    Evaluated against calendar days in Pacific Time (America/Los_Angeles),
    matching Google Gemini quota reset boundaries at midnight PT.
    """

    def __init__(self, db_path: Optional[str] = None):
        resolved = db_path or os.environ.get("QUOTA_DB_PATH", ".cache/quota.db")
        self.db_path = Path(resolved).resolve()
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database with WAL mode and schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_daily_quota (
                    model TEXT NOT NULL,
                    date_pt TEXT NOT NULL,
                    request_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (model, date_pt)
                );
            """)
            conn.commit()

    def get_pacific_now(self) -> datetime:
        """Return current datetime in Pacific Time."""
        return datetime.now(PACIFIC_TZ)

    def get_current_pacific_date(self) -> str:
        """Return YYYY-MM-DD for current Pacific date."""
        return self.get_pacific_now().strftime("%Y-%m-%d")

    def get_reset_time_description(self) -> str:
        """Return formatted description of the next midnight PT quota reset."""
        now_pt = self.get_pacific_now()
        tomorrow_pt = (now_pt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        reset_pt_str = tomorrow_pt.strftime("%Y-%m-%d 00:00:00 %Z")
        reset_local_str = tomorrow_pt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        return f"Quota resets at {reset_pt_str} ({reset_local_str} local)"

    def get_reset_time_iso(self) -> str:
        """Return ISO-8601 UTC timestamp of the next midnight PT quota reset."""
        now_pt = self.get_pacific_now()
        tomorrow_pt = (now_pt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return tomorrow_pt.astimezone(timezone.utc).isoformat()

    def check_and_increment(self, model: str, rpd_limit: Optional[int] = None) -> int:
        """Atomically check quota and increment request counter.

        Raises:
            QuotaExceededError: If current count is already >= rpd_limit.
        """
        clean_model = model.strip() if model else "default"
        date_pt = self.get_current_pacific_date()
        reset_msg = self.get_reset_time_description()

        with self._lock:
            with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                conn.isolation_level = None  # Autocommit mode to manage BEGIN explicitly
                cursor = conn.cursor()
                cursor.execute("BEGIN IMMEDIATE;")

                cursor.execute(
                    "SELECT request_count FROM model_daily_quota WHERE model = ? AND date_pt = ?",
                    (clean_model, date_pt)
                )
                row = cursor.fetchone()
                current_count = row[0] if row else 0

                if rpd_limit is not None and rpd_limit > 0 and current_count >= rpd_limit:
                    cursor.execute("ROLLBACK;")
                    reset_iso = self.get_reset_time_iso()
                    raise QuotaExceededError(
                        f"Daily quota exceeded for model '{clean_model}': {current_count}/{rpd_limit} requests used on {date_pt} PT. {reset_msg}.",
                        reset_time=reset_msg,
                        resume_after_iso=reset_iso
                    )

                cursor.execute("""
                    INSERT INTO model_daily_quota (model, date_pt, request_count)
                    VALUES (?, ?, 1)
                    ON CONFLICT(model, date_pt) DO UPDATE SET
                        request_count = model_daily_quota.request_count + 1
                    RETURNING request_count;
                """, (clean_model, date_pt))
                new_count = cursor.fetchone()[0]
                cursor.execute("COMMIT;")
                return new_count

    def set_quota_exhausted(self, model: str, limit: Optional[int] = None) -> None:
        """Fast-fail marker: set model count to at least the limit after a daily-quota 429."""
        clean_model = model.strip() if model else "default"
        date_pt = self.get_current_pacific_date()
        target = limit if limit is not None and limit > 0 else 1000

        try:
            env_rpd = int(os.environ.get("LLM_RPD", "0"))
            if env_rpd > 0:
                target = env_rpd
        except ValueError:
            pass

        with self._lock:
            with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                conn.execute("""
                    INSERT INTO model_daily_quota (model, date_pt, request_count)
                    VALUES (?, ?, ?)
                    ON CONFLICT(model, date_pt) DO UPDATE SET
                        request_count = MAX(model_daily_quota.request_count, ?);
                """, (clean_model, date_pt, target, target))
                conn.commit()

    def get_count(self, model: str, date_pt: Optional[str] = None) -> int:
        """Return current request count for model on the given or current PT date."""
        clean_model = model.strip() if model else "default"
        target_date = date_pt or self.get_current_pacific_date()
        with self._lock:
            with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT request_count FROM model_daily_quota WHERE model = ? AND date_pt = ?",
                    (clean_model, target_date)
                )
                row = cur.fetchone()
                return row[0] if row else 0

    def reset(self, model: Optional[str] = None) -> None:
        """Clear counts (for testing)."""
        with self._lock:
            with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                if model:
                    conn.execute("DELETE FROM model_daily_quota WHERE model = ?", (model,))
                else:
                    conn.execute("DELETE FROM model_daily_quota;")
                conn.commit()


_GLOBAL_QUOTA_TRACKER: Optional[DailyModelQuotaTracker] = None


def get_daily_quota_tracker() -> DailyModelQuotaTracker:
    """Return singleton DailyModelQuotaTracker instance."""
    global _GLOBAL_QUOTA_TRACKER
    if _GLOBAL_QUOTA_TRACKER is None:
        _GLOBAL_QUOTA_TRACKER = DailyModelQuotaTracker()
    return _GLOBAL_QUOTA_TRACKER


def set_global_quota_tracker(tracker: Optional[DailyModelQuotaTracker]) -> None:
    """Override singleton tracker for tests."""
    global _GLOBAL_QUOTA_TRACKER
    _GLOBAL_QUOTA_TRACKER = tracker


class AsyncSlidingWindowLimiter:
    """Sliding-window async rate limiter with minute pacing and persistent daily caps."""

    def __init__(
        self,
        rpm: float,
        rpd: Optional[int] = None,
        name: str = "limiter"
    ):
        self.rpm = max(1.0, float(rpm))
        self.rpd = int(rpd) if rpd is not None and int(rpd) > 0 else None
        self.name = name
        self._minute_window: deque[float] = deque()
        self._lock = asyncio.Lock()

    def get_rpm(self) -> float:
        return self.rpm

    def set_rpm(self, rpm: float) -> None:
        self.rpm = max(1.0, float(rpm))

    def set_rpd(self, rpd: Optional[int]) -> None:
        self.rpd = int(rpd) if rpd is not None and int(rpd) > 0 else None

    async def acquire(self, model: Optional[str] = None) -> None:
        """Acquire a slot, pacing by RPM and checking/incrementing persistent RPD."""
        # 1. Enforce persistent daily quota counter before every call/retry
        rpd_limit = self.rpd
        if rpd_limit is None and self.name == "LLM":
            rpd_env = os.environ.get("LLM_RPD")
            if rpd_env and rpd_env.isdigit():
                rpd_limit = int(rpd_env)

        if rpd_limit is not None and rpd_limit > 0:
            target_model = model or (os.environ.get("GEMINI_MODEL", "gemini-default") if self.name == "LLM" else self.name)
            tracker = get_daily_quota_tracker()
            # Atomically check and increment in SQLite
            await asyncio.to_thread(tracker.check_and_increment, target_model, rpd_limit)

        # 2. Minute window pacing
        while True:
            async with self._lock:
                now = time.monotonic()
                cutoff_minute = now - 60.0
                while self._minute_window and self._minute_window[0] < cutoff_minute:
                    self._minute_window.popleft()

                if len(self._minute_window) < self.rpm:
                    self._minute_window.append(now)
                    return

                sleep_time = max(0.01, (self._minute_window[0] + 60.0) - now)

            await asyncio.sleep(sleep_time)

    def reset(self) -> None:
        """Clear limiter state (useful for test suites)."""
        self._minute_window.clear()


_LLM_LIMITER: Optional[AsyncSlidingWindowLimiter] = None
_SEARCH_LIMITER: Optional[AsyncSlidingWindowLimiter] = None


def get_llm_rate_limiter() -> AsyncSlidingWindowLimiter:
    """Return singleton LLM rate limiter configured via LLM_RPM and LLM_RPD."""
    global _LLM_LIMITER
    if _LLM_LIMITER is None:
        try:
            rpm = float(os.environ.get("LLM_RPM", "15"))
        except ValueError:
            rpm = 15.0
        rpd_val = os.environ.get("LLM_RPD")
        rpd = int(rpd_val) if rpd_val and rpd_val.isdigit() else None
        _LLM_LIMITER = AsyncSlidingWindowLimiter(rpm=rpm, rpd=rpd, name="LLM")
    return _LLM_LIMITER


def get_search_rate_limiter() -> AsyncSlidingWindowLimiter:
    """Return singleton search rate limiter configured via SEARCH_RPM."""
    global _SEARCH_LIMITER
    if _SEARCH_LIMITER is None:
        try:
            rpm = float(os.environ.get("SEARCH_RPM", "30"))
        except ValueError:
            rpm = 30.0
        _SEARCH_LIMITER = AsyncSlidingWindowLimiter(rpm=rpm, rpd=None, name="Search")
    return _SEARCH_LIMITER


def reset_rate_limiters() -> None:
    """Reset rate limiter state (used in conftest fixtures)."""
    if _LLM_LIMITER:
        _LLM_LIMITER.reset()
    if _SEARCH_LIMITER:
        _SEARCH_LIMITER.reset()
    if _GLOBAL_QUOTA_TRACKER:
        _GLOBAL_QUOTA_TRACKER.reset()
