"""Async Rate Limiter and Quota Management for Research Agent Crew.

Provides:
- Token-bucket / sliding-window rate limiting for LLM (LLM_RPM, LLM_TPM) and Search requests.
- Persistent SQLite quota and call-event tracking (LLM_RPD, LLM_TPD) evaluated per configurable
  day reset mode (pacific_midnight, utc_midnight, rolling_24h).
- Per-model limits support via LLM_LIMITS_FILE (JSON mapping of model -> limits) with fallback
  to global env vars.
- Reservation lifecycle (reserve -> complete / release / mark_ambiguous) with atomic checks (BEGIN IMMEDIATE).
- Immediate fail-fast when a single call's token requirement exceeds TPM or remaining TPD.
- Backward compatibility for legacy DailyModelQuotaTracker and model_daily_quota schema.
"""

import os
import json
import time
import sqlite3
import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from collections import deque
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")


class ConfigurationError(RuntimeError):
    """Raised when required configuration or limits are invalid or missing."""
    pass


class QuotaExceededError(RuntimeError):
    """Raised when daily request quota (RPD) or token quota (TPD) is exceeded."""
    def __init__(
        self,
        message: str,
        reset_time: Optional[str] = None,
        resume_after_iso: Optional[str] = None
    ):
        super().__init__(message)
        self.reset_time = reset_time
        self.resume_after_iso = resume_after_iso


def get_model_limits(model: str) -> Dict[str, Any]:
    """Resolve rate limits for a specific model.

    Priority:
    1. Entry in LLM_LIMITS_FILE (if configured and valid JSON)
    2. Global environment variables: LLM_RPM, LLM_RPD, LLM_TPM, LLM_TPD, LLM_DAY_RESET
    """
    clean_model = model.strip() if model else "default"
    limits_file = os.environ.get("LLM_LIMITS_FILE", "").strip()

    file_limits: Dict[str, Any] = {}
    if limits_file and os.path.exists(limits_file):
        try:
            with open(limits_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    file_limits = data.get(clean_model, {}) or data.get("*", {})
        except Exception:
            pass

    def _parse_opt_int(val: Any) -> Optional[int]:
        if val is None:
            return None
        try:
            i = int(val)
            return i if i > 0 else None
        except (ValueError, TypeError):
            return None

    def _parse_opt_float(val: Any, default: float) -> float:
        if val is None:
            return default
        try:
            f = float(val)
            return max(1.0, f)
        except (ValueError, TypeError):
            return default

    # Fallback to env vars
    env_rpm = _parse_opt_float(os.environ.get("LLM_RPM"), 15.0)
    env_rpd = _parse_opt_int(os.environ.get("LLM_RPD"))
    env_tpm = _parse_opt_int(os.environ.get("LLM_TPM"))
    env_tpd = _parse_opt_int(os.environ.get("LLM_TPD"))
    env_reset = os.environ.get("LLM_DAY_RESET", "pacific_midnight").strip().lower()

    return {
        "rpm": _parse_opt_float(file_limits.get("rpm"), env_rpm),
        "rpd": _parse_opt_int(file_limits.get("rpd")) if "rpd" in file_limits else env_rpd,
        "tpm": _parse_opt_int(file_limits.get("tpm")) if "tpm" in file_limits else env_tpm,
        "tpd": _parse_opt_int(file_limits.get("tpd")) if "tpd" in file_limits else env_tpd,
        "day_reset": file_limits.get("day_reset", env_reset),
    }


def compute_reset_timestamps(day_reset_mode: str) -> Tuple[float, str, str]:
    """Compute day start timestamp, human-readable reset string, and ISO UTC reset timestamp.

    Supported modes:
    - 'pacific_midnight': Resets at midnight PT (Google Gemini default)
    - 'utc_midnight': Resets at midnight UTC
    - 'rolling_24h': Rolling 24-hour window from current time
    """
    mode = (day_reset_mode or "pacific_midnight").lower()
    now_utc = datetime.now(timezone.utc)

    if mode == "utc_midnight":
        now_day = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        day_start_ts = now_day.timestamp()
        tomorrow_utc = now_day + timedelta(days=1)
        reset_desc = f"Quota resets at {tomorrow_utc.strftime('%Y-%m-%d 00:00:00 UTC')} ({tomorrow_utc.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')} local)"
        reset_iso = tomorrow_utc.isoformat()
        return day_start_ts, reset_desc, reset_iso

    elif mode == "rolling_24h":
        day_start_ts = time.time() - 86400.0
        reset_desc = "Quota rolls over continuously on a 24-hour sliding window"
        reset_iso = (now_utc + timedelta(hours=1)).isoformat()
        return day_start_ts, reset_desc, reset_iso

    else:  # 'pacific_midnight'
        now_pt = datetime.now(PACIFIC_TZ)
        now_pt_day = now_pt.replace(hour=0, minute=0, second=0, microsecond=0)
        day_start_ts = now_pt_day.timestamp()
        tomorrow_pt = now_pt_day + timedelta(days=1)
        reset_desc = f"Quota resets at {tomorrow_pt.strftime('%Y-%m-%d 00:00:00 %Z')} ({tomorrow_pt.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')} local)"
        reset_iso = tomorrow_pt.astimezone(timezone.utc).isoformat()
        return day_start_ts, reset_desc, reset_iso


class DailyModelQuotaTracker:
    """Thread-safe, SQLite-backed persistent quota counter and event log.

    Handles both the legacy model_daily_quota schema and the newer event-based
    llm_call_events table with automatic schema migrations.
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
            
            # Tiny schema version table for the quota DB
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quota_db_schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );
            """)

            # Legacy table: kept for backwards compatibility and test suites
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_daily_quota (
                    model TEXT NOT NULL,
                    date_pt TEXT NOT NULL,
                    request_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (model, date_pt)
                );
            """)

            # Check version
            cur = conn.cursor()
            cur.execute("SELECT MAX(version) FROM quota_db_schema_version;")
            row = cur.fetchone()
            curr_ver = row[0] if row and row[0] is not None else 0

            if curr_ver < 1:
                cur.execute("INSERT OR REPLACE INTO quota_db_schema_version (version, applied_at) VALUES (1, ?)",
                            (datetime.now(timezone.utc).isoformat(),))

            if curr_ver < 2:
                # Call events table for unified token and request tracking
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS llm_call_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts REAL NOT NULL,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        tokens_reserved INTEGER NOT NULL,
                        tokens_actual INTEGER,
                        status TEXT NOT NULL,
                        error_kind TEXT,
                        created_at TEXT NOT NULL
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_llm_events_lookup
                    ON llm_call_events (provider, model, ts, status);
                """)
                cur.execute("INSERT OR REPLACE INTO quota_db_schema_version (version, applied_at) VALUES (2, ?)",
                            (datetime.now(timezone.utc).isoformat(),))

            conn.commit()

    def get_pacific_now(self) -> datetime:
        """Return current datetime in Pacific Time."""
        return datetime.now(PACIFIC_TZ)

    def get_current_pacific_date(self) -> str:
        """Return YYYY-MM-DD for current Pacific date."""
        return self.get_pacific_now().strftime("%Y-%m-%d")

    def get_reset_time_description(self, day_reset_mode: str = "pacific_midnight") -> str:
        """Return formatted description of the next quota reset."""
        _, desc, _ = compute_reset_timestamps(day_reset_mode)
        return desc

    def get_reset_time_iso(self, day_reset_mode: str = "pacific_midnight") -> str:
        """Return ISO-8601 UTC timestamp of the next quota reset."""
        _, _, iso = compute_reset_timestamps(day_reset_mode)
        return iso

    def reserve(
        self,
        provider: str,
        model: str,
        estimated_tokens: int,
        day_reset_mode: Optional[str] = None
    ) -> int:
        """Atomically check limits and record a reservation event.

        Raises:
            ConfigurationError: If single call tokens exceed TPM.
            QuotaExceededError: If RPD or TPD limit is exceeded.
        """
        clean_provider = provider.strip().lower() if provider else "gemini"
        clean_model = model.strip() if model else "default"
        limits = get_model_limits(clean_model)
        
        mode = day_reset_mode or limits["day_reset"]
        day_start_ts, reset_desc, reset_iso = compute_reset_timestamps(mode)
        date_pt = self.get_current_pacific_date()

        # 1. Fail fast if single call exceeds TPM
        tpm_limit = limits["tpm"]
        if tpm_limit is not None and tpm_limit > 0 and estimated_tokens > tpm_limit:
            raise ConfigurationError(
                f"Single LLM request estimated at {estimated_tokens} tokens exceeds total per-minute limit (TPM={tpm_limit}) "
                f"for model '{clean_model}'. Reduce RAG_MAX_CONTEXT_CHARS, CHAT_MAX_CONTEXT_CHARS, or max output tokens."
            )

        now_ts = time.time()
        created_at = datetime.now(timezone.utc).isoformat()

        with self._lock:
            with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                conn.isolation_level = None
                cur = conn.cursor()
                cur.execute("BEGIN IMMEDIATE;")

                # Clean up old events older than 48 hours
                cur.execute("DELETE FROM llm_call_events WHERE ts < ?", (now_ts - 172800.0,))

                # 2. Check Daily limits (RPD and TPD)
                rpd_limit = limits["rpd"]
                tpd_limit = limits["tpd"]

                cur.execute("""
                    SELECT 
                        COUNT(*),
                        COALESCE(SUM(COALESCE(tokens_actual, tokens_reserved)), 0)
                    FROM llm_call_events
                    WHERE provider = ? AND model = ? AND ts >= ?
                      AND status IN ('reserved', 'completed', 'ambiguous')
                """, (clean_provider, clean_model, day_start_ts))
                day_reqs, day_tokens = cur.fetchone()

                if rpd_limit is not None and rpd_limit > 0 and (day_reqs + 1) > rpd_limit:
                    cur.execute("ROLLBACK;")
                    raise QuotaExceededError(
                        f"Daily request quota exceeded for model '{clean_model}': {day_reqs}/{rpd_limit} requests used. {reset_desc}.",
                        reset_time=reset_desc,
                        resume_after_iso=reset_iso
                    )

                if tpd_limit is not None and tpd_limit > 0 and (day_tokens + estimated_tokens) > tpd_limit:
                    cur.execute("ROLLBACK;")
                    raise QuotaExceededError(
                        f"Daily token quota exceeded for model '{clean_model}': {day_tokens}/{tpd_limit} tokens used, required {estimated_tokens}. {reset_desc}.",
                        reset_time=reset_desc,
                        resume_after_iso=reset_iso
                    )

                # Insert reservation event
                cur.execute("""
                    INSERT INTO llm_call_events (ts, provider, model, tokens_reserved, tokens_actual, status, created_at)
                    VALUES (?, ?, ?, ?, NULL, 'reserved', ?)
                """, (now_ts, clean_provider, clean_model, estimated_tokens, created_at))
                event_id = cur.lastrowid

                # Also update legacy table for backwards-compatibility
                cur.execute("""
                    INSERT INTO model_daily_quota (model, date_pt, request_count)
                    VALUES (?, ?, 1)
                    ON CONFLICT(model, date_pt) DO UPDATE SET
                        request_count = model_daily_quota.request_count + 1;
                """, (clean_model, date_pt))

                cur.execute("COMMIT;")
                return event_id

    def complete(self, event_id: int, actual_tokens: int) -> None:
        """Mark reservation completed and reconcile with actual token usage."""
        with self._lock:
            with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                conn.execute("""
                    UPDATE llm_call_events
                    SET tokens_actual = ?, status = 'completed'
                    WHERE id = ?
                """, (max(0, int(actual_tokens)), event_id))
                conn.commit()

    def release(self, event_id: int, reason: str = "local_error") -> None:
        """Release reservation when call definitely consumed zero tokens (e.g. local error)."""
        with self._lock:
            with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                conn.execute("""
                    UPDATE llm_call_events
                    SET tokens_reserved = 0, tokens_actual = 0, status = 'released', error_kind = ?
                    WHERE id = ?
                """, (reason, event_id))
                conn.commit()

    def mark_ambiguous(self, event_id: int, reason: str = "ambiguous_failure") -> None:
        """Mark reservation ambiguous (e.g. timeout, mid-stream drop); keeps reserved tokens."""
        with self._lock:
            with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                conn.execute("""
                    UPDATE llm_call_events
                    SET status = 'ambiguous', error_kind = ?
                    WHERE id = ?
                """, (reason, event_id))
                conn.commit()

    def get_sliding_window_usage(
        self,
        provider: str,
        model: str,
        window_seconds: float = 60.0
    ) -> Tuple[int, int]:
        """Return (requests_count, tokens_count) in the last window_seconds."""
        clean_provider = provider.strip().lower() if provider else "gemini"
        clean_model = model.strip() if model else "default"
        cutoff = time.time() - window_seconds
        with self._lock:
            with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT 
                        COUNT(*),
                        COALESCE(SUM(COALESCE(tokens_actual, tokens_reserved)), 0)
                    FROM llm_call_events
                    WHERE provider = ? AND model = ? AND ts >= ?
                      AND status IN ('reserved', 'completed', 'ambiguous')
                """, (clean_provider, clean_model, cutoff))
                row = cur.fetchone()
                return (row[0], row[1]) if row else (0, 0)

    # =========================================================================
    # Legacy API compatibility for existing tests & callers
    # =========================================================================

    def check_and_increment(self, model: str, rpd_limit: Optional[int] = None) -> int:
        """Atomically check quota and increment request counter (Legacy)."""
        clean_model = model.strip() if model else "default"
        date_pt = self.get_current_pacific_date()
        reset_msg = self.get_reset_time_description("pacific_midnight")

        with self._lock:
            with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                conn.isolation_level = None
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
                    reset_iso = self.get_reset_time_iso("pacific_midnight")
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
                
                # Also record in llm_call_events
                cursor.execute("""
                    INSERT INTO llm_call_events (ts, provider, model, tokens_reserved, tokens_actual, status, created_at)
                    VALUES (?, 'gemini', ?, 0, 0, 'completed', ?)
                """, (time.time(), clean_model, datetime.now(timezone.utc).isoformat()))

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
        """Clear counts and events (for testing)."""
        with self._lock:
            with sqlite3.connect(str(self.db_path), timeout=10.0) as conn:
                if model:
                    conn.execute("DELETE FROM model_daily_quota WHERE model = ?", (model,))
                    conn.execute("DELETE FROM llm_call_events WHERE model = ?", (model,))
                else:
                    conn.execute("DELETE FROM model_daily_quota;")
                    conn.execute("DELETE FROM llm_call_events;")
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


class ReservationContext:
    """Async context manager for LLM call token reservations and lifecycle."""

    def __init__(
        self,
        tracker: DailyModelQuotaTracker,
        event_id: int,
        estimated_tokens: int,
        provider: str,
        model: str
    ):
        self.tracker = tracker
        self.event_id = event_id
        self.estimated_tokens = estimated_tokens
        self.provider = provider
        self.model = model
        self.actual_tokens: Optional[int] = None
        self._finalized = False

    def set_actual_tokens(self, tokens: int) -> None:
        """Set actual token count from provider usage metadata or output estimate."""
        self.actual_tokens = max(0, int(tokens))

    async def complete(self, actual_tokens: Optional[int] = None) -> None:
        """Manually mark completed with actual tokens."""
        if not self._finalized:
            self._finalized = True
            tokens = actual_tokens if actual_tokens is not None else (self.actual_tokens or self.estimated_tokens)
            await asyncio.to_thread(self.tracker.complete, self.event_id, tokens)

    async def release(self, reason: str = "local_error") -> None:
        """Manually release reservation (definitely consumed 0 tokens)."""
        if not self._finalized:
            self._finalized = True
            await asyncio.to_thread(self.tracker.release, self.event_id, reason)

    async def mark_ambiguous(self, reason: str = "ambiguous_failure") -> None:
        """Manually mark reservation ambiguous (timeout, connection drop, etc.)."""
        if not self._finalized:
            self._finalized = True
            await asyncio.to_thread(self.tracker.mark_ambiguous, self.event_id, reason)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._finalized:
            return False

        if exc_val is None:
            tokens = self.actual_tokens if self.actual_tokens is not None else self.estimated_tokens
            await asyncio.to_thread(self.tracker.complete, self.event_id, tokens)
        else:
            # Differentiate definitely zero-token errors from ambiguous ones
            # Zero-token errors: local validation, connection refused before send
            zero_token_indicators = [
                "ConnectionRefusedError",
                "gaierror",
                "ValidationError",
                "ConfigurationError",
                "ConnectError"
            ]
            exc_name = exc_type.__name__ if exc_type else "Exception"
            if any(ind.lower() in exc_name.lower() or ind.lower() in str(exc_val).lower() for ind in zero_token_indicators):
                await asyncio.to_thread(self.tracker.release, self.event_id, exc_name)
            else:
                # Ambiguous: server may have consumed tokens (timeout, cancelled, mid-stream disconnect, 429)
                await asyncio.to_thread(self.tracker.mark_ambiguous, self.event_id, exc_name)

        return False


class AsyncSlidingWindowLimiter:
    """Sliding-window async rate limiter with minute pacing and token reservations."""

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

    async def acquire_reservation(
        self,
        provider: str,
        model: str,
        estimated_tokens: int = 500
    ) -> ReservationContext:
        """Paces by RPM/TPM and acquires a reservation handle in SQLite."""
        tracker = get_daily_quota_tracker()
        limits = get_model_limits(model)
        rpm_limit = limits["rpm"]
        tpm_limit = limits["tpm"]

        # Minute-window pacing
        while True:
            # Check sliding window usage
            used_reqs, used_tokens = await asyncio.to_thread(
                tracker.get_sliding_window_usage, provider, model, 60.0
            )

            req_breached = used_reqs >= rpm_limit
            token_breached = tpm_limit is not None and tpm_limit > 0 and (used_tokens + estimated_tokens) > tpm_limit

            if not req_breached and not token_breached:
                break

            # Need to wait for oldest events to roll off
            await asyncio.sleep(0.2)

        event_id = await asyncio.to_thread(
            tracker.reserve, provider, model, estimated_tokens
        )
        return ReservationContext(
            tracker=tracker,
            event_id=event_id,
            estimated_tokens=estimated_tokens,
            provider=provider,
            model=model
        )

    async def acquire(self, model: Optional[str] = None) -> None:
        """Legacy slot acquisition for existing callers."""
        clean_model = model or (os.environ.get("GEMINI_MODEL", "gemini-default") if self.name == "LLM" else self.name)
        # 1. Enforce persistent daily quota counter before every call/retry
        rpd_limit = self.rpd
        if rpd_limit is None and self.name == "LLM":
            rpd_env = os.environ.get("LLM_RPD")
            if rpd_env and rpd_env.isdigit():
                rpd_limit = int(rpd_env)

        if rpd_limit is not None and rpd_limit > 0:
            tracker = get_daily_quota_tracker()
            await asyncio.to_thread(tracker.check_and_increment, clean_model, rpd_limit)

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
