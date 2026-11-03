"""Async Rate Limiter and Quota Management for Research Agent Crew.

Provides token-bucket / sliding-window rate limiting for LLM and Search requests,
enforces daily quotas (LLM_RPD), and raises QuotaExceededError when daily caps are reached.
"""

import os
import time
import asyncio
from typing import Optional
from collections import deque


class QuotaExceededError(RuntimeError):
    """Raised when daily request quota (LLM_RPD) is exceeded."""
    pass


class AsyncSlidingWindowLimiter:
    """Sliding-window async rate limiter with minute and daily caps."""

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
        self._daily_window: deque[float] = deque()
        self._lock = asyncio.Lock()

    def get_rpm(self) -> float:
        return self.rpm

    def set_rpm(self, rpm: float) -> None:
        self.rpm = max(1.0, float(rpm))

    def set_rpd(self, rpd: Optional[int]) -> None:
        self.rpd = int(rpd) if rpd is not None and int(rpd) > 0 else None

    async def acquire(self) -> None:
        """Acquire a slot, pausing if RPM is reached or raising QuotaExceededError if RPD is exceeded."""
        while True:
            async with self._lock:
                now = time.monotonic()
                wall_now = time.time()

                # Clean minute window (last 60s)
                cutoff_minute = now - 60.0
                while self._minute_window and self._minute_window[0] < cutoff_minute:
                    self._minute_window.popleft()

                # Clean daily window (last 86400s)
                cutoff_day = wall_now - 86400.0
                while self._daily_window and self._daily_window[0] < cutoff_day:
                    self._daily_window.popleft()

                # Check daily quota
                if self.rpd is not None and len(self._daily_window) >= self.rpd:
                    raise QuotaExceededError(
                        f"Daily quota exceeded for {self.name}: {len(self._daily_window)}/{self.rpd} requests in last 24h."
                    )

                # Check minute rate limit
                if len(self._minute_window) < self.rpm:
                    self._minute_window.append(now)
                    self._daily_window.append(wall_now)
                    return

                # Wait until oldest in minute window expires
                sleep_time = max(0.01, (self._minute_window[0] + 60.0) - now)

            await asyncio.sleep(sleep_time)

    def reset(self) -> None:
        """Clear limiter state (useful for test suites)."""
        self._minute_window.clear()
        self._daily_window.clear()


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
    """Return singleton Search rate limiter configured via SEARCH_RPM."""
    global _SEARCH_LIMITER
    if _SEARCH_LIMITER is None:
        try:
            rpm = float(os.environ.get("SEARCH_RPM", "30"))
        except ValueError:
            rpm = 30.0
        _SEARCH_LIMITER = AsyncSlidingWindowLimiter(rpm=rpm, name="Search")
    return _SEARCH_LIMITER


def reset_rate_limiters() -> None:
    """Reset rate limiter singletons."""
    global _LLM_LIMITER, _SEARCH_LIMITER
    if _LLM_LIMITER is not None:
        _LLM_LIMITER.reset()
    if _SEARCH_LIMITER is not None:
        _SEARCH_LIMITER.reset()
    _LLM_LIMITER = None
    _SEARCH_LIMITER = None
