"""Cooperative cancellation coordinator for Research Agent Crew.

Provides dual-layer cooperative cancellation tracking across in-process memory
and persistent database state (cancel_requested flag on runs).
"""

from typing import Set, Optional
from core.db.repositories import RunRepo

_IN_PROCESS_CANCELLED_RUNS: Set[str] = set()


class RunCancelledError(Exception):
    """Raised when a run has been cooperatively cancelled by user."""
    pass


def request_cancellation_in_process(run_id: str) -> None:
    """Register in-process cancellation fast-path flag."""
    if run_id:
        _IN_PROCESS_CANCELLED_RUNS.add(str(run_id))


def clear_cancellation_in_process(run_id: str) -> None:
    """Clear in-process cancellation flag."""
    _IN_PROCESS_CANCELLED_RUNS.discard(str(run_id))


async def is_cancellation_requested(run_id: Optional[str], db_path: Optional[str] = None) -> bool:
    """Check if cancellation was requested for run_id via fast path or DB flag."""
    if not run_id:
        return False

    if str(run_id) in _IN_PROCESS_CANCELLED_RUNS:
        return True

    try:
        repo = RunRepo(db_path=db_path)
        is_requested = await repo.is_cancel_requested(str(run_id))
        if is_requested:
            _IN_PROCESS_CANCELLED_RUNS.add(str(run_id))
            return True
    except Exception:
        pass

    return False
