"""Search caching layer for Research Agent Crew.

Provides a thread-safe, SQLite-backed cache for web search queries to avoid
redundant network calls, preserve search budget, and accelerate iterative research runs.
"""

import os
import re
import time
import json
import sqlite3
import hashlib
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from core.critic_checks import STOP_WORDS
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from core.critic_checks import STOP_WORDS


def normalize_query_for_cache(query: str) -> str:
    """Normalize query by lowercasing, stripping punctuation & stop words, and sorting tokens."""
    cleaned = re.sub(r"[^\w\s-]", " ", query.lower())
    tokens = [w for w in cleaned.split() if w and w not in STOP_WORDS]
    if not tokens:
        tokens = [w for w in cleaned.split() if w]
    tokens.sort()
    return " ".join(tokens)


def make_cache_key(query: str, max_results: int = 5, provider: str = "default") -> str:
    """Generate deterministic hash key for search query + max_results + provider."""
    norm_q = normalize_query_for_cache(query)
    raw = f"{norm_q}::{max_results}::{provider.lower().strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class SearchCache:
    """SQLite-backed persistent cache for search results with TTL and concurrency safety."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or os.environ.get("CACHE_DB_PATH", ".cache/search.db")
        self._lock = threading.Lock()
        self._initialized = False

    def is_enabled(self) -> bool:
        """Check if search cache is enabled via CACHE_ENABLED."""
        val = os.environ.get("CACHE_ENABLED", "true").lower().strip()
        return val not in ("false", "0", "no", "off")

    def _get_connection(self) -> sqlite3.Connection:
        path = Path(self._db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            conn = self._get_connection()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS search_cache (
                        cache_key TEXT PRIMARY KEY,
                        query TEXT,
                        normalized_query TEXT,
                        provider TEXT,
                        max_results INTEGER,
                        response_json TEXT,
                        result_count INTEGER,
                        created_at REAL,
                        expires_at REAL
                    );
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cache_expires 
                    ON search_cache (expires_at);
                """)
                conn.commit()
            finally:
                conn.close()
            self._initialized = True

    def get_sync(self, query: str, max_results: int = 5, provider: str = "default") -> Optional[Dict[str, Any]]:
        """Synchronously retrieve cached search result if present and not expired."""
        if not self.is_enabled():
            return None
        self._init_db()
        key = make_cache_key(query, max_results, provider)
        now = time.time()
        conn = self._get_connection()
        try:
            with self._lock:
                cur = conn.cursor()
                cur.execute(
                    "SELECT response_json, expires_at FROM search_cache WHERE cache_key = ?",
                    (key,)
                )
                row = cur.fetchone()
                if row is None:
                    return None
                expires_at = row["expires_at"]
                if now > expires_at:
                    cur.execute("DELETE FROM search_cache WHERE cache_key = ?", (key,))
                    conn.commit()
                    return None
                return json.loads(row["response_json"])
        except Exception:
            return None
        finally:
            conn.close()

    def set_sync(
        self,
        query: str,
        result_data: Dict[str, Any],
        max_results: int = 5,
        provider: str = "default"
    ) -> None:
        """Synchronously store search result with TTL.
        
        Never caches error or timeout results. Empty results cached at most 10 minutes.
        """
        if not self.is_enabled():
            return
        if not isinstance(result_data, dict):
            return

        status = result_data.get("status")
        # Never cache error, timeout, or rate-limited results
        if status in ("error", "timeout", "rate_limited"):
            return

        results_list = result_data.get("results", [])
        is_empty = status == "no_results" or len(results_list) == 0

        try:
            base_ttl = float(os.environ.get("CACHE_TTL_SECONDS", "21600"))
        except ValueError:
            base_ttl = 21600.0

        # Empty results capped to 10 minutes (600s)
        if is_empty:
            ttl = min(base_ttl, 600.0)
        else:
            ttl = base_ttl

        self._init_db()
        key = make_cache_key(query, max_results, provider)
        norm_q = normalize_query_for_cache(query)
        now = time.time()
        expires_at = now + ttl

        conn = self._get_connection()
        try:
            with self._lock:
                conn.execute("""
                    INSERT OR REPLACE INTO search_cache 
                    (cache_key, query, normalized_query, provider, max_results, response_json, result_count, created_at, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    key,
                    query,
                    norm_q,
                    provider,
                    max_results,
                    json.dumps(result_data),
                    len(results_list),
                    now,
                    expires_at
                ))
                conn.commit()
        except Exception:
            pass
        finally:
            conn.close()

    async def get(self, query: str, max_results: int = 5, provider: str = "default") -> Optional[Dict[str, Any]]:
        """Asynchronously retrieve cached search result via worker thread."""
        return await asyncio.to_thread(self.get_sync, query, max_results, provider)

    async def set(
        self,
        query: str,
        result_data: Dict[str, Any],
        max_results: int = 5,
        provider: str = "default"
    ) -> None:
        """Asynchronously store search result via worker thread."""
        await asyncio.to_thread(self.set_sync, query, result_data, max_results, provider)

    def clear(self) -> None:
        """Clear all entries in the cache."""
        self._init_db()
        try:
            with self._lock:
                with self._get_connection() as conn:
                    conn.execute("DELETE FROM search_cache")
                    conn.commit()
        except Exception:
            pass


_DEFAULT_CACHE: Optional[SearchCache] = None


def set_global_cache(cache: Optional[SearchCache]) -> None:
    """Explicitly override global search cache instance (for testing)."""
    global _DEFAULT_CACHE
    _DEFAULT_CACHE = cache


def get_search_cache() -> SearchCache:
    """Get the active SearchCache instance, tracking CACHE_DB_PATH changes."""
    global _DEFAULT_CACHE
    current_path = os.environ.get("CACHE_DB_PATH", ".cache/search.db")
    if _DEFAULT_CACHE is None or _DEFAULT_CACHE._db_path != current_path:
        _DEFAULT_CACHE = SearchCache(db_path=current_path)
    return _DEFAULT_CACHE
