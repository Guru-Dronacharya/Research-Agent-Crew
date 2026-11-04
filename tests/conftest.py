"""Pytest configuration and global fixtures.

Automatically injects the StubLLM test double during test execution,
stubs page-fetching in the executor, installs an offline network guard blocking
outbound calls to non-loopback addresses, and resets rate limits and singletons.
"""

import socket
import ipaddress
import pytest
from core.llm import set_global_llm
from mcp_tools.client import reset_global_tools
from core.cache import set_global_cache
from core.ratelimit import reset_rate_limiters
from core.rag import reset_embedder_cache
from tests.stubs import StubLLM


def _is_loopback_host(host: str) -> bool:
    """Return True if host is None, empty, localhost, or a valid loopback IP address."""
    if not host:
        return True
    host_clean = str(host).lower().strip("[]").strip()
    if host_clean in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "::"):
        return True
    try:
        ip = ipaddress.ip_address(host_clean)
        return ip.is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def network_guard(monkeypatch):
    """Autouse network guard that blocks any outbound connection or DNS resolution to non-loopback addresses."""
    orig_getaddrinfo = socket.getaddrinfo
    orig_connect = socket.socket.connect
    orig_connect_ex = socket.socket.connect_ex

    def _guarded_getaddrinfo(host, port, *args, **kwargs):
        if host is not None and not _is_loopback_host(str(host)):
            raise RuntimeError(
                f"Network access blocked by test network guard: DNS resolution for non-loopback host '{host}' is forbidden during offline tests."
            )
        return orig_getaddrinfo(host, port, *args, **kwargs)

    def _guarded_connect(self, address):
        host = None
        if isinstance(address, tuple) and len(address) > 0:
            host = str(address[0])
        elif isinstance(address, str):
            host = address

        if host is not None and not _is_loopback_host(host):
            raise RuntimeError(
                f"Network access blocked by test network guard: Outbound connection to non-loopback address '{host}' is forbidden during offline tests."
            )
        return orig_connect(self, address)

    def _guarded_connect_ex(self, address):
        host = None
        if isinstance(address, tuple) and len(address) > 0:
            host = str(address[0])
        elif isinstance(address, str):
            host = address

        if host is not None and not _is_loopback_host(host):
            raise RuntimeError(
                f"Network access blocked by test network guard: Outbound connection to non-loopback address '{host}' is forbidden during offline tests."
            )
        return orig_connect_ex(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", _guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _guarded_connect_ex)


@pytest.fixture(autouse=True)
def stub_executor_page_fetch(monkeypatch):
    """Monkeypatch the page-fetch function used by the executor with a deterministic stub returning snippet-only pages."""
    from core.rag.fetch import FetchedPage

    async def _stub_fetch_and_extract_page(url: str, title: str = "", snippet: str = "", **kwargs) -> FetchedPage:
        return FetchedPage(
            url=url,
            title=title,
            text=snippet.strip(),
            snippet_only=True,
            status="snippet_fallback"
        )

    monkeypatch.setattr("core.nodes.executor.fetch_and_extract_page", _stub_fetch_and_extract_page)


@pytest.fixture(autouse=True)
def configure_test_environment(monkeypatch):
    """Configure fast rate limits, disable cache by default in tests, inject StubLLM, and reset global singletons."""
    monkeypatch.setenv("LLM_RPM", "6000")
    monkeypatch.setenv("SEARCH_RPM", "6000")
    monkeypatch.setenv("CACHE_ENABLED", "false")
    monkeypatch.setenv("QUOTA_DB_PATH", ".cache/test_quota.db")
    monkeypatch.setenv("USE_FAKE_EMBEDDER", "1")
    monkeypatch.setenv("EMBEDDING_MODEL", "fake-deterministic-384")
    reset_rate_limiters()
    reset_embedder_cache()
    stub = StubLLM()
    set_global_llm(stub)
    yield stub
    set_global_llm(None)
    reset_global_tools()
    set_global_cache(None)
    reset_rate_limiters()
    reset_embedder_cache()
