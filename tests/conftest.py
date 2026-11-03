"""Pytest configuration and global fixtures.

Automatically injects the StubLLM test double during test execution
so all graph nodes and units run deterministically without external API calls.
"""

import pytest
from core.llm import set_global_llm
from mcp_tools.client import reset_global_tools
from core.cache import set_global_cache
from core.ratelimit import reset_rate_limiters
from tests.stubs import StubLLM


@pytest.fixture(autouse=True)
def configure_test_environment(monkeypatch):
    """Configure fast rate limits, disable cache by default in tests, inject StubLLM, and reset global singletons."""
    monkeypatch.setenv("LLM_RPM", "6000")
    monkeypatch.setenv("SEARCH_RPM", "6000")
    monkeypatch.setenv("CACHE_ENABLED", "false")
    reset_rate_limiters()
    stub = StubLLM()
    set_global_llm(stub)
    yield stub
    set_global_llm(None)
    reset_global_tools()
    set_global_cache(None)
    reset_rate_limiters()
