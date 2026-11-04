"""Tests for LLM factory fail-loud behavior, ConfigurationError, and environment configuration."""

import pytest
from core.llm import get_agent_llm, get_critic_llm, set_global_llm, ConfigurationError


def test_get_agent_llm_fails_loudly_without_gemini_model(monkeypatch):
    """Verify get_agent_llm raises ConfigurationError when GEMINI_MODEL is unset."""
    set_global_llm(None)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")

    with pytest.raises(ConfigurationError) as exc_info:
        get_agent_llm()

    assert "GEMINI_MODEL" in str(exc_info.value)


def test_get_agent_llm_fails_loudly_without_gemini_key(monkeypatch):
    """Verify get_agent_llm raises ConfigurationError when GEMINI_API_KEY is missing."""
    set_global_llm(None)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ConfigurationError) as exc_info:
        get_agent_llm()

    assert "GEMINI_API_KEY is required" in str(exc_info.value)


@pytest.mark.slow
def test_critic_model_defaults_to_gemini_model(monkeypatch):
    """Verify CRITIC_MODEL defaults to GEMINI_MODEL when unset."""
    set_global_llm(None)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")
    monkeypatch.delenv("CRITIC_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")

    critic_llm = get_critic_llm()
    assert critic_llm.model_name == "gemini-2.0-flash"
    assert critic_llm.temperature == 0.0
