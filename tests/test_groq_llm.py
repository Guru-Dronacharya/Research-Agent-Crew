"""Unit tests for Groq LLM provider, factory validation, 429 retry-after, and structured outputs."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import BaseModel, Field

from core.llm import (
    get_provider,
    validate_config,
    resolve_model_name,
    ConfigurationError,
    GroqAgentLLM,
    _extract_retry_after,
    _is_transient_error,
    strip_reasoning_tags,
)


class DummyPlan(BaseModel):
    title: str = Field(...)
    steps: list[str] = Field(default_factory=list)


def test_provider_resolution_and_validation(monkeypatch):
    """Verify provider resolution and key/model enforcement."""
    # Gemini mode: requires GEMINI_API_KEY and GEMINI_MODEL
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_groq_key_should_not_be_used")

    with pytest.raises(ConfigurationError) as exc:
        validate_config("gemini")
    assert "GEMINI_API_KEY is required" in str(exc.value)

    # Groq mode: requires GROQ_API_KEY and GROQ_MODEL, ignores missing Gemini keys
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test12345678901234567890")
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    with pytest.raises(ConfigurationError) as exc:
        validate_config("groq")
    assert "GROQ_MODEL is required" in str(exc.value)

    # Valid Groq configuration
    monkeypatch.setenv("GROQ_MODEL", "llama-model")
    validate_config("groq")
    assert get_provider() == "groq"
    assert resolve_model_name("PLANNER_MODEL") == "llama-model"


def test_role_model_override_groq(monkeypatch):
    """Role env vars override global GROQ_MODEL."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_MODEL", "global-groq")
    monkeypatch.setenv("PLANNER_MODEL", "fast-planner")
    monkeypatch.setenv("SYNTH_MODEL", "large-synth")

    assert resolve_model_name("PLANNER_MODEL") == "fast-planner"
    assert resolve_model_name("SYNTH_MODEL") == "large-synth"
    assert resolve_model_name("CRITIC_MODEL") == "global-groq"


def test_extract_retry_after_groq_headers():
    """Verify extraction of retry-after from Groq SDK headers."""
    class MockResp:
        def __init__(self, headers):
            self.headers = headers

    class MockError(Exception):
        def __init__(self, msg, headers=None):
            super().__init__(msg)
            if headers:
                self.response = MockResp(headers)

    # standard retry-after (seconds)
    err1 = MockError("Rate limit", headers={"retry-after": "15.5"})
    assert _extract_retry_after(err1) == 15.5

    # retry-after-ms (milliseconds)
    err2 = MockError("Rate limit", headers={"retry-after-ms": "2500"})
    assert _extract_retry_after(err2) == 2.5

    # x-ratelimit-reset-requests duration format
    err3 = MockError("Rate limit", headers={"x-ratelimit-reset-requests": "1.2s"})
    assert _extract_retry_after(err3) == 1.2

    # x-ratelimit-reset-tokens duration format
    err4 = MockError("Rate limit", headers={"x-ratelimit-reset-tokens": "6m30s"})
    assert _extract_retry_after(err4) == 390.0

    # transient check
    assert _is_transient_error(err1) is True
    assert _is_transient_error(err4) is False  # > 60s is considered daily quota exhaustion


def test_strip_reasoning_tags():
    """Strip <think>...</think> and count reasoning tokens."""
    raw = "<think>Pondering step 1\nStep 2</think>{\n  \"title\": \"Analysis\",\n  \"steps\": [\"step 1\"]\n}"
    clean, think_tokens = strip_reasoning_tags(raw)
    assert "<think>" not in clean
    assert "</think>" not in clean
    assert clean.startswith("{")
    assert think_tokens > 0


@pytest.mark.slow
@pytest.mark.asyncio
async def test_groq_structured_output_repair_retry_once(monkeypatch):
    """Structured output invokes repair prompt on validation failure exactly once."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test12345678901234567890")
    monkeypatch.setenv("GROQ_MODEL", "mock-model")

    groq_llm = GroqAgentLLM(model_name="mock-model")

    call_count = 0
    mock_runnable = MagicMock()

    async def mock_ainvoke(prompt_input):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "raw": MagicMock(content="broken", response_metadata={"finish_reason": "stop"}),
                "parsed": None,
                "parsing_error": "Validation failed: missing field 'title'"
            }
        else:
            return {
                "raw": MagicMock(content='{"title": "Recovered", "steps": []}', response_metadata={"finish_reason": "stop"}),
                "parsed": DummyPlan(title="Recovered", steps=[]),
                "parsing_error": None
            }

    mock_runnable.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_runnable)
    groq_llm.llm = mock_llm

    plan = await groq_llm._invoke_structured(DummyPlan, "Generate plan")
    assert plan.title == "Recovered"
    assert call_count == 2  # Proves retry executed exactly once


@pytest.mark.slow
@pytest.mark.asyncio
async def test_groq_structured_output_length_truncation_retry(monkeypatch):
    """When finish_reason is length, retries with larger budget."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test12345678901234567890")
    monkeypatch.setenv("GROQ_MODEL", "mock-model")

    groq_llm = GroqAgentLLM(model_name="mock-model")

    call_count = 0
    mock_runnable = MagicMock()

    async def mock_ainvoke(prompt_input):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Hit max tokens
            return {
                "raw": MagicMock(content="truncated text", response_metadata={"finish_reason": "length"}),
                "parsed": None,
                "parsing_error": "Incomplete json"
            }
        else:
            # Second attempt with doubled budget succeeds
            return {
                "raw": MagicMock(content='{"title": "Complete", "steps": []}', response_metadata={"finish_reason": "stop"}),
                "parsed": DummyPlan(title="Complete", steps=[]),
                "parsing_error": None
            }

    mock_runnable.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_runnable)
    groq_llm.llm = mock_llm

    plan = await groq_llm._invoke_structured(DummyPlan, "Generate plan")
    assert plan.title == "Complete"
    assert call_count == 2


@pytest.mark.slow
def test_invalid_structured_mode_raises(monkeypatch):
    """Invalid GROQ_STRUCTURED_MODE raises ConfigurationError."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test12345678901234567890")
    monkeypatch.setenv("GROQ_MODEL", "mock-model")
    monkeypatch.setenv("GROQ_STRUCTURED_MODE", "unsupported_mode")

    groq_llm = GroqAgentLLM(model_name="mock-model")
    with pytest.raises(ConfigurationError) as exc:
        groq_llm._get_structured_mode()
    assert "Invalid GROQ_STRUCTURED_MODE" in str(exc.value)
