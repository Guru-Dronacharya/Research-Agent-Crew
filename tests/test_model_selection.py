"""Unit tests for per-step model selection and configuration enforcement."""

import pytest
from core.llm import (
    resolve_model_name,
    get_planner_llm,
    get_synth_llm,
    get_critic_llm,
    ConfigurationError,
    GeminiAgentLLM,
)


def test_resolve_model_fallback_to_gemini_model():
    """When role model is unset, falls back to GEMINI_MODEL."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GEMINI_MODEL", "gemini-default-test")
        mp.delenv("PLANNER_MODEL", raising=False)
        mp.delenv("SYNTH_MODEL", raising=False)
        mp.delenv("CRITIC_MODEL", raising=False)

        assert resolve_model_name("PLANNER_MODEL") == "gemini-default-test"
        assert resolve_model_name("SYNTH_MODEL") == "gemini-default-test"
        assert resolve_model_name("CRITIC_MODEL") == "gemini-default-test"


def test_resolve_model_role_override():
    """Role-specific env vars override GEMINI_MODEL."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GEMINI_MODEL", "gemini-default-test")
        mp.setenv("PLANNER_MODEL", "gemini-planner-flash")
        mp.setenv("SYNTH_MODEL", "gemini-synth-pro")
        mp.setenv("CRITIC_MODEL", "gemini-critic-flash")

        assert resolve_model_name("PLANNER_MODEL") == "gemini-planner-flash"
        assert resolve_model_name("SYNTH_MODEL") == "gemini-synth-pro"
        assert resolve_model_name("CRITIC_MODEL") == "gemini-critic-flash"


@pytest.mark.critical
def test_resolve_model_missing_gemini_model_raises_config_error():
    """Missing GEMINI_MODEL and role model raises ConfigurationError."""
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("GEMINI_MODEL", raising=False)
        mp.delenv("PLANNER_MODEL", raising=False)

        with pytest.raises(ConfigurationError):
            resolve_model_name("PLANNER_MODEL")


def test_explicit_runtime_override():
    """Explicit runtime model argument takes top priority."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GEMINI_MODEL", "gemini-default")
        mp.setenv("PLANNER_MODEL", "gemini-planner")

        assert resolve_model_name("PLANNER_MODEL", explicit_override="gemini-custom") == "gemini-custom"


@pytest.mark.slow
def test_get_llm_factories_instantiate_expected_model():
    """Factories instantiate GeminiAgentLLM with corresponding model names."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GEMINI_API_KEY", "fake_key_for_testing")
        mp.setenv("GEMINI_MODEL", "gemini-global")
        mp.setenv("PLANNER_MODEL", "gemini-fast-planner")
        mp.setenv("SYNTH_MODEL", "gemini-smart-synth")
        mp.setenv("CRITIC_MODEL", "gemini-fast-critic")

        # Temporarily clear global stub to test real factory construction
        from core.llm import set_global_llm
        set_global_llm(None)

        planner_llm = get_planner_llm()
        assert isinstance(planner_llm, GeminiAgentLLM)
        assert planner_llm.model_name == "gemini-fast-planner"

        synth_llm = get_synth_llm()
        assert isinstance(synth_llm, GeminiAgentLLM)
        assert synth_llm.model_name == "gemini-smart-synth"

        critic_llm = get_critic_llm()
        assert isinstance(critic_llm, GeminiAgentLLM)
        assert critic_llm.model_name == "gemini-fast-critic"
        assert critic_llm.temperature == 0.0
