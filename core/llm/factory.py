"""LLM Factory and configuration validator supporting Gemini and Groq."""

import os
from typing import Optional

from core.ratelimit import ConfigurationError
from core.llm.base import BaseAgentLLM
from core.llm.gemini import GeminiAgentLLM
from core.llm.groq import GroqAgentLLM

# Global injectable override for test suites
_GLOBAL_LLM_OVERRIDE: Optional[BaseAgentLLM] = None


def set_global_llm(llm: Optional[BaseAgentLLM]) -> None:
    """Inject an LLM instance (such as StubLLM in test suites)."""
    global _GLOBAL_LLM_OVERRIDE
    _GLOBAL_LLM_OVERRIDE = llm


def get_provider() -> str:
    """Resolve active LLM provider (gemini or groq)."""
    env_provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if env_provider:
        if env_provider not in ("gemini", "groq"):
            raise ConfigurationError(
                f"Unsupported LLM_PROVIDER '{env_provider}'. Supported options: gemini, groq."
            )
        return env_provider

    # Heuristic fallback if LLM_PROVIDER unset
    if os.environ.get("GROQ_API_KEY") and not (
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    ):
        return "groq"
    return "gemini"


def validate_config(provider: Optional[str] = None) -> None:
    """Startup configuration validator: requires ONLY the selected provider's key and model."""
    target_provider = provider or get_provider()

    if target_provider == "groq":
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise ConfigurationError(
                "GROQ_API_KEY is required when LLM_PROVIDER=groq. Please set GROQ_API_KEY in your environment or .env file."
            )
        model = os.environ.get("GROQ_MODEL", "").strip()
        if not model:
            raise ConfigurationError(
                "GROQ_MODEL is required when LLM_PROVIDER=groq. Please specify GROQ_MODEL in your environment or .env file."
            )

    elif target_provider == "gemini":
        key = (
            os.environ.get("GEMINI_API_KEY", "").strip()
            or os.environ.get("GOOGLE_API_KEY", "").strip()
        )
        if not key:
            raise ConfigurationError(
                "GEMINI_API_KEY is required. Please set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or .env file."
            )
        model = os.environ.get("GEMINI_MODEL", "").strip()
        if not model:
            raise ConfigurationError(
                "GEMINI_MODEL environment variable is not set. Please specify a valid Gemini model in your .env file."
            )
    else:
        raise ConfigurationError(f"Unknown LLM provider '{target_provider}'.")


def resolve_model_name(role_env_var: str, explicit_override: Optional[str] = None) -> str:
    """Resolve model name following priority:
    1. Explicit runtime override (if provided)
    2. Role-specific env var (PLANNER_MODEL, SYNTH_MODEL, CRITIC_MODEL, CHAT_MODEL)
    3. Provider default (GROQ_MODEL if groq, GEMINI_MODEL if gemini)
    """
    if explicit_override and explicit_override.strip():
        return explicit_override.strip()

    role_model = os.environ.get(role_env_var, "").strip()
    if role_model:
        return role_model

    provider = get_provider()
    if provider == "groq":
        groq_model = os.environ.get("GROQ_MODEL", "").strip()
        if not groq_model:
            raise ConfigurationError(
                f"GROQ_MODEL environment variable is not set. Please set GROQ_MODEL (or {role_env_var}) in your .env file."
            )
        return groq_model
    else:
        gemini_model = os.environ.get("GEMINI_MODEL", "").strip()
        if not gemini_model:
            raise ConfigurationError(
                f"GEMINI_MODEL environment variable is not set. Please set GEMINI_MODEL (or {role_env_var}) in your .env file."
            )
        return gemini_model


def _create_llm_for_role(role_env: str, temperature: float, explicit_model: Optional[str] = None) -> BaseAgentLLM:
    """Internal helper to instantiate the appropriate provider LLM."""
    if _GLOBAL_LLM_OVERRIDE is not None:
        return _GLOBAL_LLM_OVERRIDE

    provider = get_provider()
    target_model = resolve_model_name(role_env, explicit_model)

    if provider == "groq":
        return GroqAgentLLM(model_name=target_model, temperature=temperature)
    else:
        return GeminiAgentLLM(model_name=target_model, temperature=temperature)


def get_planner_llm(model_name: Optional[str] = None) -> BaseAgentLLM:
    """Acquire Planner agent LLM (PLANNER_MODEL or provider global model)."""
    return _create_llm_for_role("PLANNER_MODEL", temperature=0.2, explicit_model=model_name)


def get_synth_llm(model_name: Optional[str] = None) -> BaseAgentLLM:
    """Acquire Synthesizer agent LLM (SYNTH_MODEL or provider global model)."""
    return _create_llm_for_role("SYNTH_MODEL", temperature=0.2, explicit_model=model_name)


def get_critic_llm(model_name: Optional[str] = None) -> BaseAgentLLM:
    """Acquire Critic LLM judge (CRITIC_MODEL or provider global model, temperature 0.0)."""
    return _create_llm_for_role("CRITIC_MODEL", temperature=0.0, explicit_model=model_name)


def get_agent_llm(model_name: Optional[str] = None) -> BaseAgentLLM:
    """Acquire primary agent LLM."""
    if _GLOBAL_LLM_OVERRIDE is not None:
        return _GLOBAL_LLM_OVERRIDE
    provider = get_provider()
    global_env = "GROQ_MODEL" if provider == "groq" else "GEMINI_MODEL"
    return _create_llm_for_role(global_env, temperature=0.2, explicit_model=model_name)
