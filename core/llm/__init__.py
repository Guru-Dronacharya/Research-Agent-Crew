"""LLM Provider abstraction package supporting Google Gemini and Groq."""

from core.llm.base import (
    BaseAgentLLM,
    _extract_retry_after,
    _is_transient_error,
    llm_retry,
    wait_with_retry_after,
    estimate_tokens,
    strip_reasoning_tags,
)
from core.ratelimit import ConfigurationError, QuotaExceededError
from core.llm.gemini import GeminiAgentLLM
from core.llm.groq import GroqAgentLLM
from core.llm.context import (
    get_role_max_tokens,
    derive_safe_context_chars,
    check_context_budget_warnings,
)
from core.llm.factory import (
    get_provider,
    validate_config,
    resolve_model_name,
    get_planner_llm,
    get_synth_llm,
    get_critic_llm,
    get_agent_llm,
    set_global_llm,
    _GLOBAL_LLM_OVERRIDE,
)

__all__ = [
    "BaseAgentLLM",
    "GeminiAgentLLM",
    "GroqAgentLLM",
    "ConfigurationError",
    "QuotaExceededError",
    "_extract_retry_after",
    "_is_transient_error",
    "llm_retry",
    "wait_with_retry_after",
    "estimate_tokens",
    "strip_reasoning_tags",
    "get_role_max_tokens",
    "derive_safe_context_chars",
    "check_context_budget_warnings",
    "get_provider",
    "validate_config",
    "resolve_model_name",
    "get_planner_llm",
    "get_synth_llm",
    "get_critic_llm",
    "get_agent_llm",
    "set_global_llm",
    "_GLOBAL_LLM_OVERRIDE",
]
