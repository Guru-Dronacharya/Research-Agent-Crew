"""Context auto-sizing and role-based token budget derivation."""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_role_max_tokens(role: str) -> int:
    """Return configured max output tokens for an agent role.

    Synthesis requires a larger budget than standard 2048 to prevent truncation.
    """
    clean_role = (role or "").strip().lower()
    if clean_role == "planner":
        default = 2048
        env_var = "PLANNER_MAX_TOKENS"
    elif clean_role == "critic":
        default = 2048
        env_var = "CRITIC_MAX_TOKENS"
    elif clean_role in ("synth", "synthesizer"):
        default = 4096  # Larger than 2048 as specified
        env_var = "SYNTH_MAX_TOKENS"
    elif clean_role == "chat":
        default = 2048
        env_var = "CHAT_MAX_TOKENS"
    else:
        default = 2048
        env_var = "LLM_MAX_OUTPUT_TOKENS"

    val = os.environ.get(env_var)
    if val and val.isdigit():
        return max(512, int(val))
    return default


def derive_safe_context_chars(
    tpm_limit: Optional[int],
    max_output_tokens: int = 2048,
    chars_per_token: int = 4,
    safety_margin: float = 0.8
) -> int:
    """Derive safe context characters budget from per-minute token limit (LLM_TPM).

    Formula:
        safe_chars = (tpm_limit - max_output_tokens) * chars_per_token * safety_margin
    """
    if tpm_limit is None or tpm_limit <= 0:
        return 24000

    available_input_tokens = max(500, tpm_limit - max_output_tokens)
    safe_chars = int(available_input_tokens * chars_per_token * safety_margin)
    return max(1000, safe_chars)


def check_context_budget_warnings(model_name: Optional[str] = None) -> None:
    """Validate configured context sizes against LLM_TPM and log warnings if tight."""
    from core.ratelimit import get_model_limits

    limits = get_model_limits(model_name or "default")
    tpm = limits.get("tpm")
    if not tpm or tpm <= 0:
        return

    # Check RAG context
    synth_tokens = get_role_max_tokens("synth")
    safe_rag_chars = derive_safe_context_chars(tpm, max_output_tokens=synth_tokens)
    try:
        configured_rag_chars = int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "24000"))
    except ValueError:
        configured_rag_chars = 24000

    if configured_rag_chars > safe_rag_chars:
        logger.warning(
            f"Configured RAG_MAX_CONTEXT_CHARS ({configured_rag_chars}) exceeds safe budget ({safe_rag_chars}) "
            f"for model TPM limit of {tpm}. Single synthesis calls may exceed TPM. "
            f"Recommended: set RAG_MAX_CONTEXT_CHARS={safe_rag_chars}."
        )

    # Check Chat context
    chat_tokens = get_role_max_tokens("chat")
    safe_chat_chars = derive_safe_context_chars(tpm, max_output_tokens=chat_tokens)
    try:
        configured_chat_chars = int(os.environ.get("CHAT_MAX_CONTEXT_CHARS", "8000"))
    except ValueError:
        configured_chat_chars = 8000

    if configured_chat_chars > safe_chat_chars:
        logger.warning(
            f"Configured CHAT_MAX_CONTEXT_CHARS ({configured_chat_chars}) exceeds safe budget ({safe_chat_chars}) "
            f"for model TPM limit of {tpm}. Single chat requests may exceed TPM. "
            f"Recommended: set CHAT_MAX_CONTEXT_CHARS={safe_chat_chars}."
        )
