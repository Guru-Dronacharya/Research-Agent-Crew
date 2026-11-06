"""Unit tests for context auto-sizing and budget warnings."""

import logging
from core.llm.context import (
    get_role_max_tokens,
    derive_safe_context_chars,
    check_context_budget_warnings,
)


def test_role_max_output_tokens(monkeypatch):
    """Verify role token budgets have sensible defaults and synthesis is > 2048."""
    assert get_role_max_tokens("planner") == 2048
    assert get_role_max_tokens("critic") == 2048
    assert get_role_max_tokens("synth") == 4096  # Synthesis is larger than 2048
    assert get_role_max_tokens("chat") == 2048

    # Overrides work
    monkeypatch.setenv("SYNTH_MAX_TOKENS", "6000")
    assert get_role_max_tokens("synth") == 6000


def test_derive_safe_context_chars():
    """Derive safe context chars budget from TPM."""
    # TPM = 6000, output budget = 2048
    # Available input tokens = 3952 -> * 4 chars * 0.8 safety margin = 12646 chars
    safe = derive_safe_context_chars(tpm_limit=6000, max_output_tokens=2048)
    assert 12000 <= safe <= 13000

    # Unlimited TPM returns default 24000
    assert derive_safe_context_chars(tpm_limit=None) == 24000
    assert derive_safe_context_chars(tpm_limit=0) == 24000


def test_context_budget_warning_logged(monkeypatch, caplog):
    """When configured context chars exceed TPM budget, log a warning."""
    monkeypatch.setenv("LLM_TPM", "4000")
    monkeypatch.setenv("RAG_MAX_CONTEXT_CHARS", "24000")

    with caplog.at_level(logging.WARNING):
        check_context_budget_warnings()

    assert any("exceeds safe budget" in record.message for record in caplog.records)
