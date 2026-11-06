"""Unit tests for Groq and Google API key redaction in events and payloads."""

from core.events import sanitize_and_cap_payload, redact_sensitive_strings


def test_redact_groq_api_keys():
    """Groq API keys (gsk_...) are redacted from text and payloads."""
    raw_text = "Sending request using auth key: gsk_1234567890abcdef1234567890abcdef in header"
    redacted = redact_sensitive_strings(raw_text)
    assert "gsk_" not in redacted
    assert "[REDACTED]" in redacted

    payload = {
        "query": "What is AI?",
        "debug_info": "Used key gsk_abcdef1234567890abcdef1234567890 for auth",
        "nested": {
            "api_key": "raw_secret_value",
            "message": "Key gsk_999999999988888888887777777777 attached"
        }
    }
    clean = sanitize_and_cap_payload(payload)
    assert "gsk_" not in str(clean)
    assert clean["debug_info"] == "Used key [REDACTED] for auth"
    assert clean["nested"]["api_key"] == "[REDACTED]"
    assert clean["nested"]["message"] == "Key [REDACTED] attached"


def test_redact_google_api_keys():
    """Google Gemini API keys (AIza...) are redacted from text and payloads."""
    raw_text = "Connection established with AIzaSyAbcdef1234567890abcdef12345678901 endpoint"
    redacted = redact_sensitive_strings(raw_text)
    assert "AIza" not in redacted
    assert "[REDACTED]" in redacted
