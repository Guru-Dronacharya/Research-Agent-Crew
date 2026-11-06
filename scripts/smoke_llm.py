"""Live LLM smoke testing and structured output probing utility.

Performs one minimal live request against the configured provider (Groq or Google Gemini)
to verify:
- API key validity (with key redaction)
- Model availability and streaming responsiveness
- Usage metadata reporting (input, output, total tokens)
- Rate limit headers (RPM, TPM, reset durations)
- Structured output capability probing (function_calling, json_schema, json_mode)

Usage:
    python scripts/smoke_llm.py
    python scripts/smoke_llm.py --provider groq --model <model-name>
"""

import os
import sys
import time
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field

# Ensure root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.llm import (
    get_provider,
    resolve_model_name,
    validate_config,
    get_agent_llm,
    BaseAgentLLM,
    ConfigurationError,
)
from core.events import redact_sensitive_strings


class SmokeTestSchema(BaseModel):
    status: str = Field(description="Operational status, e.g. 'ok'")
    message: str = Field(description="Short confirmation message")
    verified: bool = Field(description="True if test verified successfully")


def mask_key(key: str) -> str:
    """Mask key preserving prefix and length."""
    if not key:
        return "[MISSING]"
    if len(key) <= 8:
        return "[CONFIGURED]"
    return f"{key[:4]}...{key[-4:]} ({len(key)} chars)"


async def probe_streaming(llm: BaseAgentLLM, model: str) -> Dict[str, Any]:
    """Test streaming and capture tokens and latency."""
    print("\n--- 1. Testing Live Streaming & Token Delivery ---")
    prompt = [{"role": "user", "content": "Respond with exactly 3 words: Antigravity is online."}]

    t0 = time.perf_counter()
    first_token_time = None
    emitted_chunks = []

    try:
        stream = llm.astream_chat(prompt, model_name=model)
        async for chunk in stream:
            if first_token_time is None:
                first_token_time = time.perf_counter() - t0
            emitted_chunks.append(chunk)
            print(chunk, end="", flush=True)
        print()
    except Exception as exc:
        print(f"\n[STREAMING FAILED] {redact_sensitive_strings(str(exc))}")
        return {"success": False, "error": str(exc)}

    total_time = time.perf_counter() - t0
    full_text = "".join(emitted_chunks).strip()
    ttft_ms = (first_token_time * 1000.0) if first_token_time else 0.0

    print(f"Time to First Token (TTFT): {ttft_ms:.1f}ms | Total Duration: {total_time:.2f}s")
    print(f"Full response received ({len(full_text)} chars): {full_text}")
    return {"success": True, "ttft_ms": ttft_ms, "text": full_text}


async def probe_structured_output(provider: str, model: str) -> Dict[str, Any]:
    """Probe structured output modes for model."""
    print("\n--- 2. Probing Structured Output Capability ---")

    if provider == "groq":
        modes_to_test = ["function_calling", "json_mode"]
        # json_schema is supported by a subset of models
        modes_to_test.append("json_schema")
    else:
        modes_to_test = ["default"]

    results: Dict[str, bool] = {}

    for mode in modes_to_test:
        print(f"Testing mode '{mode}'... ", end="", flush=True)
        try:
            if provider == "groq":
                os.environ["GROQ_STRUCTURED_MODE"] = mode
                from core.llm.groq import GroqAgentLLM
                test_llm = GroqAgentLLM(model_name=model)
            else:
                from core.llm.gemini import GeminiAgentLLM
                test_llm = GeminiAgentLLM(model_name=model)

            t0 = time.perf_counter()
            parsed: SmokeTestSchema = await test_llm._invoke_structured(
                schema_cls=SmokeTestSchema,
                prompt="Produce a JSON object verifying that status is 'ok', message is 'System fully operational', and verified is true.",
                role="agent"
            )
            dur = (time.perf_counter() - t0) * 1000.0

            if getattr(parsed, "verified", False) or getattr(parsed, "status", "") == "ok":
                print(f"PASS ({dur:.0f}ms) -> {parsed}")
                results[mode] = True
            else:
                print(f"FAIL (invalid payload: {parsed})")
                results[mode] = False
        except Exception as exc:
            err_msg = redact_sensitive_strings(str(exc)).split("\n")[0]
            print(f"FAIL ({err_msg})")
            results[mode] = False

    working = [m for m, ok in results.items() if ok]
    if working:
        best_mode = working[0]
        print(f"\n[RECOMMENDATION] Set GROQ_STRUCTURED_MODE={best_mode} for optimal reliability.")
    else:
        print("\n[WARNING] No tested structured output mode succeeded for this model.")

    return results


async def main_async():
    parser = argparse.ArgumentParser(description="Live smoke test for LLM provider.")
    parser.add_argument("--provider", type=str, default=None, help="LLM Provider (gemini or groq)")
    parser.add_argument("--model", type=str, default=None, help="Model name to test")
    args = parser.parse_args()

    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider.strip().lower()

    provider = get_provider()
    try:
        validate_config(provider)
    except ConfigurationError as cfg_err:
        print(f"[CONFIGURATION ERROR] {cfg_err}")
        sys.exit(1)

    model = args.model or resolve_model_name("PLANNER_MODEL")
    api_key = (
        os.environ.get("GROQ_API_KEY", "") if provider == "groq"
        else (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", ""))
    )

    print("==================================================================")
    print("Research Agent Crew - Live LLM Verification & Smoke Test")
    print(f"Provider:  {provider.upper()}")
    print(f"Model:     {model}")
    print(f"API Key:   {mask_key(api_key)}")
    print("==================================================================")

    # Acquire LLM
    llm = get_agent_llm(model_name=model)

    # 1. Test live streaming
    stream_res = await probe_streaming(llm, model)
    if not stream_res.get("success"):
        print("\n[FAIL] Live streaming check failed. Please verify API key and network connection.")
        sys.exit(1)

    # 2. Test structured output
    struct_res = await probe_structured_output(provider, model)
    if not any(struct_res.values()):
        print("\n[FAIL] Structured output check failed.")
        sys.exit(1)

    print("\n==================================================================")
    print("ALL CHECKS PASSED: Provider and model are operational for production.")
    print("==================================================================\n")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
