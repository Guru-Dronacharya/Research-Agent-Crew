# Research Agent Crew

Production-grade autonomous multi-agent research crew orchestrated via **LangGraph**, **Google Gemini** / **Groq**, and **Model Context Protocol (MCP)** tools.

## Architecture

The system executes a cyclical research workflow:

```
[ User Research Task ]
          │
          ▼
   ┌──────────────┐
   │ Planner Node │ ◄────────────────────────────────────────┐
   └──────┬───────┘                                          │
          │ Subtasks & Queries                               │ Revision Directives
          ▼                                                  │ (Coverage / Evidence Gaps)
   ┌──────────────┐      MCP Client (JSON-RPC)               │
   │Executor Node │ ◄─────────────────────────► [ Search MCP Server ]
   └──────┬───────┘
          │ Subtask Findings & Sources
          ▼
   ┌─────────────────┐
   │Synthesizer Node │
   └──────┬──────────┘
          │ Draft Markdown Report
          ▼
   ┌──────────────┐
   │ Critic Node  │ ────► [ PASS or Max Retries ] ───► ┌───────────────┐
   └──────────────┘                                    │Finalizer Node │ ──► [ Filesystem MCP ]
          │                                            └───────────────┘
          └──────────────► [ FAIL & retries < max ] ──────────┘
```

1. **Planner**: Breaks master queries into focused subtasks and formulates targeted search queries. During revision loops, receives critic audit metrics and creates subtasks specifically targeting missing evidence.
2. **Executor**: Invokes Model Context Protocol (MCP) search tools asynchronously in separate threads, enforces per-subtask budget allocations, and flags evidence gaps when sources cannot be retrieved.
3. **Synthesizer**: Compiles retrieved subtask findings into structured research reports. Grounded strictly in retrieved evidence, with verifiable citations and an explicit Open Gaps section.
4. **Critic**: Evaluates reports using hybrid evaluation:
   - **Deterministic Audit**: Citation validity (URL verification against retrieved sources), keyword-based source relevance, evidence gap rate, and budget invariant verification.
   - **LLM Judge**: Evaluates Coverage, Evidence, and Coherence with hard scoring caps tied to deterministic metrics.
5. **Finalizer**: Enriches reports with audit scorecards, generates JSON telemetry summaries, and persists artifacts via the Filesystem MCP server.

---

## Prerequisites & Configuration

The application supports both **Google Gemini** and **Groq** via a unified provider abstraction (`LLM_PROVIDER=gemini|groq`).

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

### Provider Configuration

#### Option A: Groq (Free Tier / Production)
Set in `.env`:
```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=your_chosen_model_name
GROQ_STRUCTURED_MODE=function_calling
```
- **Structured Output Modes**: `function_calling` (default for tool-capable models), `json_schema`, or `json_mode`. No model-name assumptions are hardcoded; mode is selected entirely via `GROQ_STRUCTURED_MODE`.
- Run `python scripts/smoke_llm.py` before running the app to probe your model's supported structured output mode and streaming latency.

#### Option B: Google Gemini
Set in `.env`:
```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

### Rate Limiting & Quota Management (Tokens & Requests)

Both requests and tokens are tracked in a persistent SQLite quota database (`.cache/quota.db`) with sliding windows and configurable daily reset boundaries:

- `LLM_RPM`: Requests per minute limit (default `15`).
- `LLM_RPD`: Requests per day limit (default: None).
- `LLM_TPM`: Tokens per minute limit (default: None / 0 for unlimited).
- `LLM_TPD`: Tokens per day limit (default: None / 0 for unlimited).
- `LLM_DAY_RESET`: Daily reset boundary mode:
  - `pacific_midnight`: Midnight Pacific Time (Google Gemini default).
  - `utc_midnight`: Midnight UTC (Groq default).
  - `rolling_24h`: Continuous 24-hour sliding window.
- `LLM_LIMITS_FILE`: Optional path to a JSON file mapping per-model rate limits (e.g. `data/limits.json`).

#### Finding Your Limits in the Groq Console
1. Log in to [Groq Console](https://console.groq.com/).
2. Navigate to **Settings** $\to$ **Limits**.
3. Locate your chosen model to view its exact RPM, RPD, TPM, and TPD allocations.
4. Copy these into your `.env` or `LLM_LIMITS_FILE`.

### Per-Role Model and Token Sizing

- `PLANNER_MODEL`, `SYNTH_MODEL`, `CRITIC_MODEL`, `CHAT_MODEL`: Optional per-step model overrides.
- `PLANNER_MAX_TOKENS`: Max tokens for planner subtask decomposition (default `2048`).
- `CRITIC_MAX_TOKENS`: Max tokens for critic evaluation judge (default `2048`).
- `SYNTH_MAX_TOKENS`: Max tokens for report synthesis (default `4096`).
- `CHAT_MAX_TOKENS`: Max tokens for chat responses (default `2048`).

If a model terminates with `finish_reason=length`, the system retries once with an expanded token budget if within TPM/TPD limits.

---

## Live Provider Smoke Test

To verify your live API key, model streaming, and structured output capabilities before running runs:

```bash
python scripts/smoke_llm.py
```

### Example Expected Output

```text
==================================================================
Research Agent Crew - Live LLM Verification & Smoke Test
Provider:  GROQ
Model:     <model-name>
API Key:   gsk_...9a12 (56 chars)
==================================================================

--- 1. Testing Live Streaming & Token Delivery ---
Antigravity is online.
Time to First Token (TTFT): 142.3ms | Total Duration: 0.28s
Full response received (22 chars): Antigravity is online.

--- 2. Probing Structured Output Capability ---
Testing mode 'function_calling'... PASS (180ms) -> status='ok' message='System fully operational' verified=True
Testing mode 'json_mode'... PASS (195ms) -> status='ok' message='System fully operational' verified=True
Testing mode 'json_schema'... FAIL (400 Bad Request: model does not support schema enforcement)

[RECOMMENDATION] Set GROQ_STRUCTURED_MODE=function_calling for optimal reliability.

==================================================================
ALL CHECKS PASSED: Provider and model are operational for production.
==================================================================
```

---

## RAG Embedding & Score Calibration

1. **Download Local Embedding Model**:
   ```bash
   python -m core.rag.download_model
   ```

2. **Calibrate Grounding Thresholds (100% Offline)**:
   ```bash
   python scripts/calibrate_scores.py
   ```
   Evaluates cosine similarity percentiles (p10, p50, p95) on calibration sample pairs and prints recommended `RAG_MIN_SCORE` and `CHAT_MIN_SCORE` values.

---

## Launching Applications

### Launch FastAPI Service
```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Launch Streamlit Observability Dashboard
```bash
python -m streamlit run ui/streamlit_app.py
```

---

## Testing Modes

The test suite is structured into three execution modes to balance feedback speed with thoroughness:

### 1. Fast Mode (`pytest -m "not slow"`)
- **Target**: Fast local feedback loop (< 20s).
- **Execution time**: ~19s (137 tests).
- **Scope**: Core unit tests, in-memory state, deterministic ratelimit counters, schema migrations, and prompt routing.
```bash
python -m pytest -m "not slow" -q --durations=10
```

### 2. Pre-commit Mode (`pytest -m "not slow or critical"`)
- **Target**: Pre-commit CI validation of all system invariants (< 30s).
- **Execution time**: ~28s (143 tests).
- **Scope**: All fast tests PLUS critical invariant verification:
  - Search budget hard ceilings under parallelism
  - Concurrent rate limiter & token reservation safety
  - Mid-stream disconnect & node crash event persistence
  - Missing-model fail-fast configuration enforcement
  - Citation validation & hallucination prevention
  - Offline network guard enforcement (blocking non-loopback calls)
  - Cross-process database-backed chat lock acquisition & stale takeover
```bash
python -m pytest -m "not slow or critical" -q --durations=10
```

### 3. Full Suite (`pytest`)
- **Target**: Comprehensive end-to-end integration and resilience testing.
- **Execution time**: ~105s (171 tests).
- **Scope**: All 171 tests, including multi-process worker crashes and resume recovery (`test_real_subprocess_kill_and_resumption`), cross-process SQLite database tailing (`test_cross_process_db_tailing_without_gaps_or_duplicates`), live SSE streaming subscriptions, and real embedding model verification.
```bash
python -m pytest -q --durations=10
```

---

## Migration Notes

- **Quota Tracking Upgrade**: Migration to schema version 2 in `.cache/quota.db` introduces `llm_call_events` for token and request tracking while maintaining 100% backward compatibility with legacy `model_daily_quota` tables.
- **Provider Switching**: Set `LLM_PROVIDER=groq` or `LLM_PROVIDER=gemini`. Existing runs, checkpointer state, and threads seamlessly resume under either provider.
