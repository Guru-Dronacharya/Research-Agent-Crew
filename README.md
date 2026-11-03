# Research Agent Crew

Production-grade autonomous multi-agent research crew orchestrated via **LangGraph**, **Google Gemini**, and **Model Context Protocol (MCP)** tools.

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

The application requires a valid Google Gemini API key and model specification.

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

### Required Environment Variables

- `GEMINI_API_KEY`: API key for Google Gemini.
- `GEMINI_MODEL`: Model ID for planning, execution, and synthesis (e.g. `gemini-2.0-flash`, `gemini-1.5-pro`).

### Optional Environment Variables

- `PLANNER_MODEL`: Model used for task decomposition (defaults to `GEMINI_MODEL`).
- `SYNTH_MODEL`: Model used for report synthesis (defaults to `GEMINI_MODEL`).
- `CRITIC_MODEL`: Model used by the Critic LLM judge (defaults to `GEMINI_MODEL`).
- `EXECUTOR_CONCURRENCY`: Max concurrent subtask executions (default `3`).
- `LLM_RPM`: Requests per minute limit for Gemini LLM calls (default `15`).
- `SEARCH_RPM`: Requests per minute limit for web search calls (default `30`).
- `LLM_RPD`: Daily Gemini request cap that raises `QuotaExceededError` and escalates gracefully (default: None).
- `DDGS_BACKENDS`: Comma-delimited DuckDuckGo search engines (default `duckduckgo,wikipedia`).
- `DDGS_TIMEOUT_SEC`: Max per-search timeout for DuckDuckGo (default `6.0`).
- `CACHE_ENABLED`: Enable SQLite search caching (default `true`).
- `CACHE_DB_PATH`: File path to SQLite search cache database (default `.cache/search.db`).
- `CACHE_TTL_SECONDS`: Cache time-to-live in seconds (default `21600` = 6 hours; empty results capped to 10 min).
- `SYNTH_MAX_SOURCES_PER_SUBTASK`: Max sources passed per subtask to synthesizer (default `4`).
- `SYNTH_SNIPPET_MAX_CHARS`: Max snippet length per source passed to synthesizer (default `300`).
- `MCP_TOOL_TIMEOUT_SEC`: Timeout in seconds for MCP tool executions (default `15.0`).
- `INITIAL_BUDGET_FRACTION`: Fraction of total search budget allocated to the initial pass (default `0.6`).
- `MAX_RETRIES`: Maximum Critic reflection loops before finalizer (default `1`).
- `SEARCH_BUDGET`: Hard limit on MCP web searches across iterations (default `10`).
- `TAVILY_API_KEY`: Optional API key for Tavily search provider.
- `BRAVE_API_KEY`: Optional API key for Brave search provider.

---

## Performance Tuning & Architecture Optimizations

The pipeline is optimized for maximum speed and throughput while preserving all grounding and correctness guardrails:

1. **Parallel Subtask & Query Execution**:
   - The executor launches pending subtasks concurrently via `asyncio.gather`, bounded by `asyncio.Semaphore(EXECUTOR_CONCURRENCY)` (default: 3).
   - Within each subtask, initial search queries run concurrently.
   - Budget shares are pre-allocated per subtask, preventing shared counter race conditions.
   - Subtask failures are isolated with `evidence_gap=True` without aborting the entire run.

2. **Global Token-Bucket Rate Limiter**:
   - Shared `AsyncSlidingWindowLimiter` paces Gemini LLM requests (`LLM_RPM`) and MCP searches (`SEARCH_RPM`).
   - Prevents 429/ResourceExhausted quota exhaustion under high concurrency.
   - Tenacity retries respect `Retry-After` headers and `retryDelay` payloads.
   - Daily quota cap (`LLM_RPD`) triggers clean escalation instead of unhandled crashes.

3. **Restricted Engine Search & Relevance Prefilter**:
   - Provider hierarchy: Tavily (if configured) $\to$ Brave (if configured) $\to$ DuckDuckGo (DDGS). If an API key succeeds, DDGS is never called.
   - DDGS is restricted via `DDGS_BACKENDS` (default `duckduckgo,wikipedia`) instead of fanning out to 7 engines simultaneously, eliminating 429/202 delays.
   - Fast failures on 429/202 return status `"rate_limited"` immediately.
   - A keyword-overlap and non-English/spam prefilter discards irrelevant, gaming, or emulator results before evidence collection.

4. **Persistent SQLite Search Cache**:
   - Normalizes queries (lowercasing, stop-word removal, sorted tokens) + max_results + provider.
   - Cache hits cost **zero** search budget and return in $<1$ ms.
   - Errors, rate-limited responses, and timeouts are never cached; empty results expire in 10 minutes.
   - Concurrency safe using SQLite WAL mode and worker threads.

5. **Cheaper, Smarter Critic**:
   - Runs deterministic audit checks first. If a hard-fail condition is certain (e.g. `citation_validity < 1.0` or `evidence_gap_rate > 0.34`), the LLM judge is skipped entirely, saving latency and quota.
   - When the judge runs, rationales are capped to $\le 2$ sentences per dimension.

6. **Parallel Section Synthesis**:
   - Detailed findings sections for each subtask are synthesized in parallel via `asyncio.gather`.
   - One short final merge call produces the Title and Executive Summary.
   - Top 4 sources per subtask with 300-char snippets cut prompt token volume by $>60\%$.

7. **Live Observability (UI & API)**:
   - Streamlit dashboard uses `st.status(...)` updated live via `graph.astream(stream_mode="updates")` showing active node, subtasks, searches, cache hits, and elapsed time.
   - Per-node timing breakdowns are rendered in an expandable dashboard panel.
   - FastAPI `/research/{task_id}/status` exposes real-time `current_node`, `elapsed_seconds`, `searches_performed`, and `cache_hits`.

---

## Installation & Running

```bash
pip install -r requirements.txt
```

### Launch FastAPI Service

```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Launch Streamlit Observability Dashboard

```bash
python -m streamlit run ui/streamlit_app.py
```

### Run Test Suite

```bash
python -m pytest
```
