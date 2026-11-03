"""Streamlit Observability Dashboard for Research Agent Crew.

Provides an interactive user interface to launch research workflows, watch multi-agent
state transitions in real time, inspect MCP tool calls and Critic rubric scorecards,
and download final structured reports and JSON telemetry.
"""

import sys
import os
import json
import asyncio
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load developer configuration from .env if present
env_file = PROJECT_ROOT / ".env"
if env_file.is_file():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v

import time
import traceback
import streamlit as st
from core.graph import build_research_graph, create_initial_state

st.set_page_config(
    page_title="Research Agent Crew | Observability Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling for sleek, modern UI
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="collapsedControl"] {
        display: none;
    }
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #2e3646;
        margin-bottom: 12px;
    }
    .badge-planner { background-color: #3b82f6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
    .badge-executor { background-color: #8b5cf6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
    .badge-critic { background-color: #f59e0b; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
    .badge-finalizer { background-color: #10b981; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
    .badge-pass { background-color: #059669; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .badge-fail { background-color: #dc2626; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .tool-pill { background-color: #1f2937; border: 1px solid #374151; padding: 4px 8px; border-radius: 6px; font-family: monospace; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# Developer Configuration Check (Environment / .env)
gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
env_model = os.environ.get("GEMINI_MODEL", "").strip()
chosen_model = env_model or "gemini-2.0-flash"
critic_env_model = os.environ.get("CRITIC_MODEL", "").strip() or chosen_model

try:
    max_retries = int(os.environ.get("MAX_RETRIES", "1"))
except ValueError:
    max_retries = 1

try:
    search_budget = int(os.environ.get("SEARCH_BUDGET", "10"))
except ValueError:
    search_budget = 10

is_configured = bool(gemini_key and chosen_model)


# Main App Header
st.title("Research Agent Crew 🤖🔍")
st.markdown("""
Autonomous multi-agent research crew using **LangGraph** with a **Planner $\\to$ Executor $\\to$ Synthesizer $\\to$ Critic** reflection loop, real **Model Context Protocol (MCP)** tools, and multidimensional quality rubrics. Powered by **Google Gemini**.
""")

if not is_configured:
    st.error("⚠️ **Configuration Incomplete**: `GEMINI_API_KEY` (and `GEMINI_MODEL`) must be configured in your environment or `.env` file before executing research runs.")

# Sample Preset Buttons
sample_queries = [
    "Compare LangGraph vs AutoGen vs CrewAI for production agent architectures",
    "Analyze Quantum Computing breakthroughs in 2024-2026 and enterprise cryptography risks",
    "Deep dive into Model Context Protocol (MCP) specification, architecture, and security",
]

st.markdown("##### 💡 Example Research Queries:")
cols = st.columns(len(sample_queries))
selected_preset = None
for i, col in enumerate(cols):
    if col.button(sample_queries[i], key=f"sample_{i}", use_container_width=True):
        selected_preset = sample_queries[i]

# Query Input Form
with st.form("research_form"):
    default_text = selected_preset or (
        st.session_state.current_result["task"]
        if st.session_state.current_result else
        "Compare LangGraph vs AutoGen vs CrewAI for production agent architectures"
    )
    user_query = st.text_area(
        "Enter your research question or investigation goal:",
        value=default_text,
        height=90
    )
    submit_btn = st.form_submit_button("🚀 Launch Research Crew", use_container_width=True)

# Execution Logic
if submit_btn and user_query:
    active_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    active_model = os.environ.get("GEMINI_MODEL", "").strip() or chosen_model
    if not active_key or not active_model:
        st.error("Cannot launch research crew: **GEMINI_API_KEY** and **GEMINI_MODEL** are required. Please configure them in your environment or `.env` file.")
        st.stop()

    st.session_state.is_running = True

    async def _stream_execution():
        graph = build_research_graph()
        initial_state = create_initial_state(
            task=user_query,
            max_retries=max_retries,
            max_search_budget=search_budget,
            model_name=active_model
        )
        current_state = dict(initial_state)
        start_time = time.perf_counter()

        with st.status(f"Starting research on '{user_query[:50]}...'...", expanded=True) as status_box:
            try:
                async for chunk in graph.astream(initial_state, stream_mode="updates"):
                    for node_name, updates in chunk.items():
                        current_state.update(updates)
                        elapsed = round(time.perf_counter() - start_time, 1)
                        subtasks_count = len(current_state.get("subtask_results", []))
                        searches_done = current_state.get("total_searches_performed", 0)
                        cache_hits = current_state.get("cache_hits", 0)

                        status_box.update(
                            label=f"[{node_name.upper()}] in progress | Subtasks: {subtasks_count} | Searches: {searches_done}/{search_budget} | Cache hits: {cache_hits} | Elapsed: {elapsed}s"
                        )
                        st.write(
                            f"✓ Node **{node_name.capitalize()}** completed | Searches: `{searches_done}/{search_budget}` | Cache hits: `{cache_hits}` | Elapsed: `{elapsed}s`"
                        )

                total_time = round(time.perf_counter() - start_time, 1)
                final_status = current_state.get("status", "completed").upper()
                status_box.update(
                    label=f"Investigation complete ({final_status}) in {total_time}s!",
                    state="complete",
                    expanded=False
                )
                return current_state
            except Exception as exc:
                tb_summary = traceback.format_exc()
                status_box.update(
                    label=f"Execution failed: {str(exc)}",
                    state="error",
                    expanded=True
                )
                st.error(f"**Error Details**: {str(exc)}")
                st.code(tb_summary, language="python")
                raise

    try:
        result_state = asyncio.run(_stream_execution())
        st.session_state.current_result = result_state
    except Exception:
        pass
    finally:
        st.session_state.is_running = False

# Results View
result = st.session_state.current_result
if result:
    st.markdown("---")

    # High-level Metrics Row
    c1, c2, c3, c4, c5 = st.columns(5)
    last_eval = result.get("critic_evaluations", [])[-1] if result.get("critic_evaluations") else None
    retries = result.get("retry_count", 0)
    searches = result.get("total_searches_performed", 0)
    subtasks_count = len(result.get("subtask_results", []))

    c1.metric("Status", result.get("status", "unknown").upper())
    c2.metric("Critic Verdict", last_eval.verdict if last_eval else "N/A")
    c3.metric("Rubric Score", f"{last_eval.total_score}/15" if last_eval else "N/A")
    c4.metric("Reflection Loops", f"{retries} / {max_retries}")
    c5.metric("MCP Searches", f"{searches} / {search_budget}")

    if result.get("human_escalation_needed"):
        st.warning(f"⚠️ **Human Review Recommended**: {result.get('escalation_reason')}")

    search_alerts = [
        ev for ev in result.get("trace_events", [])
        if ev.event_type == "tool_result"
        and ev.details.get("tool_name") == "search_web"
        and ev.details.get("status") in ("rate_limited", "timeout", "error")
    ]
    if search_alerts:
        with st.expander(f"⚠️ Search Diagnostic Alerts ({len(search_alerts)})", expanded=False):
            for alert in search_alerts:
                status_label = alert.details.get("status", "alert").upper()
                st.warning(f"**[{status_label}]** {alert.summary}")

    # Tabs for detailed inspection
    tab_report, tab_trace, tab_critic, tab_sources = st.tabs([
        "📄 Final Research Report",
        "🔍 Observability & Execution Trace",
        "⚖️ Critic Rubric Audit",
        "🌐 Discovered Citations"
    ])

    # TAB 1: Report View & Downloads
    with tab_report:
        final_md = result.get("final_report", "No report generated.")
        struct = result.get("final_report_structured") or result.get("draft_report_structured")

        if struct:
            # Helper accessors for dict or Pydantic object
            s_title = struct.title if hasattr(struct, "title") else struct.get("title", "Research Report")
            s_tldr = struct.tldr if hasattr(struct, "tldr") else struct.get("tldr", [])
            s_sections = struct.sections if hasattr(struct, "sections") else struct.get("sections", [])
            s_comparison = struct.comparison if hasattr(struct, "comparison") else struct.get("comparison")
            s_recommendation = struct.recommendation if hasattr(struct, "recommendation") else struct.get("recommendation")
            s_open_gaps = struct.open_gaps if hasattr(struct, "open_gaps") else struct.get("open_gaps", [])
            s_limitations = struct.limitations if hasattr(struct, "limitations") else struct.get("limitations", [])

            st.header(s_title)

            # 1. TL;DR Component
            if s_tldr:
                st.subheader("💡 Executive Summary (TL;DR)")
                for item in s_tldr:
                    st.markdown(f"- **{item}**")
                st.markdown("---")

            # 2. Sections Components
            if s_sections:
                st.subheader("📑 Research Findings by Subtask")
                for sec in s_sections:
                    sec_heading = sec.heading if hasattr(sec, "heading") else sec.get("heading", "")
                    sec_summary = sec.summary if hasattr(sec, "summary") else sec.get("summary", "")
                    sec_claims = sec.claims if hasattr(sec, "claims") else sec.get("claims", [])

                    with st.expander(f"📌 {sec_heading}", expanded=True):
                        if sec_summary:
                            st.write(sec_summary)
                        if sec_claims:
                            st.markdown("**Grounded Claims & Citations:**")
                            for c in sec_claims:
                                c_text = c.text if hasattr(c, "text") else c.get("text", "")
                                c_urls = c.source_urls if hasattr(c, "source_urls") else c.get("source_urls", [])
                                c_conf = c.confidence if hasattr(c, "confidence") else c.get("confidence", 1.0)
                                links = " ".join(f"[[{i+1}]({u})]" for i, u in enumerate(c_urls))
                                conf_pct = f"{c_conf:.0%}"
                                badge = f":green[🎯 Confidence: {conf_pct}]" if c_conf >= 0.7 else f":orange[⚠️ Confidence: {conf_pct}]"
                                st.markdown(f"- {c_text} {links} &nbsp; {badge}")
                st.markdown("---")

            # 3. Comparison Table Component
            if s_comparison:
                st.subheader("📊 Comparative Analysis")
                comp_rows = []
                for row in s_comparison:
                    dim = row.dimension if hasattr(row, "dimension") else row.get("dimension", "")
                    vals = row.values if hasattr(row, "values") else row.get("values", {})
                    comp_rows.append({"Dimension": dim, **vals})
                if comp_rows:
                    st.dataframe(comp_rows, use_container_width=True)
                st.markdown("---")

            # 4. Recommendation Component
            if s_recommendation:
                st.subheader("🎯 Recommendation")
                st.success(s_recommendation)
                st.markdown("---")

            # 5. Open Gaps Component
            if s_open_gaps:
                st.subheader("⚠️ Open Gaps & Unverified Claims")
                for gap in s_open_gaps:
                    st.warning(gap)
                st.markdown("---")

            # 6. Limitations Component
            if s_limitations:
                st.subheader("ℹ️ Limitations & Scope")
                for lim in s_limitations:
                    st.info(lim)
                st.markdown("---")

            # Full document expander
            with st.expander("📄 View Full Rendered Markdown Document", expanded=False):
                st.markdown(final_md)
        else:
            st.subheader("Synthesized Research Report")
            st.markdown(final_md)

        st.markdown("---")
        col_d1, col_d2, col_d3 = st.columns([1, 1, 1])
        col_d1.download_button(
            label="📥 Download Report (.md)",
            data=final_md,
            file_name="research_report.md",
            mime="text/markdown",
            use_container_width=True
        )

        if struct:
            if hasattr(struct, "model_dump_json"):
                struct_json_str = struct.model_dump_json(indent=2)
            else:
                struct_json_str = json.dumps(struct, indent=2)
        else:
            struct_json_str = "{}"

        col_d2.download_button(
            label="📥 Download Structured Report (.json)",
            data=struct_json_str,
            file_name="structured_report.json",
            mime="application/json",
            use_container_width=True
        )

        json_meta = json.dumps(result.get("report_metadata", {}), indent=2)
        col_d3.download_button(
            label="📥 Download Telemetry (.json)",
            data=json_meta,
            file_name="research_summary.json",
            mime="application/json",
            use_container_width=True
        )
        if result.get("mcp_saved_path"):
            st.info(f"💾 Persisted to disk via Filesystem MCP: `{result.get('mcp_saved_path')}`")

    # TAB 2: Observability & Trace Tree
    with tab_trace:
        st.subheader("Step-by-Step Observability Trace")
        st.caption("Complete chronological record of agent reasoning, MCP tool calls, parameters, and reflection events.")

        traces = result.get("trace_events", [])
        if not traces:
            st.info("No trace events recorded.")
        else:
            for i, ev in enumerate(traces, 1):
                badge_class = f"badge-{ev.step}"
                header_html = (
                    f"<span class='{badge_class}'>{ev.step.upper()}</span> "
                    f"<b>{ev.summary}</b> "
                    f"<span style='color: gray; font-size: 11px;'>({ev.timestamp[11:19]} UTC | Iteration {ev.iteration})</span>"
                )
                with st.expander(f"Step {i}: [{ev.step.upper()}] {ev.summary}"):
                    st.markdown(header_html, unsafe_allow_html=True)
                    st.markdown(f"**Event Type:** `{ev.event_type}`")
                    st.json(ev.details)

    # TAB 3: Critic Rubric Audit
    with tab_critic:
        st.subheader("Multidimensional Critic Evaluation History")
        st.markdown("""
        The Critic agent evaluates the synthesized report using a hybrid mechanism:
        **Deterministic Audit Checks** (citation validity, source relevance, evidence gap rates, coverage mapping, and budget adherence)
        combined with an **LLM Judge** scoring Coverage, Evidence, and Coherence with hard scoring caps.
        """)

        evals = result.get("critic_evaluations", [])
        traces = result.get("trace_events", [])
        eval_traces = [e for e in traces if e.event_type == "rubric_evaluation"]

        for i, ev in enumerate(evals):
            verdict_badge = "<span class='badge-pass'>PASS</span>" if ev.verdict == "PASS" else "<span class='badge-fail'>FAIL (Sent for Revision)</span>"
            st.markdown(f"### Iteration {ev.iteration} Review — {verdict_badge} (Score: {ev.total_score}/15)", unsafe_allow_html=True)
            st.write(f"**Executive Feedback:** {ev.overall_feedback}")

            # Check if this iteration has trace details for deterministic checks and caps
            matching_trace = eval_traces[i] if i < len(eval_traces) else None
            trace_details = matching_trace.details if matching_trace else {}
            caps_fired = trace_details.get("caps_fired", [])
            det_metrics = trace_details.get("deterministic_metrics", {})

            if caps_fired:
                for cap in caps_fired:
                    st.warning(f"⚠️ **Hard Scoring Cap Triggered**: {cap}")

            if det_metrics:
                st.markdown("##### ⚙️ Deterministic Audit Telemetry")
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Citation Validity", f"{det_metrics.get('citation_validity', 0.0) * 100:.1f}%")
                d2.metric("Source Relevance", f"{det_metrics.get('source_relevance', 0.0) * 100:.1f}%")
                d3.metric("Evidence Gap Rate", f"{det_metrics.get('evidence_gap_rate', 0.0) * 100:.1f}%")
                d4.metric("Unique Sources", det_metrics.get("min_sources", 0))

                cov_map = det_metrics.get("coverage_map", {})
                if cov_map:
                    cov_items = [f"`{k}`: {'✓ Covered' if v else '✗ Missing'}" for k, v in cov_map.items()]
                    st.caption("Subtask Coverage Map: " + " | ".join(cov_items))

            st.markdown("##### 📊 Rubric Dimension Scorecard")
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("Coverage", f"{ev.coverage.score}/5")
                st.caption(ev.coverage.rationale)
                st.progress(ev.coverage.score / 5.0)

            with col_r2:
                st.metric("Evidence", f"{ev.evidence.score}/5")
                st.caption(ev.evidence.rationale)
                st.progress(ev.evidence.score / 5.0)

            with col_r3:
                st.metric("Coherence", f"{ev.coherence.score}/5")
                st.caption(ev.coherence.rationale)
                st.progress(ev.coherence.score / 5.0)

            if ev.next_step_instructions:
                st.markdown("**Actionable Revision Directives for Planner:**")
                for step in ev.next_step_instructions:
                    st.markdown(f"- 🔄 *{step}*")
            st.markdown("---")

    # TAB 4: Discovered Sources
    with tab_sources:
        st.subheader("Discovered Citations & Source Evidence")
        all_sources = []
        for r in result.get("subtask_results", []):
            for s in r.sources:
                all_sources.append({
                    "Subtask": r.title,
                    "Title": s.title,
                    "URL": s.url,
                    "Snippet": s.snippet
                })

        if all_sources:
            for item in all_sources:
                st.markdown(f"##### [{item['Title']}]({item['URL']})")
                st.caption(f"Linked to: *{item['Subtask']}*")
                st.write(item["Snippet"])
                st.markdown("---")
        else:
            st.info("No external URLs discovered.")

    # Expandable Section: Per-Node Wall-Clock Timings & Performance Breakdown
    timings = result.get("timings") or {}
    node_timings = timings.get("nodes", {})
    if node_timings:
        with st.expander("⏱️ Per-Node Wall-Clock Timings & Performance Breakdown", expanded=False):
            st.markdown("##### Node Latency Distribution")
            t_cols = st.columns(len(node_timings))
            for col, (node_name, duration_ms) in zip(t_cols, node_timings.items()):
                col.metric(f"{node_name.capitalize()}", f"{duration_ms:.1f} ms")

            total_wall = timings.get("total_wall_clock_ms") or sum(node_timings.values())
            st.caption(
                f"Cumulative node execution: **{total_wall:.2f} ms** | "
                f"Tool calls tracked: **{len(timings.get('tool_calls', []))}** | "
                f"Cache hits: **{result.get('cache_hits', 0)}**"
            )
