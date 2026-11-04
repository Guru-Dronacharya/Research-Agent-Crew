"""Streamlit Observability Dashboard for Research Agent Crew.

Provides an interactive user interface to manage persistent research threads,
track background multi-agent runs, inspect checkpoints and Critic scorecards,
and resume or cancel in-flight or quota-paused investigations.
"""

import sys
import os
import json
import time
import asyncio
import threading
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load configuration from .env if present
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

import streamlit as st
from core.db.migrator import apply_migrations
from core.db.repositories import (
    ThreadRepo,
    MessageRepo,
    RunRepo,
    EventRepo
)
from core.services.runs import (
    start_run,
    resume_run,
    cancel_run,
    execute_run
)

st.set_page_config(
    page_title="Research Agent Crew | Observability Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for modern, clean UI with status badges
st.markdown("""
<style>
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
    .badge-status-completed { color: #10b981; font-weight: 600; font-size: 13px; }
    .badge-status-running { color: #3b82f6; font-weight: 600; font-size: 13px; }
    .badge-status-paused { color: #f59e0b; font-weight: 600; font-size: 13px; }
    .badge-status-failed { color: #ef4444; font-weight: 600; font-size: 13px; }
    .badge-status-cancelled { color: #9ca3af; font-weight: 600; font-size: 13px; }
    .tool-pill { background-color: #1f2937; border: 1px solid #374151; padding: 4px 8px; border-radius: 6px; font-family: monospace; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


class AsyncBackgroundRunner:
    """Dedicated long-lived background asyncio event loop thread for async execution."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="StreamlitAsyncBackgroundWorker"
        )
        self.thread.start()
        # Ensure database migrations are applied at worker startup
        self.run_sync(apply_migrations())
        # Recover interrupted runs
        run_repo = RunRepo()
        self.run_sync(run_repo.recover_interrupted_runs(cutoff_seconds=60))

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro):
        """Submit coroutine for background execution."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def run_sync(self, coro, timeout: float = 30.0):
        """Run coroutine and wait synchronously for result."""
        future = self.submit(coro)
        return future.result(timeout=timeout)


@st.cache_resource
def get_async_runner() -> AsyncBackgroundRunner:
    """Return singleton cached background runner."""
    return AsyncBackgroundRunner()


runner = get_async_runner()
thread_repo = ThreadRepo()
run_repo = RunRepo()
event_repo = EventRepo()
msg_repo = MessageRepo()

# Developer Configuration
gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
env_model = os.environ.get("GEMINI_MODEL", "").strip()
chosen_model = env_model or "gemini-2.0-flash"
try:
    max_retries = int(os.environ.get("MAX_RETRIES", "1"))
except ValueError:
    max_retries = 1
try:
    search_budget = int(os.environ.get("SEARCH_BUDGET", "10"))
except ValueError:
    search_budget = 10

is_configured = bool(gemini_key and chosen_model)

# Session State Initialization
if "active_thread_id" not in st.session_state:
    st.session_state.active_thread_id = None
if "editing_thread_id" not in st.session_state:
    st.session_state.editing_thread_id = None

# ==============================================================================
# Sidebar: Thread Management & Status Badges
# ==============================================================================
with st.sidebar:
    st.title("🔬 Research Threads")

    if st.button("➕ New Thread", use_container_width=True, type="primary"):
        new_t = runner.run_sync(thread_repo.create_thread(title="New Investigation"))
        st.session_state.active_thread_id = new_t.id
        st.rerun()

    show_archived = st.checkbox("Show Archived", value=False)
    threads = runner.run_sync(thread_repo.list_threads(include_archived=show_archived))

    if not threads:
        # Create initial thread if none exist
        default_t = runner.run_sync(thread_repo.create_thread(title="Default Investigation"))
        threads = [default_t]
        st.session_state.active_thread_id = default_t.id

    if st.session_state.active_thread_id not in [t.id for t in threads] and threads:
        st.session_state.active_thread_id = threads[0].id

    st.markdown("---")

    for t in threads:
        runs = runner.run_sync(run_repo.list_runs_by_thread(t.id))
        latest_run = runs[-1] if runs else None

        # Determine status badge
        status_icon = "⚪"
        status_class = "badge-status-cancelled"
        if latest_run:
            st_val = latest_run.status
            if st_val in ("completed", "escalated"):
                status_icon = "🟢"
                status_class = "badge-status-completed"
            elif st_val == "running":
                status_icon = "🔵"
                status_class = "badge-status-running"
            elif st_val == "paused_quota":
                status_icon = "🟠"
                status_class = "badge-status-paused"
            elif st_val == "failed":
                status_icon = "🔴"
                status_class = "badge-status-failed"
            elif st_val == "cancelled":
                status_icon = "⚪"
                status_class = "badge-status-cancelled"

        is_active = (t.id == st.session_state.active_thread_id)
        btn_label = f"{status_icon} {'👉 ' if is_active else ''}{t.title[:28]}"

        col_t, col_menu = st.columns([5, 1])
        if col_t.button(btn_label, key=f"sel_{t.id}", use_container_width=True):
            st.session_state.active_thread_id = t.id
            st.rerun()

        with col_menu:
            with st.popover("⚙️"):
                st.caption(f"Thread: {t.title}")
                new_name = st.text_input("Rename", value=t.title, key=f"ren_{t.id}")
                if st.button("Save Name", key=f"save_ren_{t.id}"):
                    if new_name.strip():
                        runner.run_sync(thread_repo.rename_thread(t.id, new_name.strip()))
                        st.rerun()

                if t.archived:
                    if st.button("Unarchive", key=f"unarch_{t.id}"):
                        runner.run_sync(thread_repo.archive_thread(t.id, archived=False))
                        st.rerun()
                else:
                    if st.button("Archive", key=f"arch_{t.id}"):
                        runner.run_sync(thread_repo.archive_thread(t.id, archived=True))
                        st.rerun()

                if st.button("🗑️ Delete", key=f"del_{t.id}", type="secondary"):
                    runner.run_sync(thread_repo.delete_thread(t.id))
                    st.session_state.active_thread_id = None
                    st.rerun()


# ==============================================================================
# Main Workspace: Active Thread & Runs
# ==============================================================================
active_id = st.session_state.active_thread_id
active_thread = runner.run_sync(thread_repo.get_thread(active_id)) if active_id else None

if not active_thread:
    st.info("Select or create a thread in the sidebar to begin.")
    st.stop()

st.title(f"🔍 {active_thread.title}")
st.caption(f"Thread ID: `{active_thread.id}` | Created: {active_thread.created_at[:19]}")

if not is_configured:
    st.error("⚠️ **Configuration Incomplete**: `GEMINI_API_KEY` (and `GEMINI_MODEL`) must be configured in your environment or `.env` file.")

# Fetch thread history and runs from SQLite
runs = runner.run_sync(run_repo.list_runs_by_thread(active_thread.id))
latest_run = runs[-1] if runs else None

# Check current run status
if latest_run and latest_run.status == "running":
    # Cancel control header
    ch_col1, ch_col2 = st.columns([4, 1])
    with ch_col1:
        st.info(f"⏳ **Run in Progress**: `{latest_run.task}`")
    with ch_col2:
        if st.button("⏹️ Cancel Run", key=f"cancel_{latest_run.id}", type="secondary", use_container_width=True):
            runner.run_sync(cancel_run(latest_run.id))
            st.warning("Cancellation requested...")
            time.sleep(1)
            st.rerun()

    # Retrieve all persisted events ordered by seq
    events = runner.run_sync(event_repo.list_events_by_run(latest_run.id))

    # Parse live progress telemetry from events
    active_node = "planner"
    discovered_sources = []
    sections_received = []
    subtasks_info = {}
    last_critic_info = None
    live_searches = 0
    live_cache_hits = 0

    for ev in events:
        etype = ev.type
        payload = ev.data or ev.payload or {}
        enode = ev.node or payload.get("node")
        if etype == "node_started":
            active_node = enode or active_node
        elif etype == "decomposition":
            for s in payload.get("subtasks", []):
                s_id = s.get("id") if isinstance(s, dict) else getattr(s, "id", None)
                s_title = s.get("title") if isinstance(s, dict) else getattr(s, "title", "")
                if s_id:
                    subtasks_info[s_id] = {"title": s_title, "sources": 0, "status": "pending"}
        elif etype == "source_found":
            discovered_sources.append(payload)
            s_id = ev.subtask_id or payload.get("subtask_id")
            if s_id and s_id in subtasks_info:
                subtasks_info[s_id]["sources"] += 1
                subtasks_info[s_id]["status"] = "evidence gathered"
            if payload.get("cached"):
                live_cache_hits += 1
        elif etype == "tool_call":
            if payload.get("tool_name") == "search_web":
                live_searches += 1
                s_id = ev.subtask_id or payload.get("subtask_id")
                if s_id and s_id in subtasks_info:
                    subtasks_info[s_id]["status"] = "searching"
        elif etype == "report_section":
            sections_received.append(payload)
        elif etype in ("critic_verdict", "rubric_evaluation"):
            last_critic_info = payload

    # Live Metrics Banner
    lm1, lm2, lm3, lm4 = st.columns(4)
    lm1.metric("Active Node", f"🤖 {active_node.upper()}")
    lm2.metric("Searches Executed", f"{live_searches} / {search_budget}")
    lm3.metric("Cache Hits", live_cache_hits)
    lm4.metric("Sources Found", len(discovered_sources))

    # Live Progress Status Container
    with st.status(f"⚡ Research Crew Live Stream: {active_node.capitalize()} Agent active ({len(events)} events)", expanded=True):
        st.markdown(f"**Current Phase:** `{active_node}` | **Discovered Evidence Items:** `{len(discovered_sources)}`")

        # Display subtasks progress if decomposed
        if subtasks_info:
            st.markdown("**Subtasks Execution Tracker:**")
            st_cols = st.columns(min(len(subtasks_info), 4))
            for idx, (sid, sinfo) in enumerate(subtasks_info.items()):
                with st_cols[idx % len(st_cols)]:
                    st.caption(f"**{sid}**: {sinfo['title'][:30]}")
                    st.write(f"Status: `{sinfo['status']}` ({sinfo['sources']} sources)")

        # Display synthesized sections as they arrive
        if sections_received:
            st.markdown("---")
            st.markdown("**📝 Live Synthesizer Sections:**")
            for sec in sections_received[-2:]:
                st.write(f"📌 **{sec.get('heading', '')}**: {sec.get('summary', '')[:120]}...")

        # Display critic verdict if ready
        if last_critic_info:
            st.markdown("---")
            v_val = last_critic_info.get("verdict", "")
            t_score = last_critic_info.get("total_score", 0)
            st.write(f"⚖️ **Critic Evaluation:** `{v_val}` (Score: `{t_score}/15`)")

        # Recent events log
        st.markdown("---")
        st.caption("Latest Telemetry Stream:")
        for ev in events[-6:]:
            p = ev.data or ev.payload or {}
            summ = p.get("summary") or p.get("message") or p.get("action_taken") or str(p)[:80]
            st.text(f"[{ev.ts[11:19]}] seq={ev.seq} | {ev.node or 'system'} | {ev.type} | {summ}")

    time.sleep(0.8)
    st.rerun()

elif latest_run and latest_run.status == "paused_quota":
    st.warning(
        f"⚠️ **Daily Quota Paused**: This run hit Google Gemini daily quota. "
        f"Scheduled to resume after `{latest_run.resume_after}`."
    )
    if latest_run.error:
        st.caption(f"Error detail: {latest_run.error}")

    c_r1, c_r2, _ = st.columns([1.5, 1.5, 4])
    with c_r1:
        if st.button("▶️ Resume Run", key=f"res_{latest_run.id}", type="primary"):
            try:
                runner.submit(resume_run(latest_run.id, force=False))
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(str(e))
    with c_r2:
        if st.button("⚡ Force Resume Now", key=f"fres_{latest_run.id}"):
            runner.submit(resume_run(latest_run.id, force=True))
            time.sleep(0.5)
            st.rerun()

elif latest_run and latest_run.status == "failed":
    st.error(f"❌ **Run Interrupted/Failed**: {latest_run.error or 'Process was interrupted.'}")
    col_rf1, col_rf2, _ = st.columns([1.5, 1.5, 4])
    with col_rf1:
        if st.button("🔄 Resume from Checkpoint", key=f"res_fail_{latest_run.id}", type="primary"):
            runner.submit(resume_run(latest_run.id, force=True))
            time.sleep(0.5)
            st.rerun()


# ==============================================================================
# Query Launch Form (when idle or completed)
# ==============================================================================
is_busy = bool(latest_run and latest_run.status == "running")

with st.form("research_form"):
    default_text = ""
    user_query = st.text_area(
        "Enter your research question or investigation goal:",
        value=default_text,
        height=80,
        placeholder="e.g. Compare LangGraph vs AutoGen vs CrewAI for production multi-agent architectures",
        disabled=is_busy
    )
    submit_btn = st.form_submit_button("🚀 Launch Research Crew", use_container_width=True, disabled=is_busy)

if submit_btn and user_query.strip():
    if not is_configured:
        st.error("Cannot launch: GEMINI_API_KEY is missing.")
        st.stop()

    cfg = {
        "max_retries": max_retries,
        "search_budget": search_budget,
        "model_name": chosen_model
    }

    # Auto-title thread if it still has default title
    if active_thread.title in ("New Investigation", "Default Investigation"):
        new_title = ThreadRepo.auto_title(user_query)
        runner.run_sync(thread_repo.rename_thread(active_thread.id, new_title))

    # Create run and dispatch to background worker thread
    new_run = runner.run_sync(start_run(
        thread_id=active_thread.id,
        task=user_query.strip(),
        config=cfg
    ))
    runner.submit(execute_run(new_run.id))
    st.rerun()


# ==============================================================================
# Results View: Latest Report & Observations
# ==============================================================================
if latest_run and latest_run.report_md:
    st.markdown("---")

    # High-level Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    metrics_data = latest_run.metrics or {}
    searches = metrics_data.get("searches_performed", 0)
    cache_hits = metrics_data.get("cache_hits", 0)
    retries = metrics_data.get("retries", 0)

    m_col1.metric("Run Status", latest_run.status.upper())
    m_col2.metric("Reflection Loops", f"{retries} / {max_retries}")
    m_col3.metric("MCP Searches", f"{searches} / {search_budget}")
    m_col4.metric("Cache Hits", cache_hits)

    struct = None
    if latest_run.report_json:
        try:
            struct = json.loads(latest_run.report_json)
        except Exception:
            pass

    tab_report, tab_trace, tab_sources = st.tabs([
        "📄 Final Research Report",
        "🔍 Observability & Run Events",
        "🌐 Discovered Citations"
    ])

    with tab_report:
        if struct and isinstance(struct, dict):
            s_title = struct.get("title", active_thread.title)
            s_tldr = struct.get("tldr", [])
            s_sections = struct.get("sections", [])
            s_comparison = struct.get("comparison")
            s_recommendation = struct.get("recommendation")
            s_open_gaps = struct.get("open_gaps", [])
            s_limitations = struct.get("limitations", [])

            st.header(s_title)

            # 1. TL;DR
            if s_tldr:
                st.subheader("💡 Executive Summary (TL;DR)")
                for item in s_tldr:
                    st.markdown(f"- **{item}**")
                st.markdown("---")

            # 2. Sections
            if s_sections:
                st.subheader("📑 Research Findings by Subtask")
                for sec in s_sections:
                    sec_heading = sec.get("heading", "")
                    sec_summary = sec.get("summary", "")
                    sec_claims = sec.get("claims", [])
                    with st.expander(f"📌 {sec_heading}", expanded=True):
                        if sec_summary:
                            st.write(sec_summary)
                        if sec_claims:
                            st.markdown("**Grounded Claims & Citations:**")
                            for c in sec_claims:
                                c_text = c.get("text", "")
                                c_urls = c.get("source_urls", [])
                                c_conf = c.get("confidence", 1.0)
                                links = " ".join(f"[[{i+1}]({u})]" for i, u in enumerate(c_urls))
                                conf_pct = f"{c_conf:.0%}"
                                badge = f":green[🎯 Confidence: {conf_pct}]" if c_conf >= 0.7 else f":orange[⚠️ Confidence: {conf_pct}]"
                                st.markdown(f"- {c_text} {links} &nbsp; {badge}")
                st.markdown("---")

            # 3. Comparison
            if s_comparison:
                st.subheader("📊 Comparative Analysis")
                comp_rows = []
                for row in s_comparison:
                    dim = row.get("dimension", "")
                    vals = row.get("values", {})
                    comp_rows.append({"Dimension": dim, **vals})
                if comp_rows:
                    st.dataframe(comp_rows, use_container_width=True)
                st.markdown("---")

            # 4. Recommendation
            if s_recommendation:
                st.subheader("🎯 Recommendation")
                st.success(s_recommendation)
                st.markdown("---")

            # 5. Open Gaps
            if s_open_gaps:
                st.subheader("⚠️ Open Gaps & Unverified Items")
                for gap in s_open_gaps:
                    st.warning(gap)
                st.markdown("---")

            # 6. Limitations
            if s_limitations:
                st.subheader("ℹ️ Limitations & Scope")
                for lim in s_limitations:
                    st.info(lim)
                st.markdown("---")

            with st.expander("📄 View Full Markdown Document", expanded=False):
                st.markdown(latest_run.report_md)
        else:
            st.markdown(latest_run.report_md)

        st.markdown("---")
        cd1, cd2 = st.columns(2)
        cd1.download_button(
            "📥 Download Report (.md)",
            data=latest_run.report_md,
            file_name=f"report_{latest_run.id[:8]}.md",
            mime="text/markdown",
            use_container_width=True
        )
        if latest_run.report_json:
            cd2.download_button(
                "📥 Download Structured JSON",
                data=latest_run.report_json,
                file_name=f"report_{latest_run.id[:8]}.json",
                mime="application/json",
                use_container_width=True
            )

    with tab_trace:
        run_events = runner.run_sync(event_repo.list_events_by_run(latest_run.id))
        st.subheader(f"Run Telemetry Events ({len(run_events)})")
        for ev in run_events:
            with st.expander(f"[{ev.ts[11:19]}] **{ev.node or 'system'}** — {ev.type}: {ev.payload.get('summary', '')}"):
                st.json(ev.payload)

    with tab_sources:
        st.subheader("Extracted Source Links")
        if struct and isinstance(struct, dict):
            sections = struct.get("sections", [])
            all_urls = set()
            for s in sections:
                for c in s.get("claims", []):
                    all_urls.update(c.get("source_urls", []))
            if all_urls:
                for u in sorted(all_urls):
                    st.markdown(f"- [{u}]({u})")
            else:
                st.info("No sources explicitly cited.")
        else:
            st.info("Source citations embedded in markdown report.")

