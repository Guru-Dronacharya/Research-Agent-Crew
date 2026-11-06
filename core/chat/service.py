"""Thread Chat Service orchestrating streaming, retrieval, quota safety, and persistence.

Guarantees:
1. One active chat per thread (409 on concurrent chat or running research crew).
2. User message persisted BEFORE any LLM call.
3. Assistant message persisted in a finally block (complete, or partial flagged quota|client_disconnect|error).
4. Zero-LLM 3-way routing (canned, thread, quick, research).
5. On-the-fly report chunks with source URLs mapped without altering documents.kind.
6. Delimiter-neutralized untrusted chunks.
7. Streaming citation buffering, validation, and stripping.
8. Daily quota tracking before every LLM call.
"""

import os
import uuid
import asyncio
from typing import AsyncIterator, Dict, Any, List, Optional, Set
from pydantic import BaseModel

from core.db.repositories import (
    ThreadRepo,
    MessageRepo,
    RunRepo,
    DocumentRepo,
    ChunkRepo,
    ChatLockRepo,
    EventRepo
)
from core.llm import get_agent_llm, resolve_model_name, get_provider
from core.ratelimit import QuotaExceededError
from core.rag import query_hybrid_rag, get_embedder, RetrievedChunk
from core.chat.citation_filter import CitationStreamFilter
from core.chat.router import classify_query, RouterDecision
from core.chat.quick_path import execute_quick_path_retrieval

def get_chat_model() -> str:
    try:
        return resolve_model_name("CHAT_MODEL")
    except Exception:
        return os.environ.get("CHAT_MODEL", "default")

CHAT_MODEL = os.environ.get("CHAT_MODEL", os.environ.get("GEMINI_MODEL", "default"))
CHAT_MAX_CONTEXT_CHARS = int(os.environ.get("CHAT_MAX_CONTEXT_CHARS", "8000"))
CHAT_HISTORY_TURNS = int(os.environ.get("CHAT_HISTORY_TURNS", "6"))
CHAT_MIN_SCORE = float(os.environ.get("CHAT_MIN_SCORE", "0.35"))

_ACTIVE_CHATS: Set[str] = set()
_ACTIVE_CHATS_LOCK = asyncio.Lock()


class ChatConflictError(RuntimeError):
    """Raised when a concurrent chat or running research crew is active on a thread."""
    pass


class ChatEvent(BaseModel):
    """Container for SSE chat events."""
    event: str
    data: Dict[str, Any]


def _build_report_chunks(runs: List[Any], thread_id: str) -> List[RetrievedChunk]:
    """Build on-the-fly chunks from the thread's latest completed report without altering documents.kind."""
    report_chunks: List[RetrievedChunk] = []
    completed_runs = [r for r in runs if r.status in ("completed", "finished") and (r.report_json or r.report_md)]
    if not completed_runs:
        return report_chunks

    latest_run = completed_runs[0]
    if latest_run.report_json:
        try:
            import json as _json
            data = _json.loads(latest_run.report_json) if isinstance(latest_run.report_json, str) else latest_run.report_json
            sections = data.get("sections", []) if isinstance(data, dict) else []
            for idx, sec in enumerate(sections):
                chunk_id = f"report-{idx + 1}"
                heading = sec.get("heading", f"Section {idx + 1}")
                summary = sec.get("summary", "")
                claims = sec.get("claims", [])

                source_urls: List[str] = []
                claims_texts: List[str] = []
                for c in claims:
                    if isinstance(c, dict):
                        source_urls.extend(c.get("source_urls", []))
                        claims_texts.append(f"- {c.get('text', '')}")
                    elif hasattr(c, "source_urls"):
                        source_urls.extend(c.source_urls)
                        claims_texts.append(f"- {c.text}")

                seen = set()
                unique_urls = [u for u in source_urls if not (u in seen or seen.add(u))]
                claims_text = "\n".join(claims_texts)
                text = f"## {heading}\n{summary}\n{claims_text}".strip()

                report_chunks.append(
                    RetrievedChunk(
                        id=chunk_id,
                        document_id=f"run-{latest_run.id}",
                        thread_id=thread_id,
                        text=text,
                        score=0.95,
                        source_url=unique_urls[0] if unique_urls else "",
                        title=f"Report: {heading}",
                        meta={
                            "is_report": True,
                            "run_id": latest_run.id,
                            "source_urls": unique_urls,
                            "heading": heading
                        }
                    )
                )
            if report_chunks:
                return report_chunks
        except Exception:
            pass

    # Fallback to report_md if report_json is missing or unparseable
    if latest_run.report_md:
        sections = latest_run.report_md.split("\n## ")
        for idx, sec_text in enumerate(sections):
            if not sec_text.strip():
                continue
            chunk_id = f"report-{idx + 1}"
            lines = sec_text.splitlines()
            heading = lines[0].strip("# ") if lines else "Report Section"
            report_chunks.append(
                RetrievedChunk(
                    id=chunk_id,
                    document_id=f"run-{latest_run.id}",
                    thread_id=thread_id,
                    text=f"## {sec_text}".strip(),
                    score=0.90,
                    source_url="",
                    title=f"Report: {heading}",
                    meta={"is_report": True, "run_id": latest_run.id, "source_urls": []}
                )
            )

    return report_chunks


async def stream_thread_chat(
    thread_id: str,
    message: str,
    allow_general_knowledge: Optional[bool] = None,
    mode: str = "auto",
    db_path: Optional[str] = None
) -> AsyncIterator[ChatEvent]:
    """Execute streaming thread chat with citations, zero-LLM routing, and quota safety."""
    thread_repo = ThreadRepo(db_path=db_path)
    msg_repo = MessageRepo(db_path=db_path)
    run_repo = RunRepo(db_path=db_path)
    chat_lock_repo = ChatLockRepo(db_path=db_path)

    # 1. Concurrency Check: In-process fast path + Active research run check + Cross-process DB lock
    async with _ACTIVE_CHATS_LOCK:
        if thread_id in _ACTIVE_CHATS:
            raise ChatConflictError(f"A chat is already active on thread '{thread_id}'.")

    active_runs = await run_repo.list_runs_by_thread(thread_id)
    event_repo = EventRepo(db_path=db_path)
    for r in active_runs:
        if r.status in ("running", "executing", "pending"):
            events = await event_repo.list_events_by_run(r.id)
            current_node = None
            for ev in reversed(events):
                if ev.type == "node_started":
                    current_node = ev.node
                    break
                elif ev.type == "node_finished":
                    break
            if current_node in ("synthesizer", "critic", "synthesis"):
                raise ChatConflictError(
                    f"Cannot chat while a research run ('{r.id}') is in {current_node} on thread '{thread_id}'."
                )
            raise ChatConflictError(
                f"Cannot chat while a research run ('{r.id}', status='{r.status}') is active on thread '{thread_id}'."
            )

    owner_id = f"chat_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    acquired = await chat_lock_repo.acquire_lock(thread_id, owner_id)
    if not acquired:
        active_lock = await chat_lock_repo.get_active_lock(thread_id)
        other_owner = active_lock["owner_id"] if active_lock else "another process"
        raise ChatConflictError(
            f"A chat is already active on thread '{thread_id}' (held by {other_owner})."
        )

    async with _ACTIVE_CHATS_LOCK:
        _ACTIVE_CHATS.add(thread_id)

    # Start background heartbeat loop while streaming
    heartbeat_stop = asyncio.Event()

    async def _heartbeat_loop():
        stale_sec = int(os.environ.get("HEARTBEAT_STALE_SECONDS", "60"))
        interval = max(1.0, min(float(stale_sec) / 4.0, 3.0))
        while not heartbeat_stop.is_set():
            try:
                await asyncio.sleep(interval)
                if not heartbeat_stop.is_set():
                    await chat_lock_repo.refresh_heartbeat(thread_id, owner_id)
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    heartbeat_task = asyncio.create_task(_heartbeat_loop())

    # Variables for final assistant persistence
    accumulated_deltas: List[str] = []
    partial = False
    partial_reason: Optional[str] = None
    gemini_requests = 0
    general_knowledge_used = False
    low_confidence = False
    final_citations: List[Dict[str, Any]] = []
    canned_mode = False
    no_material_mode = False

    try:
        thread = await thread_repo.get_thread(thread_id)
        if not thread:
            raise ValueError(f"Thread '{thread_id}' not found.")

        # Persist thread settings if general knowledge toggle was passed
        effective_gk = thread.settings.get("allow_general_knowledge", False)
        if allow_general_knowledge is not None:
            effective_gk = bool(allow_general_knowledge)
            await thread_repo.update_thread_settings(thread_id, {"allow_general_knowledge": effective_gk})

        # 2. Persist the user message BEFORE any search or LLM call
        await msg_repo.create_message(
            thread_id=thread_id,
            role="user",
            content=message
        )

        # 3. Route Query (Zero-LLM Heuristic)
        chosen_mode = mode
        if chosen_mode == "auto":
            decision: RouterDecision = await classify_query(
                query=message,
                thread_id=thread_id,
                db_path=db_path,
                min_score=CHAT_MIN_SCORE
            )
            chosen_mode = decision.mode
            if decision.canned_response:
                canned_mode = True
                accumulated_deltas.append(decision.canned_response)
                yield ChatEvent(event="answer_delta", data={"delta": decision.canned_response})
                yield ChatEvent(event="citation", data={"citations": []})
                yield ChatEvent(
                    event="done",
                    data={
                        "model": "canned",
                        "gemini_requests": 0,
                        "mode": "canned"
                    }
                )
                return

            if chosen_mode == "research":
                reply = (
                    f"This query was routed to the **Full Research Crew** ({decision.reason}).\n\n"
                    "Would you like to initiate a multi-subtask deep dive research run on this thread?"
                )
                accumulated_deltas.append(reply)
                yield ChatEvent(event="answer_delta", data={"delta": reply})
                yield ChatEvent(event="citation", data={"citations": []})
                yield ChatEvent(
                    event="done",
                    data={
                        "model": "router",
                        "gemini_requests": 0,
                        "mode": "research",
                        "reason": decision.reason
                    }
                )
                return

        # 4. Context Preparation (Quick Path vs Thread Knowledge)
        retrieved_chunks: List[RetrievedChunk] = []

        if chosen_mode == "quick":
            q_chunks, is_low_conf, _ = await execute_quick_path_retrieval(
                thread_id=thread_id,
                query=message,
                db_path=db_path
            )
            retrieved_chunks = q_chunks
            low_confidence = is_low_conf
        else:
            # Thread knowledge mode: retrieve from indexed docs + on-the-fly report chunks
            doc_repo = DocumentRepo(db_path=db_path)
            chunk_repo = ChunkRepo(db_path=db_path)
            embedder = get_embedder()

            th_chunks, gap = await query_hybrid_rag(
                thread_id=thread_id,
                query=message,
                embedder=embedder,
                chunk_repo=chunk_repo,
                document_repo=doc_repo,
                top_k=5,
                min_score=CHAT_MIN_SCORE
            )
            retrieved_chunks.extend(th_chunks)

            # On-the-fly report chunks
            runs = await run_repo.list_runs_by_thread(thread_id)
            report_chunks = _build_report_chunks(runs, thread_id)
            retrieved_chunks.extend(report_chunks[:3])

        # Evaluate support
        max_score = max((c.score for c in retrieved_chunks), default=0.0)
        has_support = len(retrieved_chunks) > 0 and (max_score >= CHAT_MIN_SCORE or any(c.meta.get("is_report") for c in retrieved_chunks))

        if not has_support:
            if not effective_gk:
                no_material_mode = True
                reply = (
                    "This thread does not contain supporting material to answer your question.\n\n"
                    "You can enable **Allow General Knowledge** to answer from model memory, "
                    "or launch a **Full Research Run** to gather live evidence."
                )
                accumulated_deltas.append(reply)
                yield ChatEvent(event="answer_delta", data={"delta": reply})
                yield ChatEvent(event="citation", data={"citations": []})
                yield ChatEvent(
                    event="done",
                    data={
                        "model": "grounding-guard",
                        "gemini_requests": 0,
                        "grounded": False
                    }
                )
                return
            else:
                general_knowledge_used = True

        # Build chunk map and sanitize chunks against prompt injection
        chunk_map: Dict[str, Dict[str, Any]] = {}
        context_parts: List[str] = []
        current_chars = 0

        for ch in retrieved_chunks:
            chunk_map[ch.id] = {
                "id": ch.id,
                "title": ch.title,
                "url": ch.source_url,
                "source_urls": ch.meta.get("source_urls", [ch.source_url] if ch.source_url else []),
                "page": ch.meta.get("page")
            }
            # Neutralize potential delimiters in untrusted text
            clean_text = ch.text.replace("</evidence_chunk>", "&lt;/evidence_chunk&gt;").replace("<evidence_chunk", "&lt;evidence_chunk")
            part = f'<evidence_chunk id="{ch.id}">\nTitle: {ch.title}\nContent: {clean_text}\n</evidence_chunk>\n'
            if current_chars + len(part) > CHAT_MAX_CONTEXT_CHARS:
                break
            context_parts.append(part)
            current_chars += len(part)

        context_str = "\n".join(context_parts)

        # 5. Build Filtered History
        all_messages = await msg_repo.list_messages(thread_id)
        # Exclude partial, no-supporting-material, canned, and current turn
        filtered_history = [
            m for m in all_messages
            if not m.metadata.get("partial", False)
            and not m.metadata.get("no_supporting_material", False)
            and not m.metadata.get("canned", False)
            and m.content != message
        ]
        recent_history = filtered_history[-CHAT_HISTORY_TURNS * 2:]

        prompt_messages: List[Dict[str, str]] = []
        gk_instruction = ""
        if general_knowledge_used:
            gk_instruction = (
                "NOTE: No direct thread evidence was found. You are answering from general knowledge. "
                "Explicitly preface or conclude that this response is drawn from general knowledge and not thread sources.\n"
            )

        confidence_instruction = ""
        if low_confidence:
            confidence_instruction = (
                "NOTE: Only web snippets were available (full pages could not be extracted). "
                "Mark your answer as 'low confidence: sources not fully read'.\n"
            )

        system_prompt = (
            "You are an accurate, grounded research assistant. Answer the user's question clearly and concisely.\n"
            f"{gk_instruction}{confidence_instruction}"
            "Ground your statements in the provided evidence chunks whenever possible.\n"
            "Cite every grounded claim using inline bracket citations referencing valid chunk ids, like [chunk-1] or [report-1].\n"
            "Do NOT invent or hallucinate citation ids. Only cite ids that exist in the evidence chunks below.\n\n"
            f"=== EVIDENCE CHUNKS ===\n{context_str}\n=== END EVIDENCE CHUNKS ==="
        )
        prompt_messages.append({"role": "system", "content": system_prompt})

        for hist_msg in recent_history:
            prompt_messages.append({"role": hist_msg.role, "content": hist_msg.content})

        prompt_messages.append({"role": "user", "content": message})

        # 6. Stream with Citation Filter & Quota Tracking
        valid_ids = set(chunk_map.keys())
        citation_filter = CitationStreamFilter(valid_chunk_ids=valid_ids)
        agent_llm = get_agent_llm()

        gemini_requests = 1

        try:
            active_model = get_chat_model()
            stream = agent_llm.astream_chat(prompt_messages, model_name=active_model)
            async for raw_delta in stream:
                clean_deltas = citation_filter.process_delta(raw_delta)
                for cd in clean_deltas:
                    accumulated_deltas.append(cd)
                    yield ChatEvent(event="answer_delta", data={"delta": cd})

            # Finalize citation filter buffer
            for trailing in citation_filter.finalize():
                accumulated_deltas.append(trailing)
                yield ChatEvent(event="answer_delta", data={"delta": trailing})

            final_citations = citation_filter.build_citations(chunk_map)
            yield ChatEvent(event="citation", data={"citations": final_citations})
            yield ChatEvent(
                event="done",
                data={
                    "model": get_chat_model(),
                    "provider": get_provider(),
                    "gemini_requests": 1,
                    "grounded": has_support and not general_knowledge_used,
                    "low_confidence": low_confidence,
                    "general_knowledge_used": general_knowledge_used
                }
            )

        except QuotaExceededError as q_err:
            partial = True
            partial_reason = "quota"
            yield ChatEvent(
                event="error",
                data={
                    "error": str(q_err),
                    "reset_time": getattr(q_err, "reset_time", "N/A"),
                    "partial": True
                }
            )
        except (asyncio.CancelledError, GeneratorExit):
            partial = True
            partial_reason = "client_disconnect"
            raise
        except Exception as exc:
            partial = True
            partial_reason = "error"
            yield ChatEvent(
                event="error",
                data={"error": str(exc), "partial": True}
            )

    finally:
        # Stop background heartbeat task
        heartbeat_stop.set()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except (asyncio.CancelledError, Exception):
            pass

        # Release DB-backed lock
        try:
            await asyncio.shield(chat_lock_repo.release_lock(thread_id, owner_id))
        except Exception:
            pass

        # Release in-process lock
        async with _ACTIVE_CHATS_LOCK:
            _ACTIVE_CHATS.discard(thread_id)

        # 7. Persist Assistant Message in finally block

        full_content = "".join(accumulated_deltas).strip()

        # If partial and zero tokens were received, do NOT persist an empty assistant message
        if partial and not full_content:
            pass
        elif not canned_mode and not no_material_mode:
            metadata: Dict[str, Any] = {
                "partial": partial,
                "partial_reason": partial_reason,
                "model": get_chat_model(),
                    "provider": get_provider(),
                "gemini_requests": gemini_requests,
                "citations": final_citations,
                "general_knowledge_used": general_knowledge_used,
                "low_confidence": low_confidence
            }
            try:
                await asyncio.shield(msg_repo.create_message(
                    thread_id=thread_id,
                    role="assistant",
                    content=full_content,
                    metadata=metadata
                ))
            except Exception:
                pass
        elif canned_mode:
            try:
                await asyncio.shield(msg_repo.create_message(
                    thread_id=thread_id,
                    role="assistant",
                    content=full_content,
                    metadata={"canned": True, "gemini_requests": 0}
                ))
            except Exception:
                pass
        elif no_material_mode:
            try:
                await asyncio.shield(msg_repo.create_message(
                    thread_id=thread_id,
                    role="assistant",
                    content=full_content,
                    metadata={"no_supporting_material": True, "gemini_requests": 0}
                ))
            except Exception:
                pass
