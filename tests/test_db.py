"""Unit tests for Database migrations, Thread/Message/Run/Event repositories,
cascade deletion, checkpoint purging, and concurrent writer safety.
"""

import asyncio
import pytest
import aiosqlite

from core.db.connection import get_db
from core.db.migrator import apply_migrations, get_current_user_version
from core.db.repositories import (
    ThreadRepo,
    MessageRepo,
    RunRepo,
    EventRepo
)
from core.checkpoint import get_checkpointer


@pytest.mark.asyncio
async def test_migrations_apply_once_and_rerun_is_noop(tmp_path):
    """Applying migrations on a fresh DB creates tables, and re-running is a no-op."""
    db_file = str(tmp_path / "test_mig.db")

    ver1 = await apply_migrations(db_path=db_file)
    assert ver1 >= 1

    async with get_db(db_file) as conn:
        curr_ver = await get_current_user_version(conn)
        assert curr_ver == ver1

        # Check all four tables exist
        async with conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('threads', 'runs', 'messages', 'run_events');"
        ) as cur:
            tables = [row[0] for row in await cur.fetchall()]
            assert set(tables) == {"threads", "runs", "messages", "run_events"}

    # Re-run migration; should be an immediate no-op returning same version
    ver2 = await apply_migrations(db_path=db_file)
    assert ver2 == ver1


@pytest.mark.asyncio
async def test_thread_crud_and_autotitle_without_llm(tmp_path):
    """Thread CRUD operations work properly and auto-title truncates without an LLM call."""
    db_file = str(tmp_path / "test_threads.db")
    await apply_migrations(db_path=db_file)

    repo = ThreadRepo(db_path=db_file)

    # 1. Test auto-titling logic
    short_prompt = "What is MCP?"
    assert repo.auto_title(short_prompt) == "What is MCP?"

    long_prompt = "Compare LangGraph vs AutoGen vs CrewAI for production enterprise multi-agent architectures and state management"
    auto_t = repo.auto_title(long_prompt)
    assert len(auto_t) <= 50
    assert auto_t.endswith("...")

    # 2. Create thread with auto-title
    t1 = await repo.create_thread(prompt=long_prompt)
    assert t1.title == auto_t
    assert t1.archived is False

    # 3. Create thread with custom title
    t2 = await repo.create_thread(title="Custom Title 123")
    assert t2.title == "Custom Title 123"

    # 4. List threads
    threads = await repo.list_threads()
    assert len(threads) == 2

    # 5. Rename thread
    renamed = await repo.rename_thread(t2.id, "Renamed Title")
    assert renamed.title == "Renamed Title"

    # 6. Archive thread
    archived = await repo.archive_thread(t1.id, archived=True)
    assert archived.archived is True

    # Without include_archived, only 1 is listed
    active_only = await repo.list_threads(include_archived=False)
    assert len(active_only) == 1
    assert active_only[0].id == t2.id

    # With include_archived, both listed
    all_threads = await repo.list_threads(include_archived=True)
    assert len(all_threads) == 2


@pytest.mark.slow
@pytest.mark.asyncio
async def test_cascade_delete_and_checkpoints_purged(tmp_path, monkeypatch):
    """Deleting a thread cascades to runs/messages/events and purges checkpoints & writes via adelete_thread without touching other threads."""
    db_file = str(tmp_path / "test_cascade.db")
    chk_file = str(tmp_path / "test_checkpoints.db")
    monkeypatch.setenv("DATABASE_PATH", db_file)
    monkeypatch.setenv("CHECKPOINT_DB_PATH", chk_file)

    await apply_migrations(db_path=db_file)

    t_repo = ThreadRepo(db_path=db_file)
    m_repo = MessageRepo(db_path=db_file)
    r_repo = RunRepo(db_path=db_file)
    e_repo = EventRepo(db_path=db_file)

    # Create Thread 1 (to be deleted)
    t1 = await t_repo.create_thread(title="Cascade Thread 1")
    r1 = await r_repo.create_run(thread_id=t1.id, task="Task 1")
    await m_repo.create_message(thread_id=t1.id, role="user", content="Hello 1", run_id=r1.id)
    await e_repo.append_event(run_id=r1.id, ts="2026-11-03T00:00:00Z", event_type="test1")

    # Create Thread 2 (should remain untouched)
    t2 = await t_repo.create_thread(title="Untouched Thread 2")
    r2 = await r_repo.create_run(thread_id=t2.id, task="Task 2")
    await m_repo.create_message(thread_id=t2.id, role="user", content="Hello 2", run_id=r2.id)

    # Populate both checkpoints and writes tables for Thread 1 and Thread 2
    async with get_checkpointer(db_path=chk_file) as checkpointer:
        t1_key = f"{t1.id}:{r1.id}"
        t2_key = f"{t2.id}:{r2.id}"
        await checkpointer.conn.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, '', 'chk1', '', 'json', X'00', X'00')",
            (t1_key,)
        )
        await checkpointer.conn.execute(
            "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, '', 'chk1', 't1', 0, 'ch', 'json', X'00')",
            (t1_key,)
        )
        await checkpointer.conn.execute(
            "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, '', 'chk2', '', 'json', X'00', X'00')",
            (t2_key,)
        )
        await checkpointer.conn.execute(
            "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value) VALUES (?, '', 'chk2', 't2', 0, 'ch', 'json', X'00')",
            (t2_key,)
        )
        await checkpointer.conn.commit()

    # Verify rows exist before deletion
    async with aiosqlite.connect(chk_file) as conn:
        assert (await (await conn.execute("SELECT count(*) FROM checkpoints WHERE thread_id = ?", (t1_key,))).fetchone())[0] == 1
        assert (await (await conn.execute("SELECT count(*) FROM writes WHERE thread_id = ?", (t1_key,))).fetchone())[0] == 1
        assert (await (await conn.execute("SELECT count(*) FROM checkpoints WHERE thread_id = ?", (t2_key,))).fetchone())[0] == 1
        assert (await (await conn.execute("SELECT count(*) FROM writes WHERE thread_id = ?", (t2_key,))).fetchone())[0] == 1

    # Delete Thread 1
    deleted = await t_repo.delete_thread(t1.id)
    assert deleted is True

    # Verify Thread 1 data cascaded in app DB
    async with get_db(db_file) as conn:
        assert (await (await conn.execute("SELECT count(*) FROM threads WHERE id = ?", (t1.id,))).fetchone())[0] == 0
        assert (await (await conn.execute("SELECT count(*) FROM runs WHERE thread_id = ?", (t1.id,))).fetchone())[0] == 0
        assert (await (await conn.execute("SELECT count(*) FROM messages WHERE thread_id = ?", (t1.id,))).fetchone())[0] == 0
        # Thread 2 untouched in app DB
        assert (await (await conn.execute("SELECT count(*) FROM threads WHERE id = ?", (t2.id,))).fetchone())[0] == 1

    # Verify checkpoints and writes for Thread 1 are ZERO, and Thread 2 is UNTOUCHED
    async with aiosqlite.connect(chk_file) as conn:
        t1_chks = (await (await conn.execute("SELECT count(*) FROM checkpoints WHERE thread_id LIKE ?", (f"{t1.id}:%",))).fetchone())[0]
        t1_writes = (await (await conn.execute("SELECT count(*) FROM writes WHERE thread_id LIKE ?", (f"{t1.id}:%",))).fetchone())[0]
        assert t1_chks == 0
        assert t1_writes == 0

        t2_chks = (await (await conn.execute("SELECT count(*) FROM checkpoints WHERE thread_id LIKE ?", (f"{t2.id}:%",))).fetchone())[0]
        t2_writes = (await (await conn.execute("SELECT count(*) FROM writes WHERE thread_id LIKE ?", (f"{t2.id}:%",))).fetchone())[0]
        assert t2_chks == 1
        assert t2_writes == 1


@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_writers_do_not_corrupt_data(tmp_path):
    """Concurrent writers appending messages and events under SQLite WAL mode succeed."""
    db_file = str(tmp_path / "test_concurrent.db")
    await apply_migrations(db_path=db_file)

    t_repo = ThreadRepo(db_path=db_file)
    r_repo = RunRepo(db_path=db_file)
    e_repo = EventRepo(db_path=db_file)

    thread = await t_repo.create_thread(title="Concurrent Thread")
    run = await r_repo.create_run(thread_id=thread.id, task="Concurrent Task")

    num_concurrent = 20

    async def _write_event(idx: int):
        await e_repo.append_event(
            run_id=run.id,
            ts="2026-11-03T00:00:00Z",
            event_type="concurrent_test",
            node="worker",
            payload={"index": idx}
        )

    await asyncio.gather(*[_write_event(i) for i in range(num_concurrent)])

    total_events = await e_repo.count_events_by_run(run.id)
    assert total_events == num_concurrent

    events = await e_repo.list_events_by_run(run.id)
    assert len(events) == num_concurrent
    indices = {e.payload["index"] for e in events}
    assert indices == set(range(num_concurrent))
