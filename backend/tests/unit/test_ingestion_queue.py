"""Tests for the durable ingestion queue (Redis Streams consumer group).

Uses the autouse fakeredis fixture from conftest (backs session_store.get_redis)
and mocks run_ingestion's collaborators so no real ML model or Chroma is touched.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.config import settings
from app.models.schemas import PrescriptionEntry
from app.services import ingestion_queue as iq
from app.services.dailymed import LeafletSection
from app.services.session_store import get_job_status, get_redis

MOCK_ENTRIES = [
    PrescriptionEntry(drug_name="lisinopril", dosage="10mg", frequency="qd")
]
MOCK_SECTIONS = [
    LeafletSection(
        drug_name="lisinopril",
        section="indications",
        text="Lisinopril is indicated for hypertension.",
    )
]


def _metric(event: str) -> float:
    """Current value of the lara_ingestion_jobs_total counter for *event*."""
    return iq._INGESTION_JOBS.labels(event=event)._value.get()


def _mock_pipeline():
    """Patch run_ingestion's collaborators for a successful single-drug ingest."""
    patches = {
        "parse_prescription": AsyncMock(return_value=MOCK_ENTRIES),
        "fetch_leaflet_sections": AsyncMock(return_value=MOCK_SECTIONS),
        "embed": AsyncMock(return_value=[[0.1] * 768]),
        "store": AsyncMock(return_value=None),
        "delete_session": AsyncMock(return_value=0),
    }
    return [patch(f"app.services.ingestion.{name}", m) for name, m in patches.items()]


# ── Producer ──────────────────────────────────────────────────────────────────


async def test_enqueue_adds_message_to_stream():
    before = _metric("enqueued")
    await iq.enqueue_ingestion("job1", "sess1", "rx text", "rid1")

    r = get_redis()
    assert await r.xlen(iq._STREAM) == 1
    _id, fields = (await r.xrange(iq._STREAM))[0]
    assert fields["job_id"] == "job1"
    assert fields["session_id"] == "sess1"
    assert fields["text"] == "rx text"
    assert fields["request_id"] == "rid1"
    assert _metric("enqueued") == before + 1


# ── Consumer happy path ─────────────────────────────────────────────────────────


async def test_read_new_processes_acks_and_sets_done():
    before = _metric("processed")
    ctx = _mock_pipeline()
    for p in ctx:
        p.start()
    try:
        await iq.enqueue_ingestion("job2", "sess2", "rx", "rid2")
        handled = await iq.read_new(None)
    finally:
        for p in ctx:
            p.stop()

    assert handled == 1
    data = await get_job_status("job2")
    assert data["status"] == "done"
    assert data["drugs_found"] == ["lisinopril"]
    # XACKed → no pending entries left for the group.
    r = get_redis()
    assert (await r.xpending(iq._STREAM, iq._GROUP))["pending"] == 0
    assert _metric("processed") == before + 1


async def test_no_drugs_marks_failed_and_acks():
    with patch(
        "app.services.ingestion.parse_prescription",
        new=AsyncMock(return_value=[]),
    ):
        await iq.enqueue_ingestion("job3", "sess3", "rx", "rid3")
        await iq.read_new(None)

    data = await get_job_status("job3")
    assert data["status"] == "failed"
    assert "No drug names found" in data["error"]
    r = get_redis()
    # Terminal logical failure is ACKed (not retried).
    assert (await r.xpending(iq._STREAM, iq._GROUP))["pending"] == 0


# ── Redelivery / idempotency ────────────────────────────────────────────────────


async def test_redelivery_does_not_duplicate_chunks():
    """run_ingestion clears the collection each attempt, so re-processing the same
    job (at-least-once redelivery) leaves a single copy of the chunks."""
    store_state: dict[str, list[str]] = {}

    async def fake_store(chunks, embeddings, metas, session_id, client=None):
        store_state.setdefault(session_id, []).extend(chunks)

    async def fake_delete(session_id, client=None):
        store_state.pop(session_id, None)
        return 0

    delete_mock = AsyncMock(side_effect=fake_delete)

    with (
        patch(
            "app.services.ingestion.parse_prescription",
            new=AsyncMock(return_value=MOCK_ENTRIES),
        ),
        patch(
            "app.services.ingestion.fetch_leaflet_sections",
            new=AsyncMock(return_value=MOCK_SECTIONS),
        ),
        patch(
            "app.services.ingestion.embed", new=AsyncMock(return_value=[[0.1] * 768])
        ),
        patch("app.services.ingestion.store", new=AsyncMock(side_effect=fake_store)),
        patch("app.services.ingestion.delete_session", new=delete_mock),
    ):
        from app.services.ingestion import run_ingestion

        await run_ingestion("job4", "sess4", "rx", "rid4", None)
        first_count = len(store_state["sess4"])
        await run_ingestion("job4", "sess4", "rx", "rid4", None)
        second_count = len(store_state["sess4"])

    assert first_count > 0
    assert second_count == first_count  # not doubled
    assert delete_mock.await_count == 2  # cleared at the start of each attempt


# ── Crash recovery (XAUTOCLAIM) ─────────────────────────────────────────────────


async def test_stale_message_is_reclaimed_and_completed():
    """Attempt 1 crashes (left pending); claim_stale reclaims and completes it."""
    from app.services.session_store import save_job_status

    calls = {"n": 0}

    async def flaky(job_id, session_id, text, rid, executor):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("worker crashed mid-ingestion")
        await save_job_status(job_id, session_id, "done", drugs_found=["lisinopril"])

    before = _metric("reclaimed")
    with patch("app.services.ingestion_queue.run_ingestion", side_effect=flaky):
        await iq.enqueue_ingestion("job5", "sess5", "rx", "rid5")
        await iq.read_new(None, consumer_name="worker-A")  # attempt 1 fails → pending
        r = get_redis()
        assert (await r.xpending(iq._STREAM, iq._GROUP))["pending"] == 1
        reclaimed = await iq.claim_stale(None, consumer_name="worker-B", min_idle_ms=0)

    assert reclaimed == 1
    assert calls["n"] == 2
    data = await get_job_status("job5")
    assert data["status"] == "done"
    assert (await get_redis().xpending(iq._STREAM, iq._GROUP))["pending"] == 0
    assert _metric("reclaimed") == before + 1


# ── Dead-letter (poison message) ────────────────────────────────────────────────


async def test_poison_message_is_dead_lettered(monkeypatch):
    monkeypatch.setattr(settings, "ingestion_max_attempts", 1)

    async def always_fail(*args, **kwargs):
        raise RuntimeError("permanent failure")

    before = _metric("dead_lettered")
    with patch("app.services.ingestion_queue.run_ingestion", side_effect=always_fail):
        await iq.enqueue_ingestion("job6", "sess6", "rx", "rid6")
        await iq.read_new(None)  # delivery 1: processed, fails → pending
        await iq.claim_stale(None, min_idle_ms=0)  # delivery 2 > max(1) → DLQ

    r = get_redis()
    assert await r.xlen(iq._DLQ) == 1
    _id, dlq_fields = (await r.xrange(iq._DLQ))[0]
    assert dlq_fields["job_id"] == "job6"
    assert "error" in dlq_fields
    assert (await r.xpending(iq._STREAM, iq._GROUP))["pending"] == 0
    data = await get_job_status("job6")
    assert data["status"] == "failed"
    assert _metric("dead_lettered") == before + 1


# ── Group bootstrap ─────────────────────────────────────────────────────────────


async def test_ensure_consumer_group_is_idempotent():
    await iq.ensure_consumer_group()
    await iq.ensure_consumer_group()  # second call must not raise (BUSYGROUP swallowed)
    groups = await get_redis().xinfo_groups(iq._STREAM)
    assert any(g["name"] == iq._GROUP for g in groups)


@pytest.mark.parametrize("block_ms", [None])
async def test_read_new_on_empty_stream_returns_zero(block_ms):
    assert await iq.read_new(None, block_ms=block_ms) == 0
