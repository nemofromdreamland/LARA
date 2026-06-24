"""Tests for session_store Redis-backed implementation.

The conftest _fake_redis fixture wires a FakeRedis instance before every test,
so no real Redis connection is required.
"""

import asyncio

from app.models.schemas import PrescriptionEntry
from app.services.session_store import (
    _hist_key,
    _key,
    append_history,
    create_session,
    get_prescription,
    get_prescription_entries,
    get_redis,
    get_session_data,
    get_upload_result,
    save_prescription,
    save_prescription_entries,
    save_upload_result,
    session_exists,
    set_session_data,
)

# ── Round-trip tests ──────────────────────────────────────────────────────────


async def test_set_and_get_returns_same_value():
    await set_session_data("s1", "mykey", {"foo": "bar", "n": 42})
    result = await get_session_data("s1", "mykey")
    assert result == {"foo": "bar", "n": 42}


async def test_get_missing_key_returns_none():
    result = await get_session_data("nosession", "nokey")
    assert result is None


async def test_get_missing_field_returns_none():
    await create_session("s2")
    result = await get_session_data("s2", "nonexistent_field")
    assert result is None


# ── TTL / expiry ──────────────────────────────────────────────────────────────


async def test_expired_session_returns_none():
    """A hash with TTL=1 must be unreadable after 1 second."""
    import unittest.mock as mock

    from app.config import settings

    # Temporarily set session_ttl_seconds to 1
    with mock.patch.object(settings, "session_ttl_seconds", 1):
        await set_session_data("expire_me", "val", "hello")
        await asyncio.sleep(1.1)

    result = await get_session_data("expire_me", "val")
    assert result is None


# ── session_exists ────────────────────────────────────────────────────────────


async def test_session_exists_after_create():
    await create_session("exists_test")
    assert await session_exists("exists_test") is True


async def test_session_exists_false_for_unknown():
    assert await session_exists("ghost_session") is False


# ── delete_session ────────────────────────────────────────────────────────────


async def test_delete_session_removes_hash(monkeypatch):
    """delete_session must remove the Redis hash (vector_store call mocked)."""
    import app.services.session_store as ss

    monkeypatch.setattr(ss, "delete_session", _make_delete_without_chroma(ss))

    await create_session("del_test")
    await set_session_data("del_test", "x", 1)
    assert await session_exists("del_test") is True

    # Call the raw Redis deletion directly (bypassing ChromaDB side-effect)
    import app.services.session_store as _m

    r = _m.get_redis()
    await r.delete(_m._key("del_test"))

    assert await session_exists("del_test") is False


def _make_delete_without_chroma(ss_module):
    """Return an async delete_session that skips the ChromaDB call."""

    async def _delete(session_id: str) -> None:
        r = ss_module.get_redis()
        await r.delete(ss_module._key(session_id))

    return _delete


# ── High-level wrappers ───────────────────────────────────────────────────────


async def test_save_and_get_prescription():
    await save_prescription("p1", "Drug A 50mg")
    assert await get_prescription("p1") == "Drug A 50mg"


async def test_get_prescription_unknown_session_returns_none():
    assert await get_prescription("nonexistent") is None


async def test_save_and_get_upload_result():
    await save_upload_result("u1", ["lisinopril"], ["tylenol"])
    found, missing = await get_upload_result("u1")
    assert found == ["lisinopril"]
    assert missing == ["tylenol"]


async def test_get_upload_result_unknown_session_returns_empty_lists():
    found, missing = await get_upload_result("ghost")
    assert found == []
    assert missing == []


async def test_save_and_get_prescription_entries():
    entries = [
        PrescriptionEntry(drug_name="ibuprofen", dosage="400mg", frequency="TID"),
        PrescriptionEntry(drug_name="azithromycin", dosage="500mg"),
    ]
    await save_prescription_entries("e1", entries)
    result = await get_prescription_entries("e1")
    assert len(result) == 2
    assert result[0].drug_name == "ibuprofen"
    assert result[0].dosage == "400mg"
    assert result[1].drug_name == "azithromycin"
    assert result[1].frequency is None


async def test_get_prescription_entries_returns_empty_for_missing_session():
    assert await get_prescription_entries("nonexistent") == []


# ── Sibling TTL refresh (no drift) ────────────────────────────────────────────


async def test_append_history_refreshes_session_ttl():
    """append_history must keep the sibling session key's TTL in lockstep."""
    sid = "ttl-hist"
    await create_session(sid)  # creates session:{sid} with full TTL
    await append_history(sid, "user", "seed")  # creates history:{sid}

    r = get_redis()
    # Simulate drift: shorten only the session key.
    await r.expire(_key(sid), 30)
    assert await r.ttl(_key(sid)) <= 30

    await append_history(sid, "assistant", "reply")

    # Both keys are now back to the full session TTL.
    assert await r.ttl(_key(sid)) > 1000
    assert await r.ttl(_hist_key(sid)) > 1000


async def test_set_session_data_refreshes_history_ttl():
    """set_session_data must keep the sibling history key's TTL in lockstep."""
    sid = "ttl-sess"
    await append_history(sid, "user", "seed")  # creates history:{sid} only

    r = get_redis()
    # Simulate drift: shorten only the history key.
    await r.expire(_hist_key(sid), 30)
    assert await r.ttl(_hist_key(sid)) <= 30

    await set_session_data(sid, "field", "value")

    # Both keys are now back to the full session TTL.
    assert await r.ttl(_hist_key(sid)) > 1000
    assert await r.ttl(_key(sid)) > 1000


async def test_set_session_data_does_not_create_phantom_history():
    """No history key should be created when none exists yet."""
    sid = "no-phantom-hist"
    await set_session_data(sid, "field", "value")
    r = get_redis()
    assert await r.exists(_hist_key(sid)) == 0


async def test_append_history_does_not_create_phantom_session():
    """No session key should be created when none exists yet."""
    sid = "no-phantom-sess"
    await append_history(sid, "user", "hi")
    r = get_redis()
    assert await r.exists(_key(sid)) == 0
