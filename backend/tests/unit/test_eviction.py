"""Tests for the Redis session lifecycle (replaces the old eviction-loop tests).

The background eviction loop has been removed; Redis TTL handles expiry natively.
These tests verify the guard and lifecycle contracts on session_store.
"""

import pytest

import app.services.session_store as ss
from app.services.session_store import (
    create_session,
    get_session_data,
    session_exists,
    set_session_data,
)


async def test_get_redis_raises_when_not_initialised(monkeypatch):
    """_get_redis() must raise RuntimeError when _redis is None."""
    monkeypatch.setattr(ss, "_redis", None)
    with pytest.raises(RuntimeError, match="Redis not initialised"):
        ss._get_redis()


async def test_close_redis_is_idempotent(monkeypatch):
    """close_redis() must not raise when _redis is already None."""
    monkeypatch.setattr(ss, "_redis", None)
    await ss.close_redis()  # must not raise


async def test_create_session_sets_created_at():
    """create_session must persist a created_at field in Redis."""
    await create_session("lifecycle_test")
    created_at = await get_session_data("lifecycle_test", "created_at")
    assert created_at is not None
    assert isinstance(created_at, float)


async def test_session_exists_true_after_create():
    await create_session("exists1")
    assert await session_exists("exists1") is True


async def test_session_exists_false_for_unknown():
    assert await session_exists("does_not_exist") is False


async def test_set_session_data_refreshes_key():
    """Writing a new field to an existing session must not lose earlier fields."""
    await create_session("refresh_test")
    await set_session_data("refresh_test", "field_a", "value_a")
    await set_session_data("refresh_test", "field_b", "value_b")

    assert await get_session_data("refresh_test", "field_a") == "value_a"
    assert await get_session_data("refresh_test", "field_b") == "value_b"
