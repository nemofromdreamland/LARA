"""Tests for the session eviction background loop in app.main."""

import asyncio
from unittest.mock import patch

import pytest

from app.main import _eviction_loop


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch):
    import app.main as m

    monkeypatch.setattr(m.settings, "expiry_interval_seconds", 0)
    monkeypatch.setattr(m.settings, "session_ttl_seconds", 7200)


async def test_eviction_loop_calls_expire_and_delete():
    """One iteration of the loop must expire sessions and delete their docs."""
    with (
        patch(
            "app.main.expire_sessions", return_value=["sid-1", "sid-2"]
        ) as mock_expire,
        patch("app.main.delete_session", return_value=3) as mock_delete,
    ):
        task = asyncio.create_task(_eviction_loop())
        await asyncio.sleep(0.05)  # let one iteration run
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    mock_expire.assert_called_with(7200)
    assert mock_delete.call_count >= 2


async def test_eviction_loop_survives_exception():
    """An exception inside the loop must not kill the task."""
    call_count = 0

    def _boom(ttl):
        nonlocal call_count
        call_count += 1
        raise RuntimeError("transient error")

    with patch("app.main.expire_sessions", side_effect=_boom):
        task = asyncio.create_task(_eviction_loop())
        await asyncio.sleep(0.1)  # let at least two iterations attempt
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert call_count >= 2
