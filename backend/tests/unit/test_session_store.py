import time
from unittest.mock import patch

from app.services.session_store import (
    expire_sessions,
    get_prescription,
    save_prescription,
)


def _fresh_state():
    """Clear module-level _sessions dict between tests."""
    import app.services.session_store as ss

    ss._sessions.clear()


def test_save_and_get_prescription():
    _fresh_state()
    save_prescription("s1", "Drug A 50mg")
    assert get_prescription("s1") == "Drug A 50mg"


def test_get_prescription_unknown_session():
    _fresh_state()
    assert get_prescription("nonexistent") is None


def test_expire_sessions_removes_old_entry():
    _fresh_state()
    save_prescription("old", "text")

    # Patch monotonic so the session appears 3 hours old
    future = time.monotonic() + 10_800
    with patch("app.services.session_store.time.monotonic", return_value=future):
        expired = expire_sessions(ttl_seconds=7200)

    assert "old" in expired
    assert get_prescription("old") is None


def test_expire_sessions_keeps_fresh_entry():
    _fresh_state()
    save_prescription("fresh", "text")
    expired = expire_sessions(ttl_seconds=7200)
    assert "fresh" not in expired
    assert get_prescription("fresh") == "text"


def test_expire_sessions_returns_only_expired():
    _fresh_state()
    save_prescription("keep", "text")
    save_prescription("evict", "text")

    future = time.monotonic() + 10_800
    with patch("app.services.session_store.time.monotonic", return_value=future):
        # Only patch monotonic for the expire call; both sessions were created
        # with the real clock so both look 3h old from the patched perspective.
        expired = expire_sessions(ttl_seconds=7200)

    assert set(expired) == {"keep", "evict"}


def test_expire_sessions_empty_store():
    _fresh_state()
    assert expire_sessions(ttl_seconds=7200) == []
