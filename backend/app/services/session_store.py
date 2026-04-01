import logging
import time

logger = logging.getLogger(__name__)

# session_id → (prescription_text, created_at_monotonic)
_sessions: dict[str, tuple[str, float]] = {}


def save_prescription(session_id: str, text: str) -> None:
    _sessions[session_id] = (text, time.monotonic())


def get_prescription(session_id: str) -> str | None:
    entry = _sessions.get(session_id)
    return entry[0] if entry is not None else None


def expire_sessions(ttl_seconds: float) -> list[str]:
    """Evict sessions older than *ttl_seconds*.

    Returns the list of evicted session IDs so callers can also clean up
    any associated external state (e.g. ChromaDB documents).
    """
    now = time.monotonic()
    expired = [
        sid
        for sid, (_, created_at) in _sessions.items()
        if now - created_at > ttl_seconds
    ]
    for sid in expired:
        del _sessions[sid]
        logger.info("Session expired and removed: %s", sid)
    return expired
