import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    prescription: str
    created_at: float
    drugs_found: list[str] = field(default_factory=list)
    missing_leaflets: list[str] = field(default_factory=list)


# session_id → SessionData
_sessions: dict[str, SessionData] = {}


def save_prescription(session_id: str, text: str) -> None:
    _sessions[session_id] = SessionData(prescription=text, created_at=time.monotonic())


def save_upload_result(
    session_id: str,
    drugs_found: list[str],
    missing_leaflets: list[str],
) -> None:
    """Record which drugs were successfully indexed and which had no leaflet."""
    entry = _sessions.get(session_id)
    if entry is not None:
        entry.drugs_found = drugs_found
        entry.missing_leaflets = missing_leaflets


def get_prescription(session_id: str) -> str | None:
    entry = _sessions.get(session_id)
    return entry.prescription if entry is not None else None


def get_upload_result(
    session_id: str,
) -> tuple[list[str], list[str]]:
    """Return (drugs_found, missing_leaflets) for *session_id*, or two empty lists."""
    entry = _sessions.get(session_id)
    if entry is None:
        return [], []
    return entry.drugs_found, entry.missing_leaflets


def expire_sessions(ttl_seconds: float) -> list[str]:
    """Evict sessions older than *ttl_seconds*.

    Returns the list of evicted session IDs so callers can also clean up
    any associated external state (e.g. ChromaDB documents).
    """
    now = time.monotonic()
    expired = [
        sid for sid, data in _sessions.items() if now - data.created_at > ttl_seconds
    ]
    for sid in expired:
        del _sessions[sid]
        logger.info("Session expired and removed: %s", sid)
    return expired
