import json
import logging
import time
from typing import Any

import redis.asyncio as aioredis

from app.models.schemas import PrescriptionEntry

logger = logging.getLogger(__name__)

_redis: aioredis.Redis | None = None


def _key(session_id: str) -> str:
    return f"session:{session_id}"


def _get_redis() -> aioredis.Redis:
    if _redis is None:
        raise RuntimeError("Redis not initialised — call init_redis() first")
    return _redis


async def init_redis(url: str) -> None:
    global _redis
    client: aioredis.Redis = aioredis.from_url(url, decode_responses=True)
    await client.ping()  # fail fast if unreachable
    _redis = client
    logger.info("Redis connected: %s", url)


async def close_redis() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


# ── Low-level generic interface ──────────────────────────────────────────────


async def create_session(session_id: str) -> None:
    from app.config import settings

    r = _get_redis()
    key = _key(session_id)
    await r.hset(key, "created_at", json.dumps(time.time()))
    await r.expire(key, settings.session_ttl_seconds)


async def set_session_data(session_id: str, field: str, value: Any) -> None:
    from app.config import settings

    r = _get_redis()
    key = _key(session_id)
    await r.hset(key, field, json.dumps(value))
    await r.expire(key, settings.session_ttl_seconds)


async def get_session_data(session_id: str, field: str) -> Any | None:
    r = _get_redis()
    raw = await r.hget(_key(session_id), field)
    if raw is None:
        return None
    return json.loads(raw)


async def session_exists(session_id: str) -> bool:
    r = _get_redis()
    return bool(await r.exists(_key(session_id)))


async def delete_session(session_id: str) -> None:
    from app.services.vector_store import delete_session as vs_delete

    r = _get_redis()
    await r.delete(_key(session_id))
    await vs_delete(session_id)


# ── High-level domain wrappers (preserves existing call-sites) ────────────────


async def save_prescription(session_id: str, text: str) -> None:
    await set_session_data(session_id, "prescription", text)


async def get_prescription(session_id: str) -> str | None:
    return await get_session_data(session_id, "prescription")


async def save_prescription_entries(
    session_id: str, entries: list[PrescriptionEntry]
) -> None:
    await set_session_data(
        session_id, "prescription_entries", [e.model_dump() for e in entries]
    )


async def get_prescription_entries(session_id: str) -> list[PrescriptionEntry]:
    raw = await get_session_data(session_id, "prescription_entries")
    if raw is None:
        return []
    return [PrescriptionEntry(**e) for e in raw]


async def save_upload_result(
    session_id: str,
    drugs_found: list[str],
    missing_leaflets: list[str],
) -> None:
    await set_session_data(session_id, "drugs_found", drugs_found)
    await set_session_data(session_id, "missing_leaflets", missing_leaflets)


async def get_upload_result(session_id: str) -> tuple[list[str], list[str]]:
    drugs_found = await get_session_data(session_id, "drugs_found") or []
    missing_leaflets = await get_session_data(session_id, "missing_leaflets") or []
    return drugs_found, missing_leaflets


# ── Upload job state ─────────────────────────────────────────────────────────

_JOB_TTL = 3600  # jobs expire after 1 h regardless of session TTL


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


async def save_job_status(
    job_id: str,
    session_id: str,
    status: str,
    drugs_found: list[str] | None = None,
    missing_leaflets: list[str] | None = None,
    error: str | None = None,
) -> None:
    r = _get_redis()
    key = _job_key(job_id)
    payload: dict[str, str] = {
        "session_id": json.dumps(session_id),
        "status": json.dumps(status),
        "drugs_found": json.dumps(drugs_found or []),
        "missing_leaflets": json.dumps(missing_leaflets or []),
        "error": json.dumps(error),
    }
    await r.hset(key, mapping=payload)
    await r.expire(key, _JOB_TTL)


async def get_job_status(job_id: str) -> dict | None:
    r = _get_redis()
    key = _job_key(job_id)
    raw = await r.hgetall(key)
    if not raw:
        return None
    return {k: json.loads(v) for k, v in raw.items()}
