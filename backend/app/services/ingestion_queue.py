"""Durable ingestion queue backed by a Redis Stream + consumer group.

Replaces the previous in-process FastAPI ``BackgroundTasks`` ingestion, which
lost jobs on a worker crash/restart. A Redis Stream gives at-least-once
delivery: the producer ``XADD``s a job, a per-worker consumer reads it via a
consumer group, runs the existing ``run_ingestion`` pipeline, and ``XACK``s on
success. A job a worker dies mid-processing stays in that consumer's
pending-entries list and is reclaimed by a healthy worker via ``XAUTOCLAIM``
(crash recovery). After ``ingestion_max_attempts`` deliveries a poison job is
moved to a dead-letter stream and its job row marked ``failed``.

Only data crosses the queue (job_id, session_id, extracted text, request_id);
the consumer supplies its own process-local ``embed_executor`` — the executor
and ML models are never serialized.
"""

import asyncio
import logging
import os
from uuid import uuid4

import redis.exceptions
from prometheus_client import Counter

from app.config import settings
from app.services.ingestion import run_ingestion
from app.services.session_store import get_redis, save_job_status
from app.utils import request_id_var

logger = logging.getLogger(__name__)

_STREAM = "ingestion:stream"
_GROUP = "ingestion:workers"
_DLQ = "ingestion:dlq"

# Consumer name used by the test drain helpers; real workers pass _consumer_name().
_DEFAULT_CONSUMER = "lara-consumer"

_INGESTION_JOBS = Counter(
    "lara_ingestion_jobs_total",
    "Ingestion job lifecycle events",
    ["event"],  # enqueued | processed | failed | reclaimed | dead_lettered
)


def _consumer_name() -> str:
    """Unique per gunicorn worker (PID) so each owns a distinct PEL."""
    return f"worker-{os.getpid()}-{uuid4().hex[:6]}"


def _decode(fields: dict) -> dict:
    """Decode a stream entry back to the run_ingestion arguments.

    Raises KeyError if a required field is missing (treated as a poison message).
    """
    return {
        "job_id": fields["job_id"],
        "session_id": fields["session_id"],
        "text": fields["text"],
        "request_id": fields.get("request_id", "no-request"),
    }


async def ensure_consumer_group() -> None:
    """Create the stream + consumer group if absent (idempotent)."""
    r = get_redis()
    try:
        await r.xgroup_create(_STREAM, _GROUP, id="0", mkstream=True)
    except redis.exceptions.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def enqueue_ingestion(
    job_id: str, session_id: str, text: str, request_id: str
) -> str:
    """Producer: append an ingestion job to the stream. Returns the message id."""
    r = get_redis()
    msg_id = await r.xadd(
        _STREAM,
        {
            "job_id": job_id,
            "session_id": session_id,
            "text": text,
            "request_id": request_id,
        },
        maxlen=settings.ingestion_stream_maxlen,
        approximate=True,
    )
    _INGESTION_JOBS.labels(event="enqueued").inc()
    logger.info(
        "enqueued ingestion job %s for session %s",
        job_id,
        session_id,
        extra={"request_id": request_id},
    )
    return msg_id


async def _dead_letter(
    r, msg_id: str, fields: dict, error: str, *, job_id, session_id
) -> None:
    """Move a message to the DLQ stream, ACK the original, mark the job failed."""
    payload = {k: str(v) for k, v in fields.items()}
    payload["error"] = error[:500]
    await r.xadd(
        _DLQ, payload, maxlen=settings.ingestion_stream_maxlen, approximate=True
    )
    await r.xack(_STREAM, _GROUP, msg_id)
    if job_id and session_id:
        try:
            await save_job_status(job_id, session_id, "failed", error=error[:500])
        except Exception:
            logger.exception(
                "could not mark job %s failed while dead-lettering", job_id
            )
    _INGESTION_JOBS.labels(event="dead_lettered").inc()


async def handle_message(
    msg_id: str, fields: dict, embed_executor, *, delivery_count: int
) -> None:
    """Run one ingestion job.

    On success: XACK. On unexpected/infra failure: leave the message pending
    (no XACK) so a later XAUTOCLAIM retries it. Once a message has been
    delivered more than ``ingestion_max_attempts`` times it is dead-lettered.
    Terminal logical outcomes (no drugs found, per-drug missing leaflets) are
    handled inside run_ingestion, which returns normally → the message is ACKed
    and not retried.
    """
    r = get_redis()
    try:
        data = _decode(fields)
    except Exception as exc:
        logger.error("malformed ingestion message %s: %s", msg_id, exc)
        await _dead_letter(
            r,
            msg_id,
            fields,
            f"malformed message: {exc}",
            job_id=fields.get("job_id"),
            session_id=fields.get("session_id"),
        )
        return

    rid = data["request_id"]
    request_id_var.set(rid)

    if delivery_count > settings.ingestion_max_attempts:
        logger.error(
            "ingestion job %s exceeded %d attempts — dead-lettering",
            data["job_id"],
            settings.ingestion_max_attempts,
            extra={"request_id": rid},
        )
        await _dead_letter(
            r,
            msg_id,
            fields,
            f"exceeded {settings.ingestion_max_attempts} delivery attempts",
            job_id=data["job_id"],
            session_id=data["session_id"],
        )
        return

    try:
        await run_ingestion(
            data["job_id"], data["session_id"], data["text"], rid, embed_executor
        )
        await r.xack(_STREAM, _GROUP, msg_id)
        _INGESTION_JOBS.labels(event="processed").inc()
    except Exception as exc:
        # Infra/unexpected failure: leave pending for reclaim instead of ACKing.
        _INGESTION_JOBS.labels(event="failed").inc()
        logger.exception(
            "ingestion attempt %d failed for job %s: %s",
            delivery_count,
            data["job_id"],
            exc,
            extra={"request_id": rid},
        )


async def read_new(
    embed_executor,
    *,
    consumer_name: str = _DEFAULT_CONSUMER,
    block_ms: int | None = None,
    count: int = 10,
) -> int:
    """Read and handle never-before-delivered messages. Returns count handled.

    ``block_ms=None`` returns immediately when the stream is empty — used by
    tests to drain. The consumer loop passes ``settings.ingestion_block_ms``.
    """
    await ensure_consumer_group()
    r = get_redis()
    resp = await r.xreadgroup(
        _GROUP, consumer_name, {_STREAM: ">"}, count=count, block=block_ms
    )
    if not resp:
        return 0
    handled = 0
    for _stream, entries in resp:
        for msg_id, msg_fields in entries:
            await handle_message(msg_id, msg_fields, embed_executor, delivery_count=1)
            handled += 1
    return handled


async def claim_stale(
    embed_executor,
    *,
    consumer_name: str = _DEFAULT_CONSUMER,
    min_idle_ms: int | None = None,
    count: int = 10,
) -> int:
    """Reclaim and handle messages idle longer than the threshold (crash recovery).

    Returns the number of reclaimed messages handled.
    """
    if min_idle_ms is None:
        min_idle_ms = settings.ingestion_reclaim_idle_seconds * 1000
    await ensure_consumer_group()
    r = get_redis()
    result = await r.xautoclaim(
        _STREAM,
        _GROUP,
        consumer_name,
        min_idle_time=min_idle_ms,
        start_id="0-0",
        count=count,
    )
    claimed = result[1] if len(result) > 1 else []
    deleted = result[2] if len(result) > 2 else []
    # Entries XDEL'd from the stream but still pending: ACK to clear the PEL.
    for did in deleted:
        await r.xack(_STREAM, _GROUP, did)
    if not claimed:
        return 0
    # XAUTOCLAIM doesn't return delivery counts; read them from the PEL.
    pending = await r.xpending_range(
        _STREAM, _GROUP, min="-", max="+", count=max(count, len(claimed))
    )
    counts = {p["message_id"]: p["times_delivered"] for p in pending}
    handled = 0
    for msg_id, msg_fields in claimed:
        _INGESTION_JOBS.labels(event="reclaimed").inc()
        await handle_message(
            msg_id,
            msg_fields,
            embed_executor,
            delivery_count=counts.get(msg_id, 1),
        )
        handled += 1
    return handled


async def run_consumer(embed_executor, consumer_name: str) -> None:
    """Per-worker consumer loop: reclaim stale jobs, then read new ones.

    Started from the app lifespan (mirrors _cleanup_loop) and cancelled on
    shutdown. A job in flight at cancellation is left pending and reclaimed by a
    healthy worker — never silently dropped.
    """
    await ensure_consumer_group()
    logger.info("ingestion consumer %s started", consumer_name)
    while True:  # pragma: no cover - body exercised via claim_stale/read_new tests
        try:
            await claim_stale(embed_executor, consumer_name=consumer_name)
            await read_new(
                embed_executor,
                consumer_name=consumer_name,
                block_ms=settings.ingestion_block_ms,
            )
        except asyncio.CancelledError:
            logger.info("ingestion consumer %s stopping", consumer_name)
            raise
        except Exception:
            logger.exception("ingestion consumer loop error — backing off 1s")
            await asyncio.sleep(1)
