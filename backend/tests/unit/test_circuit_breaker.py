"""Unit tests for RedisCircuitBreaker (state shared via fakeredis fixture).

The seeded-state tests below exercise the read-side state machine
(allow_request, record_success, fail-open behaviour) by writing the
failure/last-fail keys directly. The Lua-path tests at the bottom drive
record_failure() itself, executing the production EVAL script through
fakeredis (lupa-backed).
"""

import time
from unittest.mock import patch

import app.services.session_store as _ss
from app.services.circuit_breaker import RedisCircuitBreaker


async def _seed(name: str, failures: int, last_fail: float) -> None:
    r = _ss._get_redis()
    await r.set(f"cb:{name}:failures", failures)
    await r.set(f"cb:{name}:last_fail", str(last_fail))


async def test_closed_by_default():
    cb = RedisCircuitBreaker("test-default", failure_threshold=3)
    assert await cb.allow_request() is True


async def test_stays_closed_below_threshold():
    cb = RedisCircuitBreaker("test-below", failure_threshold=3)
    await _seed("test-below", failures=2, last_fail=time.time())
    assert await cb.allow_request() is True


async def test_opens_at_threshold():
    cb = RedisCircuitBreaker("test-open", failure_threshold=2, cooldown_seconds=60)
    await _seed("test-open", failures=2, last_fail=time.time())
    assert await cb.allow_request() is False


async def test_success_resets_failures():
    cb = RedisCircuitBreaker("test-reset", failure_threshold=2, cooldown_seconds=60)
    await _seed("test-reset", failures=2, last_fail=time.time())
    assert await cb.allow_request() is False

    await cb.record_success()
    assert await cb.allow_request() is True


async def test_half_open_after_cooldown():
    cb = RedisCircuitBreaker("test-halfopen", failure_threshold=1, cooldown_seconds=60)
    await _seed("test-halfopen", failures=1, last_fail=time.time())
    assert await cb.allow_request() is False

    # Pretend the cooldown has elapsed.
    with patch(
        "app.services.circuit_breaker.time.time",
        return_value=time.time() + 61,
    ):
        assert await cb.allow_request() is True  # probe allowed


async def test_fails_open_when_redis_unavailable():
    cb = RedisCircuitBreaker("test-noredis", failure_threshold=1)
    old = _ss._redis
    _ss._redis = None  # _get_redis() now raises RuntimeError
    try:
        assert await cb.allow_request() is True
    finally:
        _ss._redis = old


# ---------------------------------------------------------------------------
# record_failure — the real Lua script, executed via fakeredis (lupa)
# ---------------------------------------------------------------------------


async def test_record_failure_increments_counter_and_stamps_time():
    cb = RedisCircuitBreaker("test-lua-incr", failure_threshold=5)
    before = time.time()
    await cb.record_failure()
    await cb.record_failure()

    r = _ss._get_redis()
    assert int(await r.get("cb:test-lua-incr:failures")) == 2
    last_fail = float(await r.get("cb:test-lua-incr:last_fail"))
    assert before <= last_fail <= time.time()


async def test_record_failure_sets_ttl_on_both_keys():
    cb = RedisCircuitBreaker("test-lua-ttl", failure_threshold=3, cooldown_seconds=60.0)
    await cb.record_failure()

    r = _ss._get_redis()
    # TTL is cooldown_seconds * 10 so stuck state self-heals.
    for key in ("cb:test-lua-ttl:failures", "cb:test-lua-ttl:last_fail"):
        ttl = await r.ttl(key)
        assert 0 < ttl <= 600


async def test_opens_after_threshold_record_failure_calls():
    cb = RedisCircuitBreaker("test-lua-open", failure_threshold=2, cooldown_seconds=60)
    assert await cb.allow_request() is True

    await cb.record_failure()
    assert await cb.allow_request() is True  # 1/2 — still CLOSED

    await cb.record_failure()
    assert await cb.allow_request() is False  # 2/2 — OPEN


async def test_half_open_probe_failure_reopens():
    cb = RedisCircuitBreaker(
        "test-lua-reopen", failure_threshold=1, cooldown_seconds=60
    )
    await cb.record_failure()
    assert await cb.allow_request() is False  # OPEN

    with patch(
        "app.services.circuit_breaker.time.time",
        return_value=time.time() + 61,
    ):
        assert await cb.allow_request() is True  # HALF_OPEN probe

    # The probe fails: record_failure restamps last_fail to now → OPEN again.
    await cb.record_failure()
    assert await cb.allow_request() is False


async def test_success_after_failures_closes_and_clears_keys():
    cb = RedisCircuitBreaker("test-lua-close", failure_threshold=1)
    await cb.record_failure()
    assert await cb.allow_request() is False

    await cb.record_success()
    assert await cb.allow_request() is True

    r = _ss._get_redis()
    assert await r.get("cb:test-lua-close:failures") is None
    assert await r.get("cb:test-lua-close:last_fail") is None
