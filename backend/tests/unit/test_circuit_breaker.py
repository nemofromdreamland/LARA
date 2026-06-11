"""Unit tests for RedisCircuitBreaker (state shared via fakeredis fixture).

fakeredis cannot execute the Lua script behind record_failure() (requires
lupa), so breaker state is seeded directly into the failure/last-fail keys;
the tests exercise the read-side state machine (allow_request, record_success,
fail-open behaviour).
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
    assert await cb.is_open is False


async def test_stays_closed_below_threshold():
    cb = RedisCircuitBreaker("test-below", failure_threshold=3)
    await _seed("test-below", failures=2, last_fail=time.time())
    assert await cb.allow_request() is True


async def test_opens_at_threshold():
    cb = RedisCircuitBreaker("test-open", failure_threshold=2, cooldown_seconds=60)
    await _seed("test-open", failures=2, last_fail=time.time())
    assert await cb.allow_request() is False
    assert await cb.is_open is True


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
