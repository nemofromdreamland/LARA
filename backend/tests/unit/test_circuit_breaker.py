import time
from unittest.mock import patch

from app.services.circuit_breaker import CircuitBreaker


def test_closed_by_default():
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
    assert cb.allow_request() is True
    assert cb.is_open is False


def test_stays_closed_below_threshold():
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
    cb.record_failure()
    cb.record_failure()
    assert cb.allow_request() is True
    assert cb.is_open is False


def test_opens_at_threshold():
    cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60)
    cb.record_failure()
    cb.record_failure()
    cb.record_failure()
    assert cb.allow_request() is False
    assert cb.is_open is True


def test_success_resets_to_closed():
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=60)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open is True
    cb.record_success()
    assert cb.allow_request() is True
    assert cb.is_open is False


def test_half_open_after_cooldown():
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=1)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open is True

    # Simulate cooldown elapsed by patching time.monotonic
    future = time.monotonic() + 2
    with patch("app.services.circuit_breaker.time.monotonic", return_value=future):
        assert cb.allow_request() is True  # HALF_OPEN lets request through
        assert cb.is_open is False


def test_half_open_failure_reopens():
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=1)
    cb.record_failure()
    cb.record_failure()

    future = time.monotonic() + 2
    with patch("app.services.circuit_breaker.time.monotonic", return_value=future):
        assert cb.allow_request() is True  # probe allowed (HALF_OPEN)
        cb.record_failure()  # probe fails
        assert cb.is_open is True  # back to OPEN


def test_half_open_success_closes():
    cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=1)
    cb.record_failure()
    cb.record_failure()

    future = time.monotonic() + 2
    with patch("app.services.circuit_breaker.time.monotonic", return_value=future):
        cb.record_success()
        assert cb.allow_request() is True
        assert cb.is_open is False
