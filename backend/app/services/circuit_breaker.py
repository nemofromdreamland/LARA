import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class _State(Enum):
    CLOSED = "closed"  # normal — requests allowed
    OPEN = "open"  # too many failures — requests blocked
    HALF_OPEN = "half_open"  # cooldown elapsed — one probe allowed


class CircuitBreaker:
    """Simple in-process circuit breaker.

    State machine:
      CLOSED  →  (failure_threshold consecutive failures)  →  OPEN
      OPEN    →  (cooldown_seconds elapsed)                →  HALF_OPEN
      HALF_OPEN  →  (probe succeeds)                       →  CLOSED
      HALF_OPEN  →  (probe fails)                          →  OPEN
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60.0):
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._failures = 0
        self._last_failure_time: float | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def allow_request(self) -> bool:
        """Return True if the circuit should let a request through."""
        state = self._state()
        if state == _State.OPEN:
            logger.warning("Circuit breaker OPEN — request blocked")
            return False
        return True  # CLOSED or HALF_OPEN

    def record_success(self) -> None:
        if self._failures > 0:
            logger.info("Circuit breaker reset to CLOSED after success")
        self._failures = 0
        self._last_failure_time = None

    def record_failure(self) -> None:
        self._failures += 1
        self._last_failure_time = time.monotonic()
        state = self._state()
        logger.warning(
            "Circuit breaker: failure %d/%d (state=%s)",
            self._failures,
            self._threshold,
            state.value,
        )

    @property
    def is_open(self) -> bool:
        return self._state() == _State.OPEN

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _state(self) -> _State:
        if self._failures < self._threshold:
            return _State.CLOSED
        if self._last_failure_time is None:
            return _State.CLOSED
        elapsed = time.monotonic() - self._last_failure_time
        if elapsed >= self._cooldown:
            return _State.HALF_OPEN
        return _State.OPEN
