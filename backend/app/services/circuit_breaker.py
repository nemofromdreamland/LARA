import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class RedisCircuitBreaker:
    """Circuit breaker whose state is stored in Redis.

    Sharing state via Redis means all uvicorn workers see the same failure
    count and trip/reset together, giving consistent LLM routing behaviour
    in multi-worker deployments.

    Falls back to allowing requests (open) if Redis is unavailable — this is
    the safer default: worst case, all workers hammer a degraded provider
    together, but the per-provider retry logic and the Cerebras fallback
    still apply.

    State machine (same as CircuitBreaker):
      CLOSED    → (failure_threshold consecutive failures) → OPEN
      OPEN      → (cooldown_seconds elapsed)               → HALF_OPEN
      HALF_OPEN → (probe succeeds)                         → CLOSED
      HALF_OPEN → (probe fails)                            → OPEN
    """

    def __init__(
        self, name: str, failure_threshold: int = 3, cooldown_seconds: float = 60.0
    ):
        self._name = name
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._key_failures = f"cb:{name}:failures"
        self._key_last_fail = f"cb:{name}:last_fail"

    def _get_redis(self):
        from app.services.session_store import _get_redis as _session_get_redis

        return _session_get_redis()

    async def allow_request(self) -> bool:
        """Return True if the circuit should let a request through."""
        try:
            r = self._get_redis()
            failures_raw = await r.get(self._key_failures)
            failures = int(failures_raw) if failures_raw else 0
            if failures < self._threshold:
                return True
            last_fail_raw = await r.get(self._key_last_fail)
            if last_fail_raw is None:
                return True
            elapsed = time.time() - float(last_fail_raw)
            if elapsed >= self._cooldown:
                return True  # HALF_OPEN — let one probe through
            logger.warning(
                "Redis circuit breaker OPEN for %s — request blocked", self._name
            )
            return False
        except Exception:
            return True  # fail open if Redis is unavailable

    async def record_success(self) -> None:
        try:
            r = self._get_redis()
            failures_raw = await r.get(self._key_failures)
            if failures_raw and int(failures_raw) > 0:
                logger.info("Redis circuit breaker reset to CLOSED for %s", self._name)
            await r.delete(self._key_failures, self._key_last_fail)
        except Exception:
            pass

    async def record_failure(self) -> None:
        try:
            r = self._get_redis()
            failures = await r.incr(self._key_failures)
            await r.set(self._key_last_fail, time.time())
            logger.warning(
                "Redis circuit breaker: failure %d/%d for %s",
                failures,
                self._threshold,
                self._name,
            )
        except Exception:
            pass

    @property
    async def is_open(self) -> bool:
        return not await self.allow_request()


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
