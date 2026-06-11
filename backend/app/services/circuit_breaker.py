import logging
import time

logger = logging.getLogger(__name__)

# Atomically increments the failure counter, stamps the last-failure time, and
# sets a TTL on both keys so stuck state self-heals after cooldown_seconds * 10.
# KEYS[1] = failures key, KEYS[2] = last_fail key
# ARGV[1] = current timestamp (float as string), ARGV[2] = TTL in seconds
_RECORD_FAILURE_LUA = """
local failures = redis.call('INCR', KEYS[1])
redis.call('SET', KEYS[2], ARGV[1])
local ttl = tonumber(ARGV[2])
redis.call('EXPIRE', KEYS[1], ttl)
redis.call('EXPIRE', KEYS[2], ttl)
return failures
"""


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
            ttl = int(self._cooldown * 10)
            failures = await r.eval(
                _RECORD_FAILURE_LUA,
                2,
                self._key_failures,
                self._key_last_fail,
                str(time.time()),
                str(ttl),
            )
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
