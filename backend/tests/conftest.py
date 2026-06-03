import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

import app.services.session_store as _ss
from app.config import settings
from app.limiter import limiter as _original_limiter
from app.main import app


@pytest.fixture(autouse=True)
def _fake_redis():
    """Inject a fresh in-memory FakeRedis for every test."""
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    _ss._redis = fake
    yield
    _ss._redis = None


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Swap the original limiter's storage to in-memory for each test.

    The @limiter.limit() decorator captures a reference to `limiter._limiter`
    for inline checks (not just the middleware path), so we must patch the
    storage on the existing limiter object — not replace app.state.limiter.
    This avoids real Redis connections and gives isolated counters per test.
    """
    from limits.storage import MemoryStorage
    from limits.strategies import FixedWindowRateLimiter

    new_storage = MemoryStorage()
    old_storage = _original_limiter._storage
    old_strat = _original_limiter._limiter

    _original_limiter._storage = new_storage
    _original_limiter._limiter = FixedWindowRateLimiter(new_storage)

    yield

    _original_limiter._storage = old_storage
    _original_limiter._limiter = old_strat


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, headers={"X-API-Key": settings.lara_api_key})
