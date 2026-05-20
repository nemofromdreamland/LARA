import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

import app.services.session_store as _ss
from app.main import app


@pytest.fixture(autouse=True)
def _fake_redis():
    """Inject a fresh in-memory FakeRedis for every test.

    Patches both the module-level _redis reference and the init/close lifecycle
    functions so the FastAPI lifespan never touches a real Redis instance.
    """
    fake = fakeredis.aioredis.FakeRedis(decode_responses=True)
    _ss._redis = fake
    yield
    _ss._redis = None


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)
