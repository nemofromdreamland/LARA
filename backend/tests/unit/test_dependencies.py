import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

_VALID_KEY = settings.lara_api_key


@pytest.fixture
def authed_client() -> TestClient:
    return TestClient(app, headers={"X-API-Key": _VALID_KEY})


@pytest.fixture
def bare_client() -> TestClient:
    return TestClient(app)


def test_valid_key_passes(authed_client: TestClient):
    resp = authed_client.post("/session")
    assert resp.status_code == 200


def test_missing_key_returns_401(bare_client: TestClient):
    resp = bare_client.post("/session")
    assert resp.status_code == 401


def test_wrong_key_returns_401(bare_client: TestClient):
    resp = bare_client.post("/session", headers={"X-API-Key": "definitely-wrong"})
    assert resp.status_code == 401


def test_health_not_guarded(bare_client: TestClient):
    # /health must be reachable without an API key
    resp = bare_client.get("/health")
    assert resp.status_code != 401
