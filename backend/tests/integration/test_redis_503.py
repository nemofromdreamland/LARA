"""Verify that Redis ConnectionError propagates as HTTP 503 on key routes."""

from unittest.mock import AsyncMock

import pytest
import redis.exceptions
from fastapi.testclient import TestClient

import app.services.session_store as _ss
from app.config import settings
from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, headers={"X-API-Key": settings.lara_api_key})


def _broken_redis():
    """AsyncMock that raises ConnectionError on every Redis operation."""
    err = redis.exceptions.ConnectionError("connection refused")
    mock = AsyncMock()
    mock.hset = AsyncMock(side_effect=err)
    mock.expire = AsyncMock(side_effect=err)
    mock.hget = AsyncMock(side_effect=err)
    mock.hgetall = AsyncMock(side_effect=err)
    mock.exists = AsyncMock(side_effect=err)
    mock.ping = AsyncMock(side_effect=err)
    return mock


def test_post_session_redis_down_returns_503(client: TestClient):
    _ss._redis = _broken_redis()
    response = client.post("/session")
    assert response.status_code == 503
    assert "Storage unavailable" in response.json()["detail"]


def test_get_job_status_redis_down_returns_503(client: TestClient):
    _ss._redis = _broken_redis()
    response = client.get(
        "/upload/status/00000000-0000-0000-0000-000000000000",
        params={"session_id": "a" * 36},
    )
    assert response.status_code == 503
    assert "Storage unavailable" in response.json()["detail"]


def test_post_chat_redis_down_returns_503(client: TestClient):
    _ss._redis = _broken_redis()
    response = client.post(
        "/chat",
        json={
            "session_id": "a" * 36,
            "question": "What are the side effects?",
        },
    )
    assert response.status_code == 503
    assert "Storage unavailable" in response.json()["detail"]
