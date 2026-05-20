from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_all_healthy(client: TestClient):
    with (
        patch("app.services.vector_store.ping", AsyncMock(return_value=True)),
        patch("app.services.embedder.is_model_loaded", return_value=True),
        patch("app.config.settings.groq_api_key", "sk-test"),
        patch("app.config.settings.cerebras_api_key", ""),
    ):
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["components"]["chroma"]["status"] == "ok"
    assert data["components"]["embedder"]["status"] == "ok"
    assert data["components"]["llm"]["status"] == "ok"


def test_chroma_ping_raises(client: TestClient):
    with (
        patch("app.services.vector_store.ping", AsyncMock(side_effect=Exception("connection refused"))),
        patch("app.services.embedder.is_model_loaded", return_value=True),
        patch("app.config.settings.groq_api_key", "sk-test"),
        patch("app.config.settings.cerebras_api_key", ""),
    ):
        response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["components"]["chroma"]["status"] == "unavailable"


def test_model_not_loaded(client: TestClient):
    with (
        patch("app.services.vector_store.ping", AsyncMock(return_value=True)),
        patch("app.services.embedder.is_model_loaded", return_value=False),
        patch("app.config.settings.groq_api_key", "sk-test"),
        patch("app.config.settings.cerebras_api_key", ""),
    ):
        response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["components"]["embedder"]["status"] == "degraded"


def test_no_api_keys(client: TestClient):
    with (
        patch("app.services.vector_store.ping", AsyncMock(return_value=True)),
        patch("app.services.embedder.is_model_loaded", return_value=True),
        patch("app.config.settings.groq_api_key", ""),
        patch("app.config.settings.cerebras_api_key", ""),
    ):
        response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["components"]["llm"]["status"] == "degraded"
