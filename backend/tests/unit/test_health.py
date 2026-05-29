from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _cb_closed():
    """Return a mock circuit breaker whose allow_request() returns True (closed)."""
    cb = AsyncMock()
    cb.allow_request = AsyncMock(return_value=True)
    return cb


def _cb_open():
    """Return a mock circuit breaker whose allow_request() returns False (open)."""
    cb = AsyncMock()
    cb.allow_request = AsyncMock(return_value=False)
    return cb


def test_all_healthy(client: TestClient):
    with (
        patch("app.services.vector_store.ping", AsyncMock(return_value=True)),
        patch("app.services.embedder.is_model_loaded", return_value=True),
        patch("app.config.settings.groq_api_key", "sk-test"),
        patch("app.config.settings.cerebras_api_key", ""),
        patch("app.routes.health._groq_breaker", _cb_closed()),
        patch("app.routes.health._cerebras_breaker", _cb_closed()),
    ):
        response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["components"]["chroma"]["status"] == "ok"
    assert data["components"]["embedder"]["status"] == "ok"
    assert data["components"]["llm"]["status"] == "ok"
    assert data["components"]["llm_routing"]["status"] == "ok"


def test_chroma_ping_raises(client: TestClient):
    with (
        patch(
            "app.services.vector_store.ping",
            AsyncMock(side_effect=Exception("connection refused")),
        ),
        patch("app.services.embedder.is_model_loaded", return_value=True),
        patch("app.config.settings.groq_api_key", "sk-test"),
        patch("app.config.settings.cerebras_api_key", ""),
        patch("app.routes.health._groq_breaker", _cb_closed()),
        patch("app.routes.health._cerebras_breaker", _cb_closed()),
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
        patch("app.routes.health._groq_breaker", _cb_closed()),
        patch("app.routes.health._cerebras_breaker", _cb_closed()),
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
        patch("app.routes.health._groq_breaker", _cb_closed()),
        patch("app.routes.health._cerebras_breaker", _cb_closed()),
    ):
        response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["components"]["llm"]["status"] == "degraded"


def test_llm_routing_groq_open(client: TestClient):
    with (
        patch("app.services.vector_store.ping", AsyncMock(return_value=True)),
        patch("app.services.embedder.is_model_loaded", return_value=True),
        patch("app.config.settings.groq_api_key", "sk-test"),
        patch("app.config.settings.cerebras_api_key", "ck-test"),
        patch("app.routes.health._groq_breaker", _cb_open()),
        patch("app.routes.health._cerebras_breaker", _cb_closed()),
    ):
        response = client.get("/health")
    data = response.json()
    assert data["components"]["llm_routing"]["status"] == "degraded"
    assert data["components"]["llm_routing"]["detail"] == "groq_open"


def test_llm_routing_both_open(client: TestClient):
    with (
        patch("app.services.vector_store.ping", AsyncMock(return_value=True)),
        patch("app.services.embedder.is_model_loaded", return_value=True),
        patch("app.config.settings.groq_api_key", "sk-test"),
        patch("app.config.settings.cerebras_api_key", "ck-test"),
        patch("app.routes.health._groq_breaker", _cb_open()),
        patch("app.routes.health._cerebras_breaker", _cb_open()),
    ):
        response = client.get("/health")
    data = response.json()
    assert data["components"]["llm_routing"]["status"] == "degraded"
    assert data["components"]["llm_routing"]["detail"] == "both_open"
