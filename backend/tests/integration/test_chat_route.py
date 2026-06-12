import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models.schemas import ChatResponse, Source

MOCK_ANSWER = ChatResponse(
    answer="According to the Warnings section, do not use in pregnancy.",
    sources=[Source(drug_name="lisinopril", section="warnings")],
)

EMPTY_ANSWER = ChatResponse(
    answer=(
        "I couldn't find relevant information in the uploaded leaflets "
        "for your question. Drugs indexed: none. "
        "Drugs with no leaflet found: none. "
        "Try rephrasing your question or ask about a specific section "
        "(e.g. 'warnings', 'dosage', 'interactions')."
    ),
    sources=[],
)


@pytest.fixture
def session_id(client: TestClient) -> str:
    resp = client.post("/session")
    assert resp.status_code == 200
    return resp.json()["session_id"]


@patch("app.routes.chat.answer", new_callable=AsyncMock)
def test_chat_returns_answer_and_sources(
    mock_answer, client: TestClient, session_id: str
):
    mock_answer.return_value = MOCK_ANSWER

    response = client.post(
        "/chat",
        json={"session_id": session_id, "question": "Is lisinopril safe in pregnancy?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert data["answer"] == MOCK_ANSWER.answer
    assert data["sources"][0]["drug_name"] == "lisinopril"
    assert data["sources"][0]["section"] == "warnings"


@patch("app.routes.chat.answer", new_callable=AsyncMock)
def test_chat_passes_session_id_and_question(
    mock_answer, client: TestClient, session_id: str
):
    mock_answer.return_value = MOCK_ANSWER

    client.post(
        "/chat",
        json={"session_id": session_id, "question": "What is the dosage?"},
    )
    mock_answer.assert_called_once_with(session_id, "What is the dosage?", [], None)


@patch("app.routes.chat.answer", new_callable=AsyncMock)
def test_chat_empty_context_returns_not_available(
    mock_answer, client: TestClient, session_id: str
):
    mock_answer.return_value = EMPTY_ANSWER

    response = client.post(
        "/chat",
        json={"session_id": session_id, "question": "Any question"},
    )
    assert response.status_code == 200
    assert "I couldn't find relevant information" in response.json()["answer"]
    assert response.json()["sources"] == []


def test_chat_missing_fields_returns_422(client: TestClient):
    # Missing 'question' — Pydantic validation fires before session check
    response = client.post("/chat", json={"session_id": "a" * 36})
    assert response.status_code == 422


# rerank() is intentionally NOT patched here, so the real cross-encoder runs
# inside the pipeline (reranker_enabled defaults to True) — hence the ml mark.
@pytest.mark.ml
@patch("app.services.rag_pipeline.retrieve", new_callable=AsyncMock)
@patch("app.services.rag_pipeline.embed", new_callable=AsyncMock)
@patch("app.services.rag_pipeline.generate", new_callable=AsyncMock)
def test_rag_pipeline_integration(
    mock_generate, mock_embed, mock_retrieve, client: TestClient, session_id: str
):
    """Test the full pipeline: embed → retrieve → generate → response."""
    mock_embed.return_value = [[0.1] * 768]
    mock_retrieve.return_value = [
        {
            "text": "Take once daily.",
            "drug_name": "lisinopril",
            "section": "dosage",
            "distance": 0.1,
        }
    ]
    mock_generate.return_value = "According to the Dosage section, take once daily."

    response = client.post(
        "/chat",
        json={
            "session_id": session_id,
            "question": "How often should I take lisinopril?",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "dosage" in data["answer"].lower() or "daily" in data["answer"].lower()
    assert data["sources"][0]["drug_name"] == "lisinopril"


@patch("app.services.rag_pipeline.retrieve", new_callable=AsyncMock)
@patch("app.services.rag_pipeline.embed", new_callable=AsyncMock)
async def test_rag_pipeline_no_chunks_skips_llm(
    mock_embed, mock_retrieve, client: TestClient, session_id: str
):
    """When retrieve returns nothing, LLM must not be called."""
    mock_embed.return_value = [[0.1] * 768]
    mock_retrieve.return_value = []

    with patch(
        "app.services.rag_pipeline.generate", new_callable=AsyncMock
    ) as mock_gen:
        response = client.post(
            "/chat",
            json={"session_id": session_id, "question": "Any question"},
        )
        mock_gen.assert_not_called()

    assert response.status_code == 200
    assert "I couldn't find relevant information" in response.json()["answer"]


# ---------------------------------------------------------------------------
# /chat/stream tests
# ---------------------------------------------------------------------------


def _make_stream(*payloads: str):
    """Return an async generator that yields the given SSE payloads."""

    async def _gen() -> AsyncGenerator[str, None]:
        for p in payloads:
            yield p

    return _gen()


@patch("app.routes.chat.answer_stream")
def test_chat_stream_yields_tokens(mock_stream, client: TestClient, session_id: str):
    sources_payload = "[SOURCES]" + json.dumps({"sources": []})
    mock_stream.return_value = _make_stream(
        "Hello ", "world", sources_payload, "[DONE]"
    )

    with client.stream(
        "POST",
        "/chat/stream",
        json={"session_id": session_id, "question": "What are the side effects?"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = b"".join(resp.iter_bytes()).decode()

    assert "event: token" in raw
    assert json.dumps("Hello ") in raw
    assert json.dumps("world") in raw
    assert "event: done" in raw


@patch("app.routes.chat.answer_stream")
def test_chat_stream_includes_sources_event(
    mock_stream, client: TestClient, session_id: str
):
    sources = [{"drug_name": "lisinopril", "section": "warnings"}]
    sources_payload = "[SOURCES]" + json.dumps({"sources": sources})
    mock_stream.return_value = _make_stream("Answer.", sources_payload, "[DONE]")

    with client.stream(
        "POST",
        "/chat/stream",
        json={"session_id": session_id, "question": "Is it safe during pregnancy?"},
    ) as resp:
        raw = b"".join(resp.iter_bytes()).decode()

    assert "event: sources" in raw
    assert "lisinopril" in raw


@patch("app.routes.chat.answer_stream")
def test_chat_stream_missing_fields_returns_422(mock_stream, client: TestClient):
    # Missing 'question' — Pydantic validation fires before session check
    response = client.post("/chat/stream", json={"session_id": "a" * 36})
    assert response.status_code == 422
