from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.models.schemas import ChatResponse, Source

MOCK_ANSWER = ChatResponse(
    answer="According to the Warnings section, do not use in pregnancy.",
    sources=[Source(drug_name="lisinopril", section="warnings")],
)

EMPTY_ANSWER = ChatResponse(
    answer="This information is not available in the provided leaflets.",
    sources=[],
)


@patch("app.routes.chat.answer", new_callable=AsyncMock)
def test_chat_returns_answer_and_sources(mock_answer, client: TestClient):
    mock_answer.return_value = MOCK_ANSWER

    response = client.post(
        "/chat",
        json={"session_id": "sess-1", "question": "Is lisinopril safe in pregnancy?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert data["answer"] == MOCK_ANSWER.answer
    assert data["sources"][0]["drug_name"] == "lisinopril"
    assert data["sources"][0]["section"] == "warnings"


@patch("app.routes.chat.answer", new_callable=AsyncMock)
def test_chat_passes_session_id_and_question(mock_answer, client: TestClient):
    mock_answer.return_value = MOCK_ANSWER

    client.post(
        "/chat",
        json={"session_id": "my-session", "question": "What is the dosage?"},
    )
    mock_answer.assert_called_once_with("my-session", "What is the dosage?")


@patch("app.routes.chat.answer", new_callable=AsyncMock)
def test_chat_empty_context_returns_not_available(mock_answer, client: TestClient):
    mock_answer.return_value = EMPTY_ANSWER

    response = client.post(
        "/chat",
        json={"session_id": "empty-sess", "question": "Any question"},
    )
    assert response.status_code == 200
    assert "not available" in response.json()["answer"]
    assert response.json()["sources"] == []


def test_chat_missing_fields_returns_422(client: TestClient):
    response = client.post("/chat", json={"session_id": "sess-1"})
    assert response.status_code == 422


@patch("app.services.rag_pipeline.retrieve")
@patch("app.services.rag_pipeline.embed")
@patch("app.services.rag_pipeline.generate", new_callable=AsyncMock)
def test_rag_pipeline_integration(
    mock_generate, mock_embed, mock_retrieve, client: TestClient
):
    """Test the full pipeline: embed → retrieve → generate → response."""
    mock_embed.return_value = [[0.1] * 384]
    mock_retrieve.return_value = [
        {"text": "Take once daily.", "drug_name": "lisinopril", "section": "dosage"}
    ]
    mock_generate.return_value = "According to the Dosage section, take once daily."

    response = client.post(
        "/chat",
        json={
            "session_id": "sess-pipeline",
            "question": "How often should I take lisinopril?",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "dosage" in data["answer"].lower() or "daily" in data["answer"].lower()
    assert data["sources"][0]["drug_name"] == "lisinopril"


@patch("app.services.rag_pipeline.retrieve")
@patch("app.services.rag_pipeline.embed")
async def test_rag_pipeline_no_chunks_skips_llm(
    mock_embed, mock_retrieve, client: TestClient
):
    """When retrieve returns nothing, LLM must not be called."""
    mock_embed.return_value = [[0.1] * 384]
    mock_retrieve.return_value = []

    with patch(
        "app.services.rag_pipeline.generate", new_callable=AsyncMock
    ) as mock_gen:
        response = client.post(
            "/chat",
            json={"session_id": "empty-sess-2", "question": "Any question"},
        )
        mock_gen.assert_not_called()

    assert response.status_code == 200
    assert "not available" in response.json()["answer"]
