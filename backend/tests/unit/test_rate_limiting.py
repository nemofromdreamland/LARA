"""Rate limiting tests for /upload and /chat endpoints."""

import io
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import fitz
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.models.schemas import ChatResponse, PrescriptionEntry, Source
from app.services.dailymed import LeafletSection

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_SID = "00000000-0000-0000-0000-000000000001"

MOCK_ENTRIES = [
    PrescriptionEntry(drug_name="aspirin", dosage="100mg", frequency="once daily")
]
MOCK_SECTIONS = [
    LeafletSection(
        drug_name="aspirin",
        section="indications",
        text="Aspirin is indicated for pain relief. " * 20,
    )
]
MOCK_ANSWER = ChatResponse(
    answer="According to the Indications section, aspirin relieves pain.",
    sources=[Source(drug_name="aspirin", section="indications")],
)


def _make_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Aspirin 100mg once daily", fontsize=12)
    return doc.tobytes()


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Clear in-memory limiter counters before every test."""
    from app.limiter import limiter

    limiter._storage.reset()
    yield


@pytest.fixture
def client() -> TestClient:
    return TestClient(
        app,
        raise_server_exceptions=False,
        headers={"X-API-Key": settings.lara_api_key},
    )


# ---------------------------------------------------------------------------
# Upload rate limiting
# ---------------------------------------------------------------------------


def test_upload_allows_five_requests(client: TestClient):
    # Pipeline runs in the background; route handler only validates PDF and
    # returns 202, so no service mocks are needed for rate limit checks.
    pdf = _make_pdf()

    for i in range(5):
        res = client.post(
            "/upload",
            data={"session_id": f"sess-{i}"},
            files={"file": ("rx.pdf", io.BytesIO(pdf), "application/pdf")},
        )
        assert res.status_code == 202, (
            f"Request {i + 1} unexpectedly blocked: {res.text}"
        )


def test_upload_blocks_sixth_request(client: TestClient):
    pdf = _make_pdf()

    for i in range(5):
        client.post(
            "/upload",
            data={"session_id": f"sess-{i}"},
            files={"file": ("rx.pdf", io.BytesIO(pdf), "application/pdf")},
        )

    res = client.post(
        "/upload",
        data={"session_id": "sess-overflow"},
        files={"file": ("rx.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    assert res.status_code == 429


def test_upload_429_body_has_detail_key(client: TestClient):
    """Drive the limit to exhaustion and verify the JSON shape."""
    pdf = _make_pdf()

    for i in range(5):
        client.post(
            "/upload",
            data={"session_id": f"sess-{i}"},
            files={"file": ("rx.pdf", io.BytesIO(pdf), "application/pdf")},
        )

    res = client.post(
        "/upload",
        data={"session_id": "sess-overflow"},
        files={"file": ("rx.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    assert res.status_code == 429
    body = res.json()
    assert "detail" in body


# ---------------------------------------------------------------------------
# Chat rate limiting
# ---------------------------------------------------------------------------


def _make_stream(*payloads: str):
    async def _gen() -> AsyncGenerator[str, None]:
        for p in payloads:
            yield p

    return _gen()


@patch("app.routes.chat.answer", new_callable=AsyncMock)
def test_chat_allows_twenty_requests(mock_answer, client: TestClient):
    mock_answer.return_value = MOCK_ANSWER

    for i in range(20):
        res = client.post(
            "/chat",
            json={"session_id": _SID, "question": f"Question {i}"},
        )
        assert res.status_code == 200, (
            f"Request {i + 1} unexpectedly blocked: {res.text}"
        )


@patch("app.routes.chat.answer", new_callable=AsyncMock)
def test_chat_blocks_twenty_first_request(mock_answer, client: TestClient):
    mock_answer.return_value = MOCK_ANSWER

    for i in range(20):
        client.post("/chat", json={"session_id": _SID, "question": f"Question {i}"})

    res = client.post("/chat", json={"session_id": _SID, "question": "overflow"})
    assert res.status_code == 429


@patch("app.routes.chat.answer", new_callable=AsyncMock)
def test_chat_429_body_has_detail_key(mock_answer, client: TestClient):
    mock_answer.return_value = MOCK_ANSWER

    for i in range(20):
        client.post("/chat", json={"session_id": _SID, "question": f"q{i}"})

    res = client.post("/chat", json={"session_id": _SID, "question": "overflow"})
    assert res.status_code == 429
    assert "detail" in res.json()
