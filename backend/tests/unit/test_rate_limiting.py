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


@pytest.fixture
def client() -> TestClient:
    return TestClient(
        app,
        raise_server_exceptions=False,
        headers={"X-API-Key": settings.lara_api_key},
    )


@pytest.fixture
def session_id(client: TestClient) -> str:
    """Create a real session so verify_session_owner passes."""
    resp = client.post("/session")
    assert resp.status_code == 200
    client.headers["X-Session-Token"] = resp.json()["session_token"]
    return resp.json()["session_id"]


# ---------------------------------------------------------------------------
# Upload rate limiting
# ---------------------------------------------------------------------------


def test_upload_allows_five_requests(client: TestClient, session_id: str):
    pdf = _make_pdf()

    for i in range(5):
        res = client.post(
            "/upload",
            data={"session_id": session_id},
            files={"file": ("rx.pdf", io.BytesIO(pdf), "application/pdf")},
        )
        assert res.status_code == 202, (
            f"Request {i + 1} unexpectedly blocked: {res.text}"
        )


def test_upload_blocks_sixth_request(client: TestClient, session_id: str):
    pdf = _make_pdf()

    for _ in range(5):
        client.post(
            "/upload",
            data={"session_id": session_id},
            files={"file": ("rx.pdf", io.BytesIO(pdf), "application/pdf")},
        )

    res = client.post(
        "/upload",
        data={"session_id": session_id},
        files={"file": ("rx.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    assert res.status_code == 429


def test_upload_429_body_has_detail_key(client: TestClient, session_id: str):
    """Drive the limit to exhaustion and verify the JSON shape."""
    pdf = _make_pdf()

    for _ in range(5):
        client.post(
            "/upload",
            data={"session_id": session_id},
            files={"file": ("rx.pdf", io.BytesIO(pdf), "application/pdf")},
        )

    res = client.post(
        "/upload",
        data={"session_id": session_id},
        files={"file": ("rx.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    assert res.status_code == 429
    body = res.json()
    assert "detail" in body


# ---------------------------------------------------------------------------
# Rate-limit key derivation
# ---------------------------------------------------------------------------


def _make_request(headers: dict[str, str], client_host: str = "9.9.9.9"):
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": (client_host, 12345),
    }
    return Request(scope)


def test_real_ip_header_is_used():
    from app.limiter import _get_real_ip

    request = _make_request({"X-Real-IP": "1.2.3.4"})
    assert _get_real_ip(request) == "1.2.3.4"


def test_forwarded_for_is_ignored():
    """Client-controlled X-Forwarded-For must not create a fresh bucket."""
    from app.limiter import _get_real_ip

    request = _make_request({"X-Forwarded-For": "6.6.6.6"})
    assert _get_real_ip(request) == "9.9.9.9"


def test_falls_back_to_client_host():
    from app.limiter import _get_real_ip

    request = _make_request({})
    assert _get_real_ip(request) == "9.9.9.9"


@patch("app.routes.chat.answer", new_callable=AsyncMock)
def test_spoofed_forwarded_for_does_not_reset_limit(
    mock_answer, client: TestClient, session_id: str
):
    mock_answer.return_value = MOCK_ANSWER

    for i in range(20):
        client.post("/chat", json={"session_id": session_id, "question": f"q{i}"})

    res = client.post(
        "/chat",
        json={"session_id": session_id, "question": "overflow"},
        headers={"X-Forwarded-For": "6.6.6.6"},
    )
    assert res.status_code == 429


# ---------------------------------------------------------------------------
# Chat rate limiting
# ---------------------------------------------------------------------------


def _make_stream(*payloads: str):
    async def _gen() -> AsyncGenerator[str, None]:
        for p in payloads:
            yield p

    return _gen()


@patch("app.routes.chat.answer", new_callable=AsyncMock)
def test_chat_allows_twenty_requests(mock_answer, client: TestClient, session_id: str):
    mock_answer.return_value = MOCK_ANSWER

    for i in range(20):
        res = client.post(
            "/chat",
            json={"session_id": session_id, "question": f"Question {i}"},
        )
        assert res.status_code == 200, (
            f"Request {i + 1} unexpectedly blocked: {res.text}"
        )


@patch("app.routes.chat.answer", new_callable=AsyncMock)
def test_chat_blocks_twenty_first_request(
    mock_answer, client: TestClient, session_id: str
):
    mock_answer.return_value = MOCK_ANSWER

    for i in range(20):
        client.post(
            "/chat", json={"session_id": session_id, "question": f"Question {i}"}
        )

    res = client.post("/chat", json={"session_id": session_id, "question": "overflow"})
    assert res.status_code == 429


@patch("app.routes.chat.answer", new_callable=AsyncMock)
def test_chat_429_body_has_detail_key(mock_answer, client: TestClient, session_id: str):
    mock_answer.return_value = MOCK_ANSWER

    for i in range(20):
        client.post("/chat", json={"session_id": session_id, "question": f"q{i}"})

    res = client.post("/chat", json={"session_id": session_id, "question": "overflow"})
    assert res.status_code == 429
    assert "detail" in res.json()
