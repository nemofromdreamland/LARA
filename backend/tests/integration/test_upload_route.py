import io
from unittest.mock import AsyncMock, patch

import fitz
import pytest
from fastapi.testclient import TestClient

from app.models.schemas import PrescriptionEntry
from app.services.dailymed import LeafletSection


def _make_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    return doc.tobytes()


MOCK_SECTIONS = [
    LeafletSection(
        drug_name="lisinopril",
        section="indications",
        text="Lisinopril is indicated for hypertension. " * 10,
    ),
    LeafletSection(
        drug_name="lisinopril",
        section="warnings",
        text="Do not use in patients with a history of angioedema. " * 10,
    ),
]

MOCK_ENTRIES = [
    PrescriptionEntry(
        drug_name="lisinopril",
        dosage="10mg",
        frequency="once daily",
    )
]


@pytest.fixture
def pdf_with_drug() -> bytes:
    return _make_pdf("Lisinopril 10mg once daily")


@pytest.fixture
def session_id(client: TestClient) -> str:
    resp = client.post("/session")
    assert resp.status_code == 200
    return resp.json()["session_id"]


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


@patch("app.routes.upload.parse_prescription", new_callable=AsyncMock)
@patch("app.services.ingestion.fetch_leaflet_sections", new_callable=AsyncMock)
@patch("app.routes.upload.embed", new_callable=AsyncMock)
@patch("app.routes.upload.store", new_callable=AsyncMock)
def test_upload_success(
    mock_store, mock_embed, mock_fetch, mock_parse, client, session_id, pdf_with_drug
):
    mock_parse.return_value = MOCK_ENTRIES
    mock_fetch.return_value = MOCK_SECTIONS
    mock_embed.return_value = [[0.1] * 768] * 4
    mock_store.return_value = None

    response = client.post(
        "/upload",
        data={"session_id": session_id},
        files={"file": ("rx.pdf", io.BytesIO(pdf_with_drug), "application/pdf")},
    )
    assert response.status_code == 202
    data = response.json()
    assert data["session_id"] == session_id
    assert "job_id" in data
    assert data["status"] == "processing"


@patch("app.routes.upload.parse_prescription", new_callable=AsyncMock)
@patch("app.services.ingestion.fetch_leaflet_sections", new_callable=AsyncMock)
@patch("app.routes.upload.embed", new_callable=AsyncMock)
@patch("app.routes.upload.store", new_callable=AsyncMock)
def test_upload_unknown_drug_job_accepted(
    mock_store, mock_embed, mock_fetch, mock_parse, client, session_id
):
    mock_parse.return_value = MOCK_ENTRIES
    mock_fetch.return_value = []  # DailyMed found nothing
    pdf = _make_pdf("Lisinopril 10mg once daily")

    response = client.post(
        "/upload",
        data={"session_id": session_id},
        files={"file": ("rx.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "processing"
    assert "job_id" in data


# ---------------------------------------------------------------------------
# Validation errors (synchronous — happen before background task)
# ---------------------------------------------------------------------------


def test_upload_rejects_non_pdf(client, session_id):
    response = client.post(
        "/upload",
        data={"session_id": session_id},
        files={"file": ("rx.txt", io.BytesIO(b"plain text"), "text/plain")},
    )
    assert response.status_code == 400


def test_upload_empty_pdf_returns_422(client, session_id):
    doc = fitz.open()
    doc.new_page()
    empty_pdf = doc.tobytes()

    response = client.post(
        "/upload",
        data={"session_id": session_id},
        files={"file": ("rx.pdf", io.BytesIO(empty_pdf), "application/pdf")},
    )
    assert response.status_code == 422


@patch("app.routes.upload.parse_prescription", new_callable=AsyncMock)
def test_upload_pdf_no_drugs_job_accepted(mock_parse, client, session_id):
    """No drugs found is reported asynchronously via job status, not as 422."""
    mock_parse.return_value = []
    pdf = _make_pdf("The patient is feeling well today.")

    response = client.post(
        "/upload",
        data={"session_id": session_id},
        files={"file": ("rx.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "processing"


# ---------------------------------------------------------------------------
# Job status endpoint
# ---------------------------------------------------------------------------


def test_upload_status_not_found(client, session_id):
    response = client.get(f"/upload/status/nonexistent-job-id?session_id={session_id}")
    assert response.status_code == 404
