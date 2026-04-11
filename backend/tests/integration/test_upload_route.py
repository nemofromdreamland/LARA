import io
from unittest.mock import AsyncMock, patch

import fitz
import pytest

from app.models.schemas import PrescriptionEntry
from app.services.dailymed import LeafletSection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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

# A structured PrescriptionEntry returned by the mocked parse_prescription.
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


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


@patch("app.routes.upload.parse_prescription", new_callable=AsyncMock)
@patch("app.routes.upload.fetch_leaflet_sections", new_callable=AsyncMock)
@patch("app.routes.upload.embed", new_callable=AsyncMock)
@patch("app.routes.upload.store")
def test_upload_success(
    mock_store, mock_embed, mock_fetch, mock_parse, client, pdf_with_drug
):
    mock_parse.return_value = MOCK_ENTRIES
    mock_fetch.return_value = MOCK_SECTIONS
    mock_embed.return_value = [[0.1] * 768] * 4
    mock_store.return_value = None

    response = client.post(
        "/upload",
        data={"session_id": "test-session-1"},
        files={"file": ("rx.pdf", io.BytesIO(pdf_with_drug), "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test-session-1"
    assert "lisinopril" in data["drugs_found"]
    assert data["missing_leaflets"] == []
    assert data["status"] == "ok"


@patch("app.routes.upload.parse_prescription", new_callable=AsyncMock)
@patch("app.routes.upload.fetch_leaflet_sections", new_callable=AsyncMock)
@patch("app.routes.upload.embed", new_callable=AsyncMock)
@patch("app.routes.upload.store")
def test_upload_unknown_drug_returns_no_leaflets(
    mock_store, mock_embed, mock_fetch, mock_parse, client
):
    mock_parse.return_value = MOCK_ENTRIES
    mock_fetch.return_value = []  # DailyMed found nothing
    pdf = _make_pdf("Lisinopril 10mg once daily")

    response = client.post(
        "/upload",
        data={"session_id": "test-session-2"},
        files={"file": ("rx.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "no_leaflets_found"
    assert data["drugs_found"] == []
    assert "lisinopril" in data["missing_leaflets"]


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_upload_rejects_non_pdf(client):
    response = client.post(
        "/upload",
        data={"session_id": "test-session-3"},
        files={"file": ("rx.txt", io.BytesIO(b"plain text"), "text/plain")},
    )
    assert response.status_code == 400


def test_upload_empty_pdf_returns_422(client):
    doc = fitz.open()
    doc.new_page()
    empty_pdf = doc.tobytes()

    response = client.post(
        "/upload",
        data={"session_id": "test-session-4"},
        files={"file": ("rx.pdf", io.BytesIO(empty_pdf), "application/pdf")},
    )
    assert response.status_code == 422


@patch("app.routes.upload.parse_prescription", new_callable=AsyncMock)
def test_upload_pdf_no_drugs_returns_422(mock_parse, client):
    mock_parse.return_value = []  # LLM + fallback both found nothing
    pdf = _make_pdf("The patient is feeling well today.")

    response = client.post(
        "/upload",
        data={"session_id": "test-session-5"},
        files={"file": ("rx.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    assert response.status_code == 422
