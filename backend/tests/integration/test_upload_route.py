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


# ---------------------------------------------------------------------------
# Ingestion outcomes via job status (TestClient runs the background task
# before the response returns, so the final status is immediately pollable)
# ---------------------------------------------------------------------------


def _job_status(client, job_id: str, session_id: str) -> dict:
    resp = client.get(f"/upload/status/{job_id}?session_id={session_id}")
    assert resp.status_code == 200
    return resp.json()


@patch("app.routes.upload.parse_prescription", new_callable=AsyncMock)
@patch("app.services.ingestion.fetch_leaflet_sections", new_callable=AsyncMock)
@patch("app.routes.upload.embed", new_callable=AsyncMock)
@patch("app.routes.upload.store", new_callable=AsyncMock)
def test_upload_success_job_reaches_done_with_drugs_found(
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

    data = _job_status(client, response.json()["job_id"], session_id)
    assert data["status"] == "done"
    assert data["drugs_found"] == ["lisinopril"]
    assert data["missing_leaflets"] == []
    assert data["error"] is None
    mock_store.assert_awaited_once()


@patch("app.routes.upload.parse_prescription", new_callable=AsyncMock)
@patch("app.routes.upload.process_drug", new_callable=AsyncMock)
@patch("app.routes.upload.embed", new_callable=AsyncMock)
@patch("app.routes.upload.store", new_callable=AsyncMock)
def test_upload_per_drug_failure_job_done_with_missing_leaflet(
    mock_store, mock_embed, mock_process, mock_parse, client, session_id, pdf_with_drug
):
    """One drug's DailyMed fetch blows up: the job still finishes 'done' and
    reports that drug under missing_leaflets while the other is stored."""
    mock_parse.return_value = [
        PrescriptionEntry(drug_name="lisinopril", dosage="10mg", frequency="daily"),
        PrescriptionEntry(drug_name="metformin", dosage="500mg", frequency="daily"),
    ]

    async def _process(drug: str):
        if drug == "metformin":
            raise RuntimeError("DailyMed timed out")
        return drug, ["chunk one"], [{"drug_name": drug, "section": "dosage"}]

    mock_process.side_effect = _process
    mock_embed.return_value = [[0.1] * 768]
    mock_store.return_value = None

    response = client.post(
        "/upload",
        data={"session_id": session_id},
        files={"file": ("rx.pdf", io.BytesIO(pdf_with_drug), "application/pdf")},
    )
    assert response.status_code == 202

    data = _job_status(client, response.json()["job_id"], session_id)
    assert data["status"] == "done"
    assert data["drugs_found"] == ["lisinopril"]
    assert data["missing_leaflets"] == ["metformin"]
    assert data["error"] is None


@patch("app.routes.upload.parse_prescription", new_callable=AsyncMock)
@patch("app.services.ingestion.fetch_leaflet_sections", new_callable=AsyncMock)
def test_upload_unknown_drug_job_done_with_missing_leaflet(
    mock_fetch, mock_parse, client, session_id, pdf_with_drug
):
    """No leaflet found (empty fetch, no exception) also ends in 'done' with
    the drug listed under missing_leaflets."""
    mock_parse.return_value = MOCK_ENTRIES
    mock_fetch.return_value = []

    response = client.post(
        "/upload",
        data={"session_id": session_id},
        files={"file": ("rx.pdf", io.BytesIO(pdf_with_drug), "application/pdf")},
    )
    assert response.status_code == 202

    data = _job_status(client, response.json()["job_id"], session_id)
    assert data["status"] == "done"
    assert data["drugs_found"] == []
    assert data["missing_leaflets"] == ["lisinopril"]


@patch("app.routes.upload.parse_prescription", new_callable=AsyncMock)
@patch("app.services.ingestion.fetch_leaflet_sections", new_callable=AsyncMock)
@patch("app.routes.upload.embed", new_callable=AsyncMock)
def test_upload_embed_failure_job_fails_with_error(
    mock_embed, mock_fetch, mock_parse, client, session_id, pdf_with_drug
):
    """An exception outside the per-drug gather (embedding) fails the job."""
    mock_parse.return_value = MOCK_ENTRIES
    mock_fetch.return_value = MOCK_SECTIONS
    mock_embed.side_effect = RuntimeError("embedding pool exploded")

    response = client.post(
        "/upload",
        data={"session_id": session_id},
        files={"file": ("rx.pdf", io.BytesIO(pdf_with_drug), "application/pdf")},
    )
    assert response.status_code == 202

    data = _job_status(client, response.json()["job_id"], session_id)
    assert data["status"] == "failed"
    assert "embedding pool exploded" in data["error"]
    assert data["drugs_found"] == []


@patch("app.routes.upload.parse_prescription", new_callable=AsyncMock)
def test_upload_no_drugs_job_fails_with_message(
    mock_parse, client, session_id, pdf_with_drug
):
    mock_parse.return_value = []

    response = client.post(
        "/upload",
        data={"session_id": session_id},
        files={"file": ("rx.pdf", io.BytesIO(pdf_with_drug), "application/pdf")},
    )
    assert response.status_code == 202

    data = _job_status(client, response.json()["job_id"], session_id)
    assert data["status"] == "failed"
    assert "No drug names found" in data["error"]
