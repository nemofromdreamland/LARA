import uuid
from unittest.mock import AsyncMock, patch

import pytest
import respx
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import PrescriptionEntry
from app.services.dailymed import LeafletSection

SAMPLE_IDS = {
    "maria_santos_anxiety",
    "ana_pereira_cholesterol",
    "roberto_alves_polypharmacy",
}

MOCK_SECTIONS = [
    LeafletSection(
        drug_name="sertraline",
        section="indications",
        text="Sertraline is indicated for major depressive disorder. " * 10,
    ),
]

MOCK_ENTRIES = [
    PrescriptionEntry(drug_name="sertraline", dosage="50mg", frequency="once daily"),
    PrescriptionEntry(drug_name="zolpidem", dosage="10mg", frequency="at bedtime"),
]


@pytest.fixture
def session_id(client: TestClient) -> str:
    resp = client.post("/session")
    assert resp.status_code == 200
    return resp.json()["session_id"]


def _job_status(client, job_id: str, session_id: str) -> dict:
    resp = client.get(f"/upload/status/{job_id}?session_id={session_id}")
    assert resp.status_code == 200
    return resp.json()


# ---------------------------------------------------------------------------
# GET /samples
# ---------------------------------------------------------------------------


def test_list_samples_returns_manifest(client):
    response = client.get("/samples")
    assert response.status_code == 200
    samples = response.json()["samples"]
    assert {s["id"] for s in samples} == SAMPLE_IDS
    for s in samples:
        assert s["label"]
        assert s["description"]
        assert len(s["drugs"]) >= 1


def test_list_samples_requires_api_key():
    response = TestClient(app).get("/samples")
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# POST /samples/{sample_id}
# ---------------------------------------------------------------------------


@patch("app.services.ingestion.parse_prescription", new_callable=AsyncMock)
@patch("app.services.ingestion.fetch_leaflet_sections", new_callable=AsyncMock)
@patch("app.services.ingestion.embed", new_callable=AsyncMock)
@patch("app.services.ingestion.store", new_callable=AsyncMock)
def test_load_sample_job_reaches_done_with_drugs_found(
    mock_store, mock_embed, mock_fetch, mock_parse, client, session_id
):
    mock_parse.return_value = MOCK_ENTRIES
    mock_fetch.return_value = MOCK_SECTIONS
    mock_embed.return_value = [[0.1] * 768] * 4
    mock_store.return_value = None

    response = client.post(
        "/samples/maria_santos_anxiety", json={"session_id": session_id}
    )
    assert response.status_code == 202
    data = response.json()
    assert data["session_id"] == session_id
    assert data["status"] == "processing"

    status = _job_status(client, data["job_id"], session_id)
    assert status["status"] == "done"
    assert status["drugs_found"] == ["sertraline", "zolpidem"]
    assert status["error"] is None


@patch("app.services.ingestion.parse_prescription", new_callable=AsyncMock)
@patch("app.services.ingestion.embed", new_callable=AsyncMock)
@patch("app.services.ingestion.store", new_callable=AsyncMock)
@respx.mock  # no routes registered: any live DailyMed call would fail the test
def test_load_sample_leaflets_served_from_seeded_cache(
    mock_store, mock_embed, mock_parse, client, session_id
):
    """The request-time cache seed lets the real fetch path complete with
    zero network calls — the core promise of the samples feature."""
    mock_parse.return_value = [
        PrescriptionEntry(drug_name="Sertraline 50mg"),
        PrescriptionEntry(drug_name="Zolpidem 10mg"),
    ]
    mock_embed.return_value = [[0.1] * 768]
    mock_store.return_value = None

    response = client.post(
        "/samples/maria_santos_anxiety", json={"session_id": session_id}
    )
    assert response.status_code == 202

    status = _job_status(client, response.json()["job_id"], session_id)
    assert status["status"] == "done"
    assert status["drugs_found"] == ["Sertraline 50mg", "Zolpidem 10mg"]
    assert status["missing_leaflets"] == []


def test_load_sample_unknown_id_returns_404(client, session_id):
    response = client.post("/samples/not_a_sample", json={"session_id": session_id})
    assert response.status_code == 404


def test_load_sample_expired_session_returns_410(client):
    response = client.post(
        "/samples/maria_santos_anxiety", json={"session_id": str(uuid.uuid4())}
    )
    assert response.status_code == 410


def test_load_sample_requires_api_key():
    response = TestClient(app).post(
        "/samples/maria_santos_anxiety", json={"session_id": str(uuid.uuid4())}
    )
    assert response.status_code == 401
