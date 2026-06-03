from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.models.schemas import InteractionFlag


@pytest.fixture
def session_id(client: TestClient) -> str:
    resp = client.post("/session")
    assert resp.status_code == 200
    return resp.json()["session_id"]


@patch("app.routes.interactions.get_upload_result")
@patch("app.routes.interactions.detect_interactions", new_callable=AsyncMock)
def test_interactions_returns_flags(mock_detect, mock_upload, client, session_id):
    mock_upload.return_value = (["warfarin", "aspirin"], [])
    mock_detect.return_value = [
        InteractionFlag(
            drug_a="warfarin",
            drug_b="aspirin",
            excerpt="Concurrent aspirin use may increase bleeding risk.",
        )
    ]

    resp = client.post("/interactions", json={"session_id": session_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["pairs_checked"] == 1
    assert len(data["interactions"]) == 1
    assert data["interactions"][0]["drug_a"] == "warfarin"
    assert data["interactions"][0]["drug_b"] == "aspirin"
    assert "aspirin" in data["interactions"][0]["excerpt"]


@patch("app.routes.interactions.get_upload_result")
@patch("app.routes.interactions.detect_interactions", new_callable=AsyncMock)
def test_interactions_no_flags_when_no_overlap(
    mock_detect, mock_upload, client, session_id
):
    mock_upload.return_value = (["lisinopril", "metformin"], [])
    mock_detect.return_value = []

    resp = client.post("/interactions", json={"session_id": session_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["pairs_checked"] == 1
    assert data["interactions"] == []


@patch("app.routes.interactions.get_upload_result")
@patch("app.routes.interactions.detect_interactions", new_callable=AsyncMock)
def test_interactions_single_drug_zero_pairs(
    mock_detect, mock_upload, client, session_id
):
    mock_upload.return_value = (["aspirin"], [])
    mock_detect.return_value = []

    resp = client.post("/interactions", json={"session_id": session_id})
    assert resp.status_code == 200
    assert resp.json()["pairs_checked"] == 0


def test_interactions_missing_session_id_returns_422(client):
    resp = client.post("/interactions", json={})
    assert resp.status_code == 422
