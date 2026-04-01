from unittest.mock import patch

from app.models.schemas import InteractionFlag


@patch("app.routes.interactions.get_upload_result")
@patch("app.routes.interactions.detect_interactions")
def test_interactions_returns_flags(mock_detect, mock_upload, client):
    mock_upload.return_value = (["warfarin", "aspirin"], [])
    mock_detect.return_value = [
        InteractionFlag(
            drug_a="warfarin",
            drug_b="aspirin",
            excerpt="Concurrent aspirin use may increase bleeding risk.",
        )
    ]

    resp = client.post("/interactions", json={"session_id": "sess-x"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["pairs_checked"] == 1
    assert len(data["interactions"]) == 1
    assert data["interactions"][0]["drug_a"] == "warfarin"
    assert data["interactions"][0]["drug_b"] == "aspirin"
    assert "aspirin" in data["interactions"][0]["excerpt"]


@patch("app.routes.interactions.get_upload_result")
@patch("app.routes.interactions.detect_interactions")
def test_interactions_no_flags_when_no_overlap(mock_detect, mock_upload, client):
    mock_upload.return_value = (["lisinopril", "metformin"], [])
    mock_detect.return_value = []

    resp = client.post("/interactions", json={"session_id": "sess-y"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["pairs_checked"] == 1
    assert data["interactions"] == []


@patch("app.routes.interactions.get_upload_result")
@patch("app.routes.interactions.detect_interactions")
def test_interactions_single_drug_zero_pairs(mock_detect, mock_upload, client):
    mock_upload.return_value = (["aspirin"], [])
    mock_detect.return_value = []

    resp = client.post("/interactions", json={"session_id": "sess-z"})
    assert resp.status_code == 200
    assert resp.json()["pairs_checked"] == 0


def test_interactions_missing_session_id_returns_422(client):
    resp = client.post("/interactions", json={})
    assert resp.status_code == 422
