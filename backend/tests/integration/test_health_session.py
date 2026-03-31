import uuid

from fastapi.testclient import TestClient


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_session(client: TestClient):
    response = client.post("/session")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    # Must be a valid UUID
    uuid.UUID(data["session_id"])


def test_each_session_is_unique(client: TestClient):
    ids = {client.post("/session").json()["session_id"] for _ in range(5)}
    assert len(ids) == 5
