from fastapi.testclient import TestClient

from app.main import app


def test_metrics_requires_api_key():
    unauthed = TestClient(app)
    response = unauthed.get("/metrics")
    assert response.status_code == 401


def test_metrics_rejects_wrong_api_key():
    unauthed = TestClient(app, headers={"X-API-Key": "definitely-wrong"})
    response = unauthed.get("/metrics")
    assert response.status_code == 401


def test_metrics_ok_with_api_key(client: TestClient):
    response = client.get("/metrics")
    assert response.status_code == 200
    # Prometheus exposition format — domain metrics should be registered.
    assert "lara_llm_provider_calls" in response.text
