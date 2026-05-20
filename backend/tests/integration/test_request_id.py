"""Tests for structured logging and X-Request-ID middleware."""

import logging
import uuid

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# X-Request-ID header presence tests
# ---------------------------------------------------------------------------


def test_session_response_has_request_id_header(client: TestClient):
    response = client.post("/session")
    assert response.status_code == 200
    assert "x-request-id" in response.headers


def test_health_response_has_request_id_header(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert "x-request-id" in response.headers


def test_request_id_is_valid_uuid4(client: TestClient):
    response = client.post("/session")
    rid = response.headers["x-request-id"]
    parsed = uuid.UUID(rid)
    assert parsed.version == 4


def test_each_request_gets_unique_id(client: TestClient):
    ids = {client.post("/session").headers["x-request-id"] for _ in range(5)}
    assert len(ids) == 5


# ---------------------------------------------------------------------------
# Structured log record tests
# ---------------------------------------------------------------------------


def test_log_records_contain_request_id(client: TestClient, caplog):
    """Log records emitted during a request must carry the same request_id
    as the X-Request-ID response header."""
    with caplog.at_level(logging.INFO):
        response = client.post("/session")

    rid = response.headers["x-request-id"]

    matching = [r for r in caplog.records if getattr(r, "request_id", None) == rid]
    assert matching, (
        f"No log record had request_id={rid!r}. "
        f"Records: {[(r.name, getattr(r, 'request_id', None)) for r in caplog.records]}"
    )


def test_log_records_request_id_is_string(client: TestClient, caplog):
    """request_id on log records must be a non-empty string."""
    with caplog.at_level(logging.INFO):
        client.post("/session")

    records_with_rid = [
        r for r in caplog.records if getattr(r, "request_id", None) not in (None, "no-request")
    ]
    assert records_with_rid, "Expected at least one log record with a real request_id"
    for r in records_with_rid:
        assert isinstance(r.request_id, str) and r.request_id
