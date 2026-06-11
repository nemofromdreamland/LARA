import hashlib
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.config import settings
from app.dependencies import verify_session_owner
from app.main import app

_VALID_KEY = settings.lara_api_key
_VALID_HASH = hashlib.sha256(_VALID_KEY.encode()).hexdigest()


@pytest.fixture
def authed_client() -> TestClient:
    return TestClient(app, headers={"X-API-Key": _VALID_KEY})


@pytest.fixture
def bare_client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# require_api_key
# ---------------------------------------------------------------------------


def test_valid_key_passes(authed_client: TestClient):
    resp = authed_client.post("/session")
    assert resp.status_code == 200


def test_missing_key_returns_401(bare_client: TestClient):
    resp = bare_client.post("/session")
    assert resp.status_code == 401


def test_wrong_key_returns_401(bare_client: TestClient):
    resp = bare_client.post("/session", headers={"X-API-Key": "definitely-wrong"})
    assert resp.status_code == 401


def test_health_not_guarded(bare_client: TestClient):
    # /health must be reachable without an API key
    resp = bare_client.get("/health")
    assert resp.status_code != 401


# ---------------------------------------------------------------------------
# verify_session_owner
# ---------------------------------------------------------------------------


async def test_verify_session_owner_correct_owner_passes():
    import app.services.session_store as store

    sid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    await store.create_session(sid)
    await store.save_session_owner(sid, _VALID_HASH)

    # Must not raise
    await verify_session_owner(sid, _VALID_HASH)


async def test_verify_session_owner_wrong_hash_raises_403():
    import app.services.session_store as store

    sid = "11111111-2222-3333-4444-555555555555"
    await store.create_session(sid)
    await store.save_session_owner(sid, _VALID_HASH)

    with pytest.raises(HTTPException) as exc_info:
        await verify_session_owner(sid, "wrong_hash_value")
    assert exc_info.value.status_code == 403


async def test_verify_session_owner_missing_session_raises_410():
    with pytest.raises(HTTPException) as exc_info:
        await verify_session_owner("nonexistent-session-id", _VALID_HASH)
    assert exc_info.value.status_code == 410


def test_session_owner_bound_via_http(authed_client: TestClient):
    """POST /session stores owner; subsequent calls with same key work."""
    resp = authed_client.post("/session")
    assert resp.status_code == 200
    sid = resp.json()["session_id"]

    # Session exists with matching owner — interactions must not return 403/404.
    with (
        patch("app.routes.interactions.get_upload_result") as mock_upload,
        patch(
            "app.routes.interactions.detect_interactions", new_callable=AsyncMock
        ) as mock_detect,
    ):
        mock_upload.return_value = ([], [])
        mock_detect.return_value = []
        resp2 = authed_client.post("/interactions", json={"session_id": sid})
    assert resp2.status_code == 200
