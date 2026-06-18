import hashlib
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.config import settings
from app.dependencies import verify_session_owner
from app.main import app

_VALID_KEY = settings.lara_api_key

# A per-session token and the sha256 hash stored as the session owner.
_TOKEN = "session-token-under-test"
_TOKEN_HASH = hashlib.sha256(_TOKEN.encode()).hexdigest()


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


async def test_verify_session_owner_correct_token_passes():
    import app.services.session_store as store

    sid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    await store.create_session(sid)
    await store.save_session_owner(sid, _TOKEN_HASH)

    # Must not raise
    await verify_session_owner(sid, _TOKEN)


async def test_verify_session_owner_wrong_token_raises_403():
    import app.services.session_store as store

    sid = "11111111-2222-3333-4444-555555555555"
    await store.create_session(sid)
    await store.save_session_owner(sid, _TOKEN_HASH)

    with pytest.raises(HTTPException) as exc_info:
        await verify_session_owner(sid, "the-wrong-token")
    assert exc_info.value.status_code == 403


async def test_verify_session_owner_missing_token_raises_403():
    import app.services.session_store as store

    sid = "22222222-3333-4444-5555-666666666666"
    await store.create_session(sid)
    await store.save_session_owner(sid, _TOKEN_HASH)

    # No token at all (None) must be rejected like a wrong one — not let through.
    with pytest.raises(HTTPException) as exc_info:
        await verify_session_owner(sid, None)
    assert exc_info.value.status_code == 403


async def test_verify_session_owner_missing_session_raises_410():
    with pytest.raises(HTTPException) as exc_info:
        await verify_session_owner("nonexistent-session-id", _TOKEN)
    assert exc_info.value.status_code == 410


def test_session_owner_bound_via_http(authed_client: TestClient):
    """POST /session issues a token; calls carrying it succeed, others get 403."""
    resp = authed_client.post("/session")
    assert resp.status_code == 200
    sid = resp.json()["session_id"]
    token = resp.json()["session_token"]

    with (
        patch("app.routes.interactions.get_upload_result") as mock_upload,
        patch(
            "app.routes.interactions.detect_interactions", new_callable=AsyncMock
        ) as mock_detect,
    ):
        mock_upload.return_value = ([], [])
        mock_detect.return_value = []

        # Correct token → 200.
        ok = authed_client.post(
            "/interactions",
            json={"session_id": sid},
            headers={"X-Session-Token": token},
        )
        # Same API key but no session token → 403 (isolation now holds).
        no_token = authed_client.post("/interactions", json={"session_id": sid})
        # Same API key but a different/wrong token → 403.
        wrong_token = authed_client.post(
            "/interactions",
            json={"session_id": sid},
            headers={"X-Session-Token": "not-the-right-token"},
        )

    assert ok.status_code == 200
    assert no_token.status_code == 403
    assert wrong_token.status_code == 403
