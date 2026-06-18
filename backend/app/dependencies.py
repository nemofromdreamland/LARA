import hashlib
import hmac

from fastapi import Header, HTTPException, status

from app.config import settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    if not x_api_key or not hmac.compare_digest(
        x_api_key.encode(), settings.lara_api_key.encode()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return hashlib.sha256(x_api_key.encode()).hexdigest()


async def verify_session_owner(session_id: str, session_token: str | None) -> None:
    from app.services import session_store

    owner_hash = await session_store.get_session_owner(session_id)
    if owner_hash is None:
        # 410 (not 404): the frontend treats 410 as "session expired" and
        # auto-creates a fresh session (frontend/src/api.ts SessionExpiredError).
        # This branch must stay first so an expired/missing session never leaks
        # as a 403.
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Session not found or expired",
        )
    # A missing token hashes the empty string, which won't match any stored
    # owner hash, so it falls through to 403 like a wrong token.
    token_hash = hashlib.sha256((session_token or "").encode()).hexdigest()
    if not hmac.compare_digest(owner_hash, token_hash):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session belongs to a different caller",
        )
