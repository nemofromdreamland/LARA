import hashlib

from fastapi import Header, HTTPException, status

from app.config import settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    if not x_api_key or x_api_key != settings.lara_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return hashlib.sha256(x_api_key.encode()).hexdigest()


async def verify_session_owner(session_id: str, caller_hash: str) -> None:
    from app.services import session_store

    owner_hash = await session_store.get_session_owner(session_id)
    if owner_hash is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or expired",
        )
    if owner_hash != caller_hash:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Session belongs to a different caller",
        )
