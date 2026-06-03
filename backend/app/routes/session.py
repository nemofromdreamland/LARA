import logging
import uuid

from fastapi import APIRouter, Depends

from app.dependencies import require_api_key
from app.models.schemas import SessionResponse
from app.services.session_store import create_session as store_create_session
from app.services.session_store import save_session_owner
from app.utils import get_request_id

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/session", response_model=SessionResponse)
async def create_session(
    caller_hash: str = Depends(require_api_key),
) -> SessionResponse:
    sid = str(uuid.uuid4())
    await store_create_session(sid)
    await save_session_owner(sid, caller_hash)
    logger.info("session created", extra={"request_id": get_request_id()})
    return SessionResponse(session_id=sid)
