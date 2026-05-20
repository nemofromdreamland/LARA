import logging
import uuid

from fastapi import APIRouter

from app.models.schemas import SessionResponse
from app.services.session_store import create_session as store_create_session
from app.utils import get_request_id

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/session", response_model=SessionResponse)
async def create_session() -> SessionResponse:
    sid = str(uuid.uuid4())
    await store_create_session(sid)
    logger.info("session created", extra={"request_id": get_request_id()})
    return SessionResponse(session_id=sid)
