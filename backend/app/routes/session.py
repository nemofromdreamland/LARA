import uuid

from fastapi import APIRouter

from app.models.schemas import SessionResponse

router = APIRouter()


@router.post("/session", response_model=SessionResponse)
async def create_session() -> SessionResponse:
    return SessionResponse(session_id=str(uuid.uuid4()))
