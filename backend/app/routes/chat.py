from fastapi import APIRouter

from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_pipeline import answer

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    return await answer(body.session_id, body.question)
