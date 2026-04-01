from collections.abc import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_pipeline import answer, answer_stream

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    return await answer(body.session_id, body.question)


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest) -> StreamingResponse:
    """Stream the RAG answer as Server-Sent Events.

    Each SSE event is one of:
      data: <token>          — a text chunk from the LLM
      data: [SOURCES]{json}  — sources list (sent once, after generation)
      data: [DONE]           — end-of-stream sentinel
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        async for payload in answer_stream(body.session_id, body.question):
            yield f"data: {payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
