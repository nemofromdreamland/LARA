import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.config import settings
from app.dependencies import require_api_key, verify_session_owner
from app.limiter import limiter
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag_pipeline import answer, answer_stream

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.chat_rate_limit)
async def chat(
    request: Request,
    body: ChatRequest,
    caller_hash: str = Depends(require_api_key),
) -> ChatResponse:
    await verify_session_owner(body.session_id, caller_hash)
    embed_executor = getattr(request.app.state, "embed_executor", None)
    history = [h.model_dump() for h in body.history]
    return await answer(body.session_id, body.question, history, embed_executor)


@router.post("/chat/stream")
@limiter.limit(settings.chat_rate_limit)
async def chat_stream(
    request: Request,
    body: ChatRequest,
    caller_hash: str = Depends(require_api_key),
) -> StreamingResponse:
    """Stream the RAG answer as Server-Sent Events.

    Event types:
      event: token   — a JSON-encoded text chunk from the LLM
      event: sources — JSON sources payload (sent once, after generation)
      event: done    — end-of-stream sentinel
    """

    await verify_session_owner(body.session_id, caller_hash)
    embed_executor = getattr(request.app.state, "embed_executor", None)
    history = [h.model_dump() for h in body.history]

    async def event_generator() -> AsyncGenerator[str, None]:
        async for payload in answer_stream(
            body.session_id, body.question, history, embed_executor
        ):
            if payload == "[DONE]":
                yield "event: done\ndata: \n\n"
            elif payload.startswith("[SOURCES]"):
                yield f"event: sources\ndata: {payload[9:]}\n\n"
            else:
                # JSON-encode so newlines in LLM tokens don't corrupt SSE framing
                yield f"event: token\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
