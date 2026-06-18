import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse

from app.config import settings
from app.dependencies import require_api_key, verify_session_owner
from app.limiter import limiter
from app.models.schemas import ChatRequest, ChatResponse
from app.services import session_store
from app.services.llm_client import STREAM_RESET, strip_cited_line
from app.services.rag_pipeline import answer, answer_stream

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.chat_rate_limit)
async def chat(
    request: Request,
    body: ChatRequest,
    _api_key: str = Depends(require_api_key),
    x_session_token: str | None = Header(default=None, alias="X-Session-Token"),
) -> ChatResponse:
    await verify_session_owner(body.session_id, x_session_token)
    embed_executor = getattr(request.app.state, "embed_executor", None)
    history = [h.model_dump() for h in await session_store.get_history(body.session_id)]
    result = await answer(body.session_id, body.question, history, embed_executor)
    await session_store.append_history(body.session_id, "user", body.question)
    await session_store.append_history(body.session_id, "assistant", result.answer)
    return result


@router.post("/chat/stream")
@limiter.limit(settings.chat_rate_limit)
async def chat_stream(
    request: Request,
    body: ChatRequest,
    _api_key: str = Depends(require_api_key),
    x_session_token: str | None = Header(default=None, alias="X-Session-Token"),
) -> StreamingResponse:
    """Stream the RAG answer as Server-Sent Events.

    Event types:
      event: token   — a JSON-encoded text chunk from the LLM
      event: reset   — discard all tokens received so far (mid-stream provider
                       failover regenerates the answer from scratch)
      event: sources — JSON sources payload (sent once, after generation)
      event: done    — end-of-stream sentinel
    """

    await verify_session_owner(body.session_id, x_session_token)
    embed_executor = getattr(request.app.state, "embed_executor", None)
    # Fetched before the StreamingResponse is constructed so a Redis outage
    # surfaces as a proper 503 instead of a 200 with a broken stream.
    history = [h.model_dump() for h in await session_store.get_history(body.session_id)]

    async def event_generator() -> AsyncGenerator[str, None]:
        assistant_tokens: list[str] = []
        async for payload in answer_stream(
            body.session_id, body.question, history, embed_executor
        ):
            if payload == "[DONE]":
                yield "event: done\ndata: \n\n"
                await session_store.append_history(
                    body.session_id, "user", body.question
                )
                clean_answer, _ = strip_cited_line("".join(assistant_tokens))
                await session_store.append_history(
                    body.session_id, "assistant", clean_answer
                )
            elif payload == STREAM_RESET:
                # Mid-stream failover: the partial tokens are superseded by the
                # regenerated answer, both client-side and in stored history.
                assistant_tokens.clear()
                yield "event: reset\ndata: \n\n"
            elif payload.startswith("[SOURCES]"):
                yield f"event: sources\ndata: {payload[9:]}\n\n"
            else:
                assistant_tokens.append(payload)
                # JSON-encode so newlines in LLM tokens don't corrupt SSE framing
                yield f"event: token\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
