import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.schemas import ComponentHealth, HealthResponse
from app.services import embedder, vector_store
from app.services.llm_client import _cerebras_breaker, _groq_breaker
from app.services.session_store import get_redis

router = APIRouter()


@router.get("/health")
async def health() -> JSONResponse:
    try:
        ok = await asyncio.wait_for(vector_store.ping(), timeout=2.0)
        chroma = ComponentHealth(status="ok" if ok else "unavailable")
    except asyncio.TimeoutError:
        chroma = ComponentHealth(status="unavailable", detail="timeout")
    except Exception:
        chroma = ComponentHealth(status="unavailable")

    emb = ComponentHealth(
        status="ok" if embedder.is_model_loaded() else "degraded",
        detail=None if embedder.is_model_loaded() else "model not loaded",
    )

    has_key = bool(settings.groq_api_key or settings.cerebras_api_key)
    llm = ComponentHealth(
        status="ok" if has_key else "degraded",
        detail=None if has_key else "no API key configured",
    )

    groq_open = not await _groq_breaker.allow_request()
    cerebras_open = not await _cerebras_breaker.allow_request()
    if groq_open and cerebras_open:
        llm_routing = ComponentHealth(status="degraded", detail="both_open")
    elif groq_open:
        llm_routing = ComponentHealth(status="degraded", detail="groq_open")
    elif cerebras_open:
        llm_routing = ComponentHealth(status="degraded", detail="cerebras_open")
    else:
        llm_routing = ComponentHealth(status="ok")

    try:
        await get_redis().ping()
        redis_comp = ComponentHealth(status="ok")
    except Exception as exc:
        redis_comp = ComponentHealth(status="unavailable", detail=str(exc))

    components = {
        "chroma": chroma,
        "embedder": emb,
        "llm": llm,
        "llm_routing": llm_routing,
        "redis": redis_comp,
    }
    overall: str = (
        "ok" if all(c.status == "ok" for c in components.values()) else "degraded"
    )
    response = HealthResponse(status=overall, components=components)  # type: ignore[arg-type]
    return JSONResponse(
        content=response.model_dump(), status_code=200 if overall == "ok" else 503
    )
