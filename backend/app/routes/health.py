import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.schemas import ComponentHealth, HealthResponse
from app.services import embedder, vector_store

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

    components = {"chroma": chroma, "embedder": emb, "llm": llm}
    overall: str = (
        "ok" if all(c.status == "ok" for c in components.values()) else "degraded"
    )
    response = HealthResponse(status=overall, components=components)  # type: ignore[arg-type]
    return JSONResponse(
        content=response.model_dump(), status_code=200 if overall == "ok" else 503
    )
