import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pythonjsonlogger.json import JsonFormatter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.dependencies import require_api_key
from app.exceptions import StorageUnavailableError
from app.limiter import limiter
from app.routes import chat, health, interactions, session, upload
from app.services import session_store
from app.services.embedder import preload_model
from app.services.llm_client import (
    ServiceUnavailableError,
    close_cerebras_client,
    init_cerebras_client,
)
from app.services.reranker import preload_reranker
from app.utils import request_id_var, run_sync

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = str(uuid.uuid4())
        request_id_var.set(rid)
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


async def _cleanup_loop() -> None:
    """Periodically delete ChromaDB collections for expired sessions."""
    from app.services import vector_store
    from app.services.session_store import session_exists

    while True:
        await asyncio.sleep(settings.cleanup_interval_seconds)
        try:
            client = vector_store._get_client()
            collections = await run_sync(client.list_collections)
            for col in collections:
                name = col.name if hasattr(col, "name") else str(col)
                sid = vector_store.session_id_from_collection(name)
                if sid is None:
                    continue
                if not await session_exists(sid):
                    deleted = await vector_store.delete_session(sid)
                    if deleted is not None:
                        logger.info(
                            "cleanup: removed collection %s for expired session %s",
                            name,
                            sid,
                        )
        except Exception:
            logger.exception("cleanup: error during orphaned-collection sweep")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    executor = ThreadPoolExecutor(
        max_workers=settings.thread_pool_workers,
        thread_name_prefix="lara-blocking",
    )
    asyncio.get_event_loop().set_default_executor(executor)
    embed_executor = ThreadPoolExecutor(
        max_workers=settings.embed_pool_workers,
        thread_name_prefix="lara-embed",
    )
    app.state.embed_executor = embed_executor
    await session_store.init_redis(settings.redis_url)
    await init_cerebras_client()
    await run_sync(preload_model)
    await run_sync(preload_reranker)
    # One-time migration: clean up orphaned "leaflets" collection from the old
    # single-collection design. Delete it only if it is empty.
    try:
        from app.services import vector_store as _vs

        _chroma = _vs._get_client()
        try:
            _legacy = await run_sync(_chroma.get_collection, "leaflets")
            _count = await run_sync(_legacy.count)
            if _count == 0:
                await run_sync(_chroma.delete_collection, "leaflets")
                logger.info("startup: deleted empty legacy 'leaflets' collection")
            else:
                logger.warning(
                    "startup: legacy 'leaflets' collection has %d vectors"
                    " — leaving intact",
                    _count,
                )
        except Exception:
            pass  # collection doesn't exist, nothing to migrate
    except Exception:
        logger.exception("startup: error during legacy collection migration check")

    cleanup_task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        await asyncio.gather(cleanup_task, return_exceptions=True)
        await session_store.close_redis()
        await close_cerebras_client()
        executor.shutdown(wait=False)
        embed_executor.shutdown(wait=False)


app = FastAPI(title="LARA API", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter

# expose() forwards extra kwargs to FastAPI's route registration, so the
# metrics endpoint requires the same API key as the business routes.
Instrumentator().instrument(app).expose(
    app, endpoint="/metrics", dependencies=[Depends(require_api_key)]
)


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


@app.exception_handler(ServiceUnavailableError)
async def _service_unavailable_handler(
    request: Request, exc: ServiceUnavailableError
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "LLM service unavailable — all providers are down"},
    )


@app.exception_handler(StorageUnavailableError)
async def _storage_unavailable_handler(
    request: Request, exc: StorageUnavailableError
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "Storage unavailable, please try again shortly."},
    )


# SlowAPIMiddleware must be added before RequestIDMiddleware (outermost first).
app.add_middleware(SlowAPIMiddleware)
# RequestIDMiddleware must be added before CORSMiddleware so the response
# header is set before CORS processing can short-circuit preflight responses.
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(session.router, dependencies=[Depends(require_api_key)])
app.include_router(upload.router, dependencies=[Depends(require_api_key)])
app.include_router(chat.router, dependencies=[Depends(require_api_key)])
app.include_router(interactions.router, dependencies=[Depends(require_api_key)])
