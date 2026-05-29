import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pythonjsonlogger.json import JsonFormatter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.limiter import limiter
from app.routes import chat, health, interactions, session, upload
from app.services import session_store
from app.services.embedder import preload_model
from app.services.llm_client import close_cerebras_client, init_cerebras_client
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
    """Periodically delete ChromaDB vectors for sessions that have expired in Redis."""
    from app.services import vector_store
    from app.services.session_store import session_exists

    while True:
        await asyncio.sleep(settings.cleanup_interval_seconds)
        try:
            session_ids = await vector_store.list_session_ids()
            for sid in session_ids:
                if not await session_exists(sid):
                    deleted = await vector_store.delete_session(sid)
                    if deleted:
                        logger.info(
                            "cleanup: removed %d vectors for expired session %s",
                            deleted,
                            sid,
                        )
        except Exception:
            logger.exception("cleanup: error during orphaned-vector sweep")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    executor = ThreadPoolExecutor(
        max_workers=settings.thread_pool_workers,
        thread_name_prefix="lara-blocking",
    )
    asyncio.get_event_loop().set_default_executor(executor)
    await session_store.init_redis(settings.redis_url)
    await init_cerebras_client()
    await run_sync(preload_model)
    cleanup_task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        await asyncio.gather(cleanup_task, return_exceptions=True)
        await session_store.close_redis()
        await close_cerebras_client()
        executor.shutdown(wait=False)


app = FastAPI(title="LARA API", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})


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
app.include_router(session.router)
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(interactions.router)
