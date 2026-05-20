import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pythonjsonlogger.json import JsonFormatter
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.routes import chat, health, interactions, session, upload
from app.services.session_store import expire_sessions
from app.services.vector_store import delete_session
from app.utils import request_id_var

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


async def _eviction_loop() -> None:
    """Background task: evict expired sessions on a fixed interval."""
    while True:
        await asyncio.sleep(settings.expiry_interval_seconds)
        try:
            expired = expire_sessions(settings.session_ttl_seconds)
            for sid in expired:
                deleted = delete_session(sid)
                logger.info(
                    "Evicted session %s — %d ChromaDB docs deleted", sid, deleted
                )
        except Exception:
            logger.exception("Error during session eviction")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_logging()
    task = asyncio.create_task(_eviction_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="LARA API", version="0.1.0", lifespan=lifespan)

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
