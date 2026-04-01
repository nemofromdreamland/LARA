import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import chat, health, interactions, session, upload
from app.services.session_store import expire_sessions
from app.services.vector_store import delete_session

logger = logging.getLogger(__name__)


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
    task = asyncio.create_task(_eviction_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="LARA API", version="0.1.0", lifespan=lifespan)

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
