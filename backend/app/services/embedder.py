import asyncio
from concurrent.futures import Executor
from functools import partial

from prometheus_client import Histogram
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"

_EMBED_DURATION = Histogram(
    "lara_embed_duration_seconds",
    "Time spent in the thread pool for each embed() call",
    ["source"],
)
_model: SentenceTransformer | None = None


def is_model_loaded() -> bool:
    return _model is not None


def preload_model() -> None:
    """Eagerly load the embedding model. Call from the app lifespan startup hook."""
    _get_model()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _encode_sync(texts: list[str]) -> list[list[float]]:
    return _get_model().encode(texts, show_progress_bar=False).tolist()


async def embed(
    texts: list[str],
    executor: Executor | None = None,
    source: str = "upload",
) -> list[list[float]]:
    """Embed texts using NeuML/pubmedbert-base-embeddings (768-dim, local).

    Runs encoding in *executor* (dedicated embed pool when called from the app,
    falls back to the loop default executor otherwise — keeps tests simple).
    """
    loop = asyncio.get_running_loop()
    with _EMBED_DURATION.labels(source=source).time():
        return await loop.run_in_executor(executor, partial(_encode_sync, texts))
