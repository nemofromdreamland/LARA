import asyncio
from functools import partial

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _encode_sync(texts: list[str]) -> list[list[float]]:
    return _get_model().encode(texts, show_progress_bar=False).tolist()


async def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts using NeuML/pubmedbert-base-embeddings (768-dim, local).

    Runs encoding in a thread-pool executor so the async event loop is not blocked.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(_encode_sync, texts))
