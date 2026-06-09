import asyncio
from functools import partial

from prometheus_client import Histogram
from sentence_transformers import CrossEncoder

_RERANK_DURATION = Histogram(
    "lara_rerank_duration_seconds",
    "Time spent in the thread pool for each rerank() call",
)

# BAAI/bge-reranker-base significantly outperforms the MS MARCO MiniLM models
# on heterogeneous domains (BEIR benchmark), which better suits medical leaflet
# text that is structurally different from web-search queries.
_RERANKER_MODEL = "BAAI/bge-reranker-base"
# MiniLM and BGE cross-encoders have a 512-token window; set max_length so
# the tokenizer truncates predictably rather than silently dropping tail tokens.
_MAX_LENGTH = 512
_reranker: CrossEncoder | None = None


def preload_reranker() -> None:
    """Eagerly load the cross-encoder. Call from the app lifespan startup hook."""
    _get_reranker()


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(_RERANKER_MODEL, max_length=_MAX_LENGTH)
    return _reranker


def _predict_sync(query: str, texts: list[str]) -> list[float]:
    pairs = [(query, text) for text in texts]
    return _get_reranker().predict(pairs).tolist()


async def rerank(query: str, chunks: list[dict]) -> list[dict]:
    """Score each chunk against *query* with a cross-encoder and sort desc.

    Attaches a ``rerank_score`` float to every chunk dict in-place, then
    returns the list sorted from highest to lowest score.
    """
    if not chunks:
        return chunks
    texts = [c["text"] for c in chunks]
    loop = asyncio.get_running_loop()
    with _RERANK_DURATION.time():
        scores: list[float] = await loop.run_in_executor(
            None, partial(_predict_sync, query, texts)
        )
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = score
    return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
