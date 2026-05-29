import asyncio
from functools import partial

from sentence_transformers import CrossEncoder

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker: CrossEncoder | None = None


def preload_reranker() -> None:
    """Eagerly load the cross-encoder. Call from the app lifespan startup hook."""
    _get_reranker()


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(_RERANKER_MODEL)
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
    scores: list[float] = await loop.run_in_executor(
        None, partial(_predict_sync, query, texts)
    )
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = score
    return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
