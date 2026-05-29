"""Unit tests for app.services.reranker."""

from unittest.mock import patch

import pytest

from app.services.reranker import preload_reranker, rerank


def _chunk(
    text: str, distance: float = 0.3, drug: str = "drugA", section: str = "dosage"
) -> dict:
    return {"text": text, "distance": distance, "drug_name": drug, "section": section}


# ---------------------------------------------------------------------------
# rerank — edge cases
# ---------------------------------------------------------------------------


async def test_rerank_empty_input_returns_empty():
    result = await rerank("query", [])
    assert result == []


# ---------------------------------------------------------------------------
# rerank — scoring and ordering
# ---------------------------------------------------------------------------


async def test_rerank_attaches_rerank_score_to_every_chunk():
    chunks = [_chunk("text A"), _chunk("text B")]
    with patch("app.services.reranker._predict_sync", return_value=[0.8, 0.5]):
        result = await rerank("query", chunks)
    assert all("rerank_score" in c for c in result)


async def test_rerank_sorts_descending_by_score():
    chunks = [_chunk("low relevance"), _chunk("high relevance")]
    with patch("app.services.reranker._predict_sync", return_value=[0.3, 0.9]):
        result = await rerank("query", chunks)
    assert result[0]["text"] == "high relevance"
    assert result[0]["rerank_score"] == pytest.approx(0.9)
    assert result[1]["text"] == "low relevance"
    assert result[1]["rerank_score"] == pytest.approx(0.3)


async def test_rerank_single_chunk_returned_with_score():
    chunks = [_chunk("only chunk")]
    with patch("app.services.reranker._predict_sync", return_value=[0.7]):
        result = await rerank("query", chunks)
    assert len(result) == 1
    assert result[0]["rerank_score"] == pytest.approx(0.7)


async def test_rerank_preserves_all_chunk_fields():
    chunk = {
        "text": "sample",
        "distance": 0.2,
        "drug_name": "aspirin",
        "section": "warnings",
    }
    with patch("app.services.reranker._predict_sync", return_value=[0.6]):
        result = await rerank("query", [chunk])
    assert result[0]["drug_name"] == "aspirin"
    assert result[0]["section"] == "warnings"
    assert result[0]["distance"] == pytest.approx(0.2)


async def test_rerank_equal_scores_returns_all_chunks():
    chunks = [_chunk("A"), _chunk("B"), _chunk("C")]
    with patch("app.services.reranker._predict_sync", return_value=[0.5, 0.5, 0.5]):
        result = await rerank("query", chunks)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# preload_reranker
# ---------------------------------------------------------------------------


def test_preload_reranker_calls_get_reranker():
    with patch("app.services.reranker._get_reranker") as mock_get:
        preload_reranker()
        mock_get.assert_called_once()
