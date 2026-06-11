"""Unit tests for rag_pipeline: trim_to_budget, _build_fallback_message, rerank sort."""

from unittest.mock import patch

import pytest

from app.services.rag_pipeline import _build_fallback_message, _enc, trim_to_budget


def _chunk(
    text: str, distance: float, drug: str = "drugA", section: str = "dosage"
) -> dict:
    return {"text": text, "distance": distance, "drug_name": drug, "section": section}


def _tok(text: str) -> int:
    return len(_enc.encode(text))


# ---------------------------------------------------------------------------
# trim_to_budget — all chunks fit
# ---------------------------------------------------------------------------


def test_all_chunks_fit_returned_sorted_by_distance():
    chunks = [
        _chunk("B" * 100, distance=0.3),
        _chunk("A" * 100, distance=0.1),
        _chunk("C" * 100, distance=0.4),
    ]
    result = trim_to_budget(chunks, max_tokens=10_000)

    assert len(result) == 3
    distances = [c["distance"] for c in result]
    assert distances == sorted(distances)


def test_all_chunks_fit_exact_budget():
    chunks = [_chunk("x" * 50, distance=0.2), _chunk("y" * 50, distance=0.1)]
    budget = sum(_tok(c["text"]) for c in chunks)
    result = trim_to_budget(chunks, max_tokens=budget)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# trim_to_budget — overflow: only highest-relevance chunks kept
# ---------------------------------------------------------------------------


def test_overflow_drops_least_relevant_chunks():
    chunks = [
        _chunk("A" * 100, distance=0.1),  # most relevant — keep
        _chunk("B" * 100, distance=0.2),  # second — keep
        _chunk("C" * 100, distance=0.3),  # third — would exceed budget → drop
    ]
    # budget fits exactly the two most-relevant chunks
    budget = _tok("A" * 100) + _tok("B" * 100)
    result = trim_to_budget(chunks, max_tokens=budget)

    assert len(result) == 2
    assert all(c["distance"] < 0.25 for c in result)


def test_overflow_single_chunk_too_large_returns_empty():
    chunks = [_chunk("X" * 500, distance=0.1)]
    result = trim_to_budget(chunks, max_tokens=1)
    assert result == []


def test_overflow_preserves_most_relevant_order():
    chunks = [
        _chunk("Z" * 80, distance=0.4),
        _chunk("A" * 80, distance=0.05),
        _chunk("M" * 80, distance=0.2),
    ]
    # budget fits the two most-relevant chunks
    budget = _tok("A" * 80) + _tok("M" * 80)
    result = trim_to_budget(chunks, max_tokens=budget)

    assert len(result) == 2
    assert result[0]["distance"] == 0.05
    assert result[1]["distance"] == 0.2


# ---------------------------------------------------------------------------
# Prescription summary always counted toward budget before chunks
# ---------------------------------------------------------------------------


def test_prescription_summary_counted_before_chunks():
    """Caller subtracts prescription token count from budget before calling trim."""
    prescription = "[Prescription]\n1. Aspirin\n   • Dosage: 100mg"
    prescription_tokens = _tok(prescription)
    max_context = 200
    remaining = max_context - prescription_tokens

    chunks = [
        _chunk("A" * 50, distance=0.1),
        _chunk("B" * 50, distance=0.2),
        _chunk("C" * 50, distance=0.3),
    ]
    kept = trim_to_budget(chunks, max_tokens=remaining)

    # Total tokens = prescription + kept chunks must not exceed max_context
    total_tokens = prescription_tokens + sum(_tok(c["text"]) for c in kept)
    assert total_tokens <= max_context


def test_prescription_summary_present_in_assembled_context():
    """Prescription text must appear in the context string regardless of trimming."""
    prescription = "[Prescription]\n1. Metformin\n   • Dosage: 500mg"
    chunks = [_chunk("chunk text", distance=0.1)]
    remaining = 10_000 - _tok(prescription)
    kept = trim_to_budget(chunks, max_tokens=remaining)

    context_parts = [prescription] + [
        f"[{c['drug_name']} — {c['section']}]\n{c['text']}" for c in kept
    ]
    context = "\n\n".join(context_parts)

    assert prescription in context


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty():
    assert trim_to_budget([], max_tokens=1000) == []


def test_zero_budget_returns_empty():
    chunks = [_chunk("hello", distance=0.1)]
    assert trim_to_budget(chunks, max_tokens=0) == []


# ---------------------------------------------------------------------------
# trim_to_budget — rerank_score sort
# ---------------------------------------------------------------------------


def _chunk_reranked(text: str, rerank_score: float, distance: float = 0.3) -> dict:
    return {
        "text": text,
        "distance": distance,
        "drug_name": "drugA",
        "section": "dosage",
        "rerank_score": rerank_score,
    }


def test_trim_to_budget_sorts_by_rerank_score_when_present():
    chunks = [
        _chunk_reranked(
            "A" * 100, rerank_score=0.3, distance=0.1
        ),  # highest distance-rank but lowest rerank
        _chunk_reranked("B" * 100, rerank_score=0.9, distance=0.4),  # highest rerank
        _chunk_reranked("C" * 100, rerank_score=0.6, distance=0.2),
    ]
    # budget fits only the two highest-reranked chunks
    budget = _tok("B" * 100) + _tok("C" * 100)
    result = trim_to_budget(chunks, max_tokens=budget)

    assert len(result) == 2
    assert result[0]["rerank_score"] == pytest.approx(0.9)
    assert result[1]["rerank_score"] == pytest.approx(0.6)


def test_trim_to_budget_rerank_sort_highest_first():
    chunks = [
        _chunk_reranked("X" * 50, rerank_score=0.1),
        _chunk_reranked("Y" * 50, rerank_score=0.8),
        _chunk_reranked("Z" * 50, rerank_score=0.5),
    ]
    result = trim_to_budget(chunks, max_tokens=10_000)

    scores = [c["rerank_score"] for c in result]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# _build_fallback_message
# ---------------------------------------------------------------------------


async def test_fallback_message_includes_indexed_drugs():
    with patch(
        "app.services.rag_pipeline.get_upload_result",
        return_value=(["aspirin", "metformin"], []),
    ):
        msg = await _build_fallback_message("session-id")

    assert "aspirin, metformin" in msg
    assert "Drugs indexed:" in msg


async def test_fallback_message_includes_missing_drugs():
    with patch(
        "app.services.rag_pipeline.get_upload_result",
        return_value=(["aspirin"], ["ibuprofen"]),
    ):
        msg = await _build_fallback_message("session-id")

    assert "ibuprofen" in msg
    assert "Drugs with no leaflet found:" in msg


async def test_fallback_message_suggests_rephrasing():
    with patch(
        "app.services.rag_pipeline.get_upload_result",
        return_value=(["aspirin"], []),
    ):
        msg = await _build_fallback_message("session-id")

    assert "Try rephrasing" in msg
    assert "warnings" in msg
    assert "dosage" in msg
    assert "interactions" in msg


async def test_fallback_message_empty_lists_show_none():
    with patch(
        "app.services.rag_pipeline.get_upload_result",
        return_value=([], []),
    ):
        msg = await _build_fallback_message("session-id")

    assert "Drugs indexed: none" in msg
    assert "Drugs with no leaflet found: none" in msg


async def test_fallback_message_starts_with_user_facing_text():
    with patch(
        "app.services.rag_pipeline.get_upload_result",
        return_value=(["drugA"], []),
    ):
        msg = await _build_fallback_message("session-id")

    assert msg.startswith("I couldn't find relevant information")
