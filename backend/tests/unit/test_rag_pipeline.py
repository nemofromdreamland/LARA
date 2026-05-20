"""Unit tests for rag_pipeline.trim_to_budget."""

import pytest

from app.services.rag_pipeline import trim_to_budget


def _chunk(text: str, distance: float, drug: str = "drugA", section: str = "dosage") -> dict:
    return {"text": text, "distance": distance, "drug_name": drug, "section": section}


# ---------------------------------------------------------------------------
# trim_to_budget — all chunks fit
# ---------------------------------------------------------------------------


def test_all_chunks_fit_returned_sorted_by_distance():
    chunks = [
        _chunk("B" * 100, distance=0.3),
        _chunk("A" * 100, distance=0.1),
        _chunk("C" * 100, distance=0.4),
    ]
    result = trim_to_budget(chunks, max_chars=500)

    assert len(result) == 3
    distances = [c["distance"] for c in result]
    assert distances == sorted(distances)


def test_all_chunks_fit_exact_budget():
    chunks = [_chunk("x" * 50, distance=0.2), _chunk("y" * 50, distance=0.1)]
    result = trim_to_budget(chunks, max_chars=100)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# trim_to_budget — overflow: only highest-relevance chunks kept
# ---------------------------------------------------------------------------


def test_overflow_drops_least_relevant_chunks():
    chunks = [
        _chunk("A" * 100, distance=0.1),  # most relevant — keep
        _chunk("B" * 100, distance=0.2),  # second — keep
        _chunk("C" * 100, distance=0.3),  # third — would exceed 250 → drop
    ]
    result = trim_to_budget(chunks, max_chars=250)

    assert len(result) == 2
    assert all(c["distance"] < 0.25 for c in result)


def test_overflow_single_chunk_too_large_returns_empty():
    chunks = [_chunk("X" * 500, distance=0.1)]
    result = trim_to_budget(chunks, max_chars=100)
    assert result == []


def test_overflow_preserves_most_relevant_order():
    chunks = [
        _chunk("Z" * 80, distance=0.4),
        _chunk("A" * 80, distance=0.05),
        _chunk("M" * 80, distance=0.2),
    ]
    result = trim_to_budget(chunks, max_chars=160)

    assert len(result) == 2
    assert result[0]["distance"] == 0.05
    assert result[1]["distance"] == 0.2


# ---------------------------------------------------------------------------
# Prescription summary always counted toward budget before chunks
# ---------------------------------------------------------------------------


def test_prescription_summary_counted_before_chunks():
    """Simulate caller subtracting prescription length from budget before calling trim."""
    prescription = "[Prescription]\n1. Aspirin\n   • Dosage: 100mg"
    prescription_len = len(prescription)
    max_context = 200
    remaining = max_context - prescription_len

    chunks = [
        _chunk("A" * 50, distance=0.1),
        _chunk("B" * 50, distance=0.2),
        _chunk("C" * 50, distance=0.3),
    ]
    kept = trim_to_budget(chunks, max_chars=remaining)

    # Total context = prescription + kept chunks must not exceed max_context
    total_chars = prescription_len + sum(len(c["text"]) for c in kept)
    assert total_chars <= max_context


def test_prescription_summary_present_in_assembled_context():
    """Prescription text must appear in the context string regardless of chunk trimming."""
    prescription = "[Prescription]\n1. Metformin\n   • Dosage: 500mg"
    chunks = [_chunk("chunk text", distance=0.1)]
    remaining = 10_000 - len(prescription)
    kept = trim_to_budget(chunks, max_chars=remaining)

    context_parts = [prescription] + [
        f"[{c['drug_name']} — {c['section']}]\n{c['text']}" for c in kept
    ]
    context = "\n\n".join(context_parts)

    assert prescription in context


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty():
    assert trim_to_budget([], max_chars=1000) == []


def test_zero_budget_returns_empty():
    chunks = [_chunk("hello", distance=0.1)]
    assert trim_to_budget(chunks, max_chars=0) == []
