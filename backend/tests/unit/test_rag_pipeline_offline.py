"""Offline unit tests for the rag_pipeline orchestration layer.

Covers answer(), answer_stream(), _retrieve_diverse() and the
_prepare_context() branches (reranker toggle, prescription block, distance
threshold, multi-drug diverse retrieval) with every model/LLM/store seam
mocked — runs in the default CI suite with HF_HUB_OFFLINE=1.
"""

import json
from unittest.mock import AsyncMock, patch

from app.models.schemas import PrescriptionEntry
from app.services.rag_pipeline import _retrieve_diverse, answer, answer_stream


def _chunk(
    text: str, distance: float = 0.1, drug: str = "lisinopril", section: str = "dosage"
) -> dict:
    return {"text": text, "distance": distance, "drug_name": drug, "section": section}


def _patches(**overrides):
    """Patch every rag_pipeline seam with sensible single-drug defaults.

    Returns a dict of name → mock; use overrides to customise return values.
    """
    defaults = {
        "embed": AsyncMock(return_value=[[0.1] * 768]),
        "retrieve": AsyncMock(return_value=[_chunk("Take once daily.")]),
        "rerank": AsyncMock(side_effect=lambda q, chunks, executor=None: chunks),
        "generate": AsyncMock(
            return_value="Take once daily.\nCITED: lisinopril/dosage"
        ),
        "get_upload_result": AsyncMock(return_value=(["lisinopril"], [])),
        "get_prescription_entries": AsyncMock(return_value=[]),
    }
    defaults.update(overrides)
    return defaults


class _SeamPatcher:
    """Context manager applying all rag_pipeline seam patches at once."""

    def __init__(self, mocks: dict):
        self.mocks = mocks
        self._patchers = [
            patch(f"app.services.rag_pipeline.{name}", mock)
            for name, mock in mocks.items()
        ]

    def __enter__(self):
        for p in self._patchers:
            p.start()
        return self.mocks

    def __exit__(self, *exc):
        for p in self._patchers:
            p.stop()
        return False


# ---------------------------------------------------------------------------
# answer — happy path and citation filtering
# ---------------------------------------------------------------------------


async def test_answer_happy_path_strips_cited_and_filters_sources():
    chunks = [
        _chunk("Take once daily.", distance=0.1, section="dosage"),
        _chunk("Do not use in pregnancy.", distance=0.2, section="warnings"),
    ]
    mocks = _patches(retrieve=AsyncMock(return_value=chunks))

    with _SeamPatcher(mocks):
        result = await answer("sess-1", "How often?")

    assert result.answer == "Take once daily."
    # Only the cited (drug, section) pair survives source filtering.
    assert len(result.sources) == 1
    assert result.sources[0].drug_name == "lisinopril"
    assert result.sources[0].section == "dosage"


async def test_answer_cited_none_returns_all_retrieved_sources():
    chunks = [
        _chunk("Take once daily.", distance=0.1, section="dosage"),
        _chunk("Do not use in pregnancy.", distance=0.2, section="warnings"),
    ]
    mocks = _patches(
        retrieve=AsyncMock(return_value=chunks),
        generate=AsyncMock(return_value="General answer.\nCITED: none"),
    )

    with _SeamPatcher(mocks):
        result = await answer("sess-1", "Tell me about this drug")

    assert result.answer == "General answer."
    sections = {s.section for s in result.sources}
    assert sections == {"dosage", "warnings"}


async def test_answer_no_chunks_returns_fallback_without_calling_llm():
    mocks = _patches(retrieve=AsyncMock(return_value=[]))

    with _SeamPatcher(mocks):
        result = await answer("sess-1", "Anything?")

    mocks["generate"].assert_not_awaited()
    assert result.answer.startswith("I couldn't find relevant information")
    assert result.sources == []


async def test_answer_enriches_embedding_query_with_last_user_turn():
    history = [
        {"role": "user", "content": "What is the dosage?"},
        {"role": "assistant", "content": "Once daily."},
    ]
    mocks = _patches()

    with _SeamPatcher(mocks):
        await answer("sess-1", "What about pregnancy?", history=history)

    mocks["embed"].assert_awaited_once_with(
        ["What is the dosage? What about pregnancy?"], None, source="query"
    )
    # The LLM receives the conversation history for multi-turn coherence.
    assert mocks["generate"].await_args.kwargs["history"] == history


# ---------------------------------------------------------------------------
# answer — _prepare_context branches
# ---------------------------------------------------------------------------


async def test_answer_drops_chunks_beyond_distance_threshold():
    """Chunks at or above the 0.65 distance threshold must not reach the LLM."""
    mocks = _patches(
        retrieve=AsyncMock(return_value=[_chunk("Irrelevant.", distance=0.9)])
    )

    with _SeamPatcher(mocks):
        result = await answer("sess-1", "Anything?")

    mocks["generate"].assert_not_awaited()
    assert result.answer.startswith("I couldn't find relevant information")


async def test_answer_reranker_disabled_skips_rerank():
    mocks = _patches()

    with (
        _SeamPatcher(mocks),
        patch("app.services.rag_pipeline.settings.reranker_enabled", False),
    ):
        result = await answer("sess-1", "How often?")

    mocks["rerank"].assert_not_awaited()
    assert result.answer == "Take once daily."


async def test_answer_includes_prescription_block_in_context():
    entries = [
        PrescriptionEntry(drug_name="lisinopril", dosage="10mg", frequency="daily")
    ]
    mocks = _patches(get_prescription_entries=AsyncMock(return_value=entries))

    with _SeamPatcher(mocks):
        await answer("sess-1", "How often?")

    context = mocks["generate"].await_args.args[0]
    assert "[Prescription]" in context
    assert "1. Lisinopril" in context
    assert "Dosage: 10mg" in context
    assert "Frequency: daily" in context
    # Retrieved chunks are labelled [drug — section] after the prescription.
    assert "[lisinopril — dosage]\nTake once daily." in context


async def test_answer_multi_drug_session_retrieves_per_drug():
    """With >1 indexed drug, retrieval runs once per drug (diverse branch)."""
    mocks = _patches(
        get_upload_result=AsyncMock(return_value=(["lisinopril", "metformin"], [])),
        retrieve=AsyncMock(
            side_effect=[
                [_chunk("Lisinopril info.", drug="lisinopril")],
                [_chunk("Metformin info.", distance=0.2, drug="metformin")],
            ]
        ),
    )

    with _SeamPatcher(mocks):
        await answer("sess-1", "Any interactions?")

    drug_filters = {
        call.kwargs["drug_name"] for call in mocks["retrieve"].await_args_list
    }
    assert drug_filters == {"lisinopril", "metformin"}
    context = mocks["generate"].await_args.args[0]
    assert "Lisinopril info." in context
    assert "Metformin info." in context


# ---------------------------------------------------------------------------
# _retrieve_diverse
# ---------------------------------------------------------------------------


async def test_retrieve_diverse_merges_dedupes_and_sorts_by_distance():
    per_drug = {
        "drugA": [_chunk("shared text", 0.3, drug="drugA"), _chunk("a only", 0.5)],
        "drugB": [_chunk("shared text", 0.4, drug="drugB"), _chunk("b only", 0.1)],
    }

    async def _fake_retrieve(embedding, session_id, top_k, drug_name):
        return per_drug[drug_name]

    with patch(
        "app.services.rag_pipeline.retrieve", AsyncMock(side_effect=_fake_retrieve)
    ):
        merged = await _retrieve_diverse([0.1] * 768, "sess-1", ["drugA", "drugB"])

    texts = [c["text"] for c in merged]
    assert texts.count("shared text") == 1  # deduplicated on text
    distances = [c["distance"] for c in merged]
    assert distances == sorted(distances)


async def test_retrieve_diverse_splits_top_k_across_drugs():
    """per-drug k = max(4, ceil(top_k / n_drugs)) — 20 across 2 drugs → 10 each."""
    mock_retrieve = AsyncMock(return_value=[])

    with patch("app.services.rag_pipeline.retrieve", mock_retrieve):
        await _retrieve_diverse([0.1] * 768, "sess-1", ["drugA", "drugB"])

    assert mock_retrieve.await_count == 2
    assert all(c.kwargs["top_k"] == 10 for c in mock_retrieve.await_args_list)


# ---------------------------------------------------------------------------
# answer_stream
# ---------------------------------------------------------------------------


def _fake_stream(*tokens: str):
    async def _gen(*args, **kwargs):
        for t in tokens:
            yield t

    return _gen


async def test_answer_stream_yields_tokens_then_sources_then_done():
    mocks = _patches()
    del mocks["generate"]  # streaming path uses generate_stream
    mocks["generate_stream"] = _fake_stream("Take ", "once daily.")

    with _SeamPatcher(mocks):
        payloads = [p async for p in answer_stream("sess-1", "How often?")]

    assert payloads[:2] == ["Take ", "once daily."]
    assert payloads[2].startswith("[SOURCES]")
    sources = json.loads(payloads[2][len("[SOURCES]") :])["sources"]
    assert sources == [{"drug_name": "lisinopril", "section": "dosage"}]
    assert payloads[3] == "[DONE]"
    assert len(payloads) == 4


async def test_answer_stream_no_chunks_streams_fallback_then_empty_sources():
    mocks = _patches(retrieve=AsyncMock(return_value=[]))
    del mocks["generate"]
    mocks["generate_stream"] = _fake_stream("never used")

    with _SeamPatcher(mocks):
        payloads = [p async for p in answer_stream("sess-1", "Anything?")]

    assert payloads[0].startswith("I couldn't find relevant information")
    assert json.loads(payloads[1][len("[SOURCES]") :]) == {"sources": []}
    assert payloads[2] == "[DONE]"
    assert len(payloads) == 3


async def test_answer_stream_deduplicates_sources_across_chunks():
    chunks = [
        _chunk("first dosage chunk", 0.1),
        _chunk("second dosage chunk", 0.2),
        _chunk("warnings chunk", 0.3, section="warnings"),
    ]
    mocks = _patches(retrieve=AsyncMock(return_value=chunks))
    del mocks["generate"]
    mocks["generate_stream"] = _fake_stream("Answer.")

    with _SeamPatcher(mocks):
        payloads = [p async for p in answer_stream("sess-1", "How often?")]

    sources = json.loads(payloads[-2][len("[SOURCES]") :])["sources"]
    assert sources == [
        {"drug_name": "lisinopril", "section": "dosage"},
        {"drug_name": "lisinopril", "section": "warnings"},
    ]
