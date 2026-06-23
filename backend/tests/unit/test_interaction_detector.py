from unittest.mock import patch

import chromadb
import pytest

from app.models.schemas import InteractionFlag
from app.services.interaction_detector import (
    MAX_INTERACTION_DRUGS,
    _extract_excerpt,
    _word_in_text,
    detect_interactions,
)
from app.services.vector_store import store

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedding(value: float = 0.1, dim: int = 384) -> list[float]:
    return [value] * dim


@pytest.fixture
def chroma_client():
    return chromadb.EphemeralClient()


async def _store_interaction_chunk(
    session_id: str, drug_name: str, text: str, client
) -> None:
    await store(
        [text],
        [_make_embedding()],
        [{"drug_name": drug_name, "section": "drug_interactions"}],
        session_id=session_id,
        client=client,
    )


# ---------------------------------------------------------------------------
# _word_in_text
# ---------------------------------------------------------------------------


def test_word_in_text_finds_exact_match():
    assert _word_in_text("warfarin", "Do not take with warfarin.") is True


def test_word_in_text_case_insensitive():
    assert _word_in_text("Warfarin", "avoid warfarin use") is True


def test_word_in_text_whole_word_only():
    # "warfarin" should not match inside "warfarinoid"
    assert _word_in_text("warfarin", "warfarinoid compound") is False


def test_word_in_text_not_found():
    assert _word_in_text("aspirin", "Take with food only.") is False


# ---------------------------------------------------------------------------
# _extract_excerpt
# ---------------------------------------------------------------------------


def test_extract_excerpt_returns_string():
    result = _extract_excerpt("Do not use with warfarin.", "warfarin")
    assert isinstance(result, str)
    assert len(result) > 0


def test_extract_excerpt_contains_term():
    result = _extract_excerpt(
        "This drug interacts with warfarin and may cause bleeding.", "warfarin"
    )
    assert "warfarin" in result.lower()


def test_extract_excerpt_short_text_returned_in_full():
    text = "Avoid warfarin."
    result = _extract_excerpt(text, "warfarin")
    assert "warfarin" in result.lower()


def test_extract_excerpt_term_not_found_returns_start():
    result = _extract_excerpt("Some long text " * 30, "missing")
    assert len(result) <= 300


# ---------------------------------------------------------------------------
# detect_interactions — using ephemeral ChromaDB
# ---------------------------------------------------------------------------


async def test_detect_interactions_flags_cross_mention(chroma_client):
    sid = "sess-int-1"
    await _store_interaction_chunk(
        sid,
        "warfarin",
        "Concurrent use of aspirin with warfarin increases bleeding risk.",
        chroma_client,
    )
    await _store_interaction_chunk(
        sid,
        "aspirin",
        "No specific interaction data for warfarin found in this section.",
        chroma_client,
    )

    with patch(
        "app.services.interaction_detector.get_upload_result",
        return_value=(["warfarin", "aspirin"], []),
    ):
        flags = await detect_interactions(sid, client=chroma_client)

    assert len(flags) >= 1
    assert any(f.drug_a == "warfarin" and f.drug_b == "aspirin" for f in flags)


async def test_detect_interactions_returns_interaction_flag_objects(chroma_client):
    sid = "sess-int-2"
    await _store_interaction_chunk(
        sid,
        "lisinopril",
        "Avoid concurrent use with metformin in renal impairment.",
        chroma_client,
    )

    with patch(
        "app.services.interaction_detector.get_upload_result",
        return_value=(["lisinopril", "metformin"], []),
    ):
        flags = await detect_interactions(sid, client=chroma_client)

    assert all(isinstance(f, InteractionFlag) for f in flags)
    assert all(f.excerpt for f in flags)


async def test_detect_interactions_single_drug_returns_empty(chroma_client):
    with patch(
        "app.services.interaction_detector.get_upload_result",
        return_value=(["aspirin"], []),
    ):
        flags = await detect_interactions("any-session", client=chroma_client)
    assert flags == []


async def test_detect_interactions_no_drugs_returns_empty(chroma_client):
    with patch(
        "app.services.interaction_detector.get_upload_result",
        return_value=([], []),
    ):
        flags = await detect_interactions("any-session", client=chroma_client)
    assert flags == []


async def test_detect_interactions_no_cross_mention_returns_empty(chroma_client):
    sid = "sess-int-3"
    await _store_interaction_chunk(
        sid,
        "lisinopril",
        "May interact with NSAIDs in general.",  # no specific drug name
        chroma_client,
    )

    with patch(
        "app.services.interaction_detector.get_upload_result",
        return_value=(["lisinopril", "metformin"], []),
    ):
        flags = await detect_interactions(sid, client=chroma_client)
    assert flags == []


async def test_detect_interactions_skips_when_too_many_drugs(chroma_client):
    """Above MAX_INTERACTION_DRUGS the quadratic scan is skipped, returning []."""
    too_many = [f"drug{i}" for i in range(MAX_INTERACTION_DRUGS + 1)]

    with (
        patch(
            "app.services.interaction_detector.get_upload_result",
            return_value=(too_many, []),
        ) as mock_upload,
        patch(
            "app.services.interaction_detector.get_by_section"
        ) as mock_get_by_section,
    ):
        flags = await detect_interactions("big-session", client=chroma_client)

    assert flags == []
    # The scan must be short-circuited before any per-drug chunk read.
    mock_get_by_section.assert_not_called()
    mock_upload.assert_awaited_once()


async def test_detect_interactions_deduplicates_pair(chroma_client):
    """Two chunks both mentioning drug_b should produce only one flag per pair."""
    sid = "sess-int-4"
    await _store_interaction_chunk(
        sid, "warfarin", "First mention of aspirin here.", chroma_client
    )
    await _store_interaction_chunk(
        sid, "warfarin", "Second mention: aspirin may increase INR.", chroma_client
    )

    with patch(
        "app.services.interaction_detector.get_upload_result",
        return_value=(["warfarin", "aspirin"], []),
    ):
        flags = await detect_interactions(sid, client=chroma_client)

    warfarin_aspirin = [
        f for f in flags if f.drug_a == "warfarin" and f.drug_b == "aspirin"
    ]
    assert len(warfarin_aspirin) == 1
