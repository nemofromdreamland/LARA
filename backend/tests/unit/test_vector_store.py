from unittest.mock import patch

import chromadb
import httpx
import pytest

from app.services.vector_store import (
    delete_session,
    get_by_section,
    ping,
    retrieve,
    session_id_from_collection,
    store,
)


@pytest.fixture
def chroma_client():
    """Ephemeral in-memory Chroma client — isolated per test."""
    return chromadb.EphemeralClient()


def _make_embedding(value: float, dim: int = 384) -> list[float]:
    return [value] * dim


async def test_store_and_retrieve_basic(chroma_client):
    chunks = ["Lisinopril is used for hypertension."]
    embeddings = [_make_embedding(0.1)]
    metadatas = [{"drug_name": "lisinopril", "section": "indications"}]

    await store(
        chunks, embeddings, metadatas, session_id="sess-1", client=chroma_client
    )

    results = await retrieve(
        _make_embedding(0.1), "sess-1", top_k=1, client=chroma_client
    )
    assert len(results) == 1
    assert results[0]["text"] == chunks[0]
    assert results[0]["drug_name"] == "lisinopril"
    assert results[0]["section"] == "indications"


async def test_session_isolation(chroma_client):
    """Documents from session A must not appear in session B results."""
    await store(
        ["Drug A info."],
        [_make_embedding(0.5)],
        [{"drug_name": "drugA", "section": "indications"}],
        session_id="sess-A",
        client=chroma_client,
    )
    await store(
        ["Drug B info."],
        [_make_embedding(0.5)],
        [{"drug_name": "drugB", "section": "indications"}],
        session_id="sess-B",
        client=chroma_client,
    )

    results_a = await retrieve(
        _make_embedding(0.5), "sess-A", top_k=5, client=chroma_client
    )
    assert all(r["drug_name"] == "drugA" for r in results_a)

    results_b = await retrieve(
        _make_embedding(0.5), "sess-B", top_k=5, client=chroma_client
    )
    assert all(r["drug_name"] == "drugB" for r in results_b)


async def test_store_multiple_chunks(chroma_client):
    chunks = [f"Chunk {i}" for i in range(5)]
    embeddings = [_make_embedding(0.1 * i) for i in range(5)]
    metadatas = [{"drug_name": "metformin", "section": "dosage"} for _ in range(5)]
    await store(
        chunks, embeddings, metadatas, session_id="sess-2", client=chroma_client
    )

    results = await retrieve(
        _make_embedding(0.2), "sess-2", top_k=5, client=chroma_client
    )
    assert len(results) == 5


async def test_result_keys(chroma_client):
    await store(
        ["Some text."],
        [_make_embedding(0.3)],
        [{"drug_name": "aspirin", "section": "warnings"}],
        session_id="sess-3",
        client=chroma_client,
    )
    results = await retrieve(
        _make_embedding(0.3), "sess-3", top_k=1, client=chroma_client
    )
    assert set(results[0].keys()) == {"text", "drug_name", "section", "distance"}


async def test_retrieve_returns_distances(chroma_client):
    """retrieve() must include a numeric 'distance' field on every result."""
    await store(
        ["Aspirin is used for pain relief."],
        [_make_embedding(0.5)],
        [{"drug_name": "aspirin", "section": "indications"}],
        session_id="sess-dist",
        client=chroma_client,
    )
    results = await retrieve(
        _make_embedding(0.5), "sess-dist", top_k=1, client=chroma_client
    )
    assert len(results) == 1
    assert "distance" in results[0]
    assert isinstance(results[0]["distance"], float)


async def test_retrieve_empty_session_returns_empty(chroma_client):
    results = await retrieve(
        _make_embedding(0.1), "nonexistent-session", top_k=5, client=chroma_client
    )
    assert results == []


async def test_delete_session_removes_all_docs(chroma_client):
    meta = {"drug_name": "warfarin", "section": "warnings"}
    await store(
        ["Chunk A."],
        [_make_embedding(0.1)],
        [meta],
        session_id="sess-del",
        client=chroma_client,
    )
    await store(
        ["Chunk B."],
        [_make_embedding(0.2)],
        [meta],
        session_id="sess-del",
        client=chroma_client,
    )

    deleted = await delete_session("sess-del", client=chroma_client)
    assert deleted == 2

    results = await retrieve(
        _make_embedding(0.1), "sess-del", top_k=5, client=chroma_client
    )
    assert results == []


async def test_delete_session_nonexistent_returns_zero(chroma_client):
    deleted = await delete_session("ghost-session", client=chroma_client)
    assert deleted == 0


async def test_delete_session_does_not_affect_other_sessions(chroma_client):
    await store(
        ["Keep this."],
        [_make_embedding(0.5)],
        [{"drug_name": "aspirin", "section": "dosage"}],
        session_id="sess-keep",
        client=chroma_client,
    )
    await store(
        ["Delete this."],
        [_make_embedding(0.5)],
        [{"drug_name": "aspirin", "section": "dosage"}],
        session_id="sess-gone",
        client=chroma_client,
    )

    await delete_session("sess-gone", client=chroma_client)

    kept = await retrieve(
        _make_embedding(0.5), "sess-keep", top_k=5, client=chroma_client
    )
    assert len(kept) == 1


async def test_get_by_section_returns_matching_chunks(chroma_client):
    await store(
        ["Interacts with warfarin."],
        [_make_embedding(0.1)],
        [{"drug_name": "aspirin", "section": "drug_interactions"}],
        session_id="gs-1",
        client=chroma_client,
    )
    await store(
        ["Take with food."],
        [_make_embedding(0.2)],
        [{"drug_name": "aspirin", "section": "dosage"}],
        session_id="gs-1",
        client=chroma_client,
    )
    results = await get_by_section("gs-1", "drug_interactions", client=chroma_client)
    assert len(results) == 1
    assert "warfarin" in results[0]["text"]


async def test_get_by_section_filters_by_drug_name(chroma_client):
    for drug in ["aspirin", "warfarin"]:
        await store(
            [f"{drug} interaction info."],
            [_make_embedding(0.1)],
            [{"drug_name": drug, "section": "drug_interactions"}],
            session_id="gs-2",
            client=chroma_client,
        )
    results = await get_by_section(
        "gs-2", "drug_interactions", drug_name="aspirin", client=chroma_client
    )
    assert len(results) == 1
    assert results[0]["drug_name"] == "aspirin"


async def test_get_by_section_empty_when_no_match(chroma_client):
    results = await get_by_section(
        "nonexistent", "drug_interactions", client=chroma_client
    )
    assert results == []


async def test_store_accumulates_across_calls(chroma_client):
    meta = {"drug_name": "ibuprofen", "section": "dosage"}
    await store(
        ["First chunk."],
        [_make_embedding(0.1)],
        [meta],
        session_id="sess-4",
        client=chroma_client,
    )
    await store(
        ["Second chunk."],
        [_make_embedding(0.2)],
        [meta],
        session_id="sess-4",
        client=chroma_client,
    )

    results = await retrieve(
        _make_embedding(0.15), "sess-4", top_k=5, client=chroma_client
    )
    assert len(results) == 2


async def test_retrieve_filters_by_drug_name(chroma_client):
    await store(
        ["Aspirin chunk.", "Metformin chunk."],
        [_make_embedding(0.1), _make_embedding(0.1)],
        [
            {"drug_name": "aspirin", "section": "dosage"},
            {"drug_name": "metformin", "section": "dosage"},
        ],
        session_id="sess-5",
        client=chroma_client,
    )

    results = await retrieve(
        _make_embedding(0.1),
        "sess-5",
        top_k=5,
        drug_name="aspirin",
        client=chroma_client,
    )
    assert len(results) == 1
    assert results[0]["drug_name"] == "aspirin"


def test_session_id_from_collection_roundtrip():
    sid = "11111111-2222-3333-4444-555555555555"
    assert session_id_from_collection(f"session_{sid.replace('-', '_')}") == sid
    assert session_id_from_collection("leaflets") is None


async def test_idempotent_ops_retry_transient_errors():
    """A ConnectError on the first attempt is retried and succeeds."""
    calls = {"n": 0}

    async def _flaky_run_sync(fn, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("chroma hiccup")
        return 1  # heartbeat payload

    class _DummyClient:
        def heartbeat(self):  # pragma: no cover - replaced by _flaky_run_sync
            return 1

    with (
        patch("app.services.vector_store._get_client", return_value=_DummyClient()),
        patch("app.services.vector_store.run_sync", side_effect=_flaky_run_sync),
    ):
        assert await ping() is True
    assert calls["n"] == 2
