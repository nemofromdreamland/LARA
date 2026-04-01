import chromadb
import pytest

from app.services.vector_store import delete_session, retrieve, store


@pytest.fixture
def chroma_client():
    """Ephemeral in-memory Chroma client — isolated per test."""
    return chromadb.EphemeralClient()


def _make_embedding(value: float, dim: int = 384) -> list[float]:
    return [value] * dim


def test_store_and_retrieve_basic(chroma_client):
    chunks = ["Lisinopril is used for hypertension."]
    embeddings = [_make_embedding(0.1)]
    metadatas = [
        {"session_id": "sess-1", "drug_name": "lisinopril", "section": "indications"}
    ]

    store(chunks, embeddings, metadatas, client=chroma_client)

    results = retrieve(_make_embedding(0.1), "sess-1", top_k=1, client=chroma_client)
    assert len(results) == 1
    assert results[0]["text"] == chunks[0]
    assert results[0]["drug_name"] == "lisinopril"
    assert results[0]["section"] == "indications"


def test_session_isolation(chroma_client):
    """Documents from session A must not appear in session B results."""
    store(
        ["Drug A info."],
        [_make_embedding(0.5)],
        [{"session_id": "sess-A", "drug_name": "drugA", "section": "indications"}],
        client=chroma_client,
    )
    store(
        ["Drug B info."],
        [_make_embedding(0.5)],
        [{"session_id": "sess-B", "drug_name": "drugB", "section": "indications"}],
        client=chroma_client,
    )

    results_a = retrieve(_make_embedding(0.5), "sess-A", top_k=5, client=chroma_client)
    assert all(r["drug_name"] == "drugA" for r in results_a)

    results_b = retrieve(_make_embedding(0.5), "sess-B", top_k=5, client=chroma_client)
    assert all(r["drug_name"] == "drugB" for r in results_b)


def test_store_multiple_chunks(chroma_client):
    chunks = [f"Chunk {i}" for i in range(5)]
    embeddings = [_make_embedding(0.1 * i) for i in range(5)]
    metadatas = [
        {"session_id": "sess-2", "drug_name": "metformin", "section": "dosage"}
        for _ in range(5)
    ]
    store(chunks, embeddings, metadatas, client=chroma_client)

    results = retrieve(_make_embedding(0.2), "sess-2", top_k=5, client=chroma_client)
    assert len(results) == 5


def test_result_keys(chroma_client):
    store(
        ["Some text."],
        [_make_embedding(0.3)],
        [{"session_id": "sess-3", "drug_name": "aspirin", "section": "warnings"}],
        client=chroma_client,
    )
    results = retrieve(_make_embedding(0.3), "sess-3", top_k=1, client=chroma_client)
    assert set(results[0].keys()) == {"text", "drug_name", "section"}


def test_retrieve_empty_session_returns_empty(chroma_client):
    results = retrieve(
        _make_embedding(0.1), "nonexistent-session", top_k=5, client=chroma_client
    )
    assert results == []


def test_delete_session_removes_all_docs(chroma_client):
    meta = {"session_id": "sess-del", "drug_name": "warfarin", "section": "warnings"}
    store(["Chunk A."], [_make_embedding(0.1)], [meta], client=chroma_client)
    store(["Chunk B."], [_make_embedding(0.2)], [meta], client=chroma_client)

    deleted = delete_session("sess-del", client=chroma_client)
    assert deleted == 2

    results = retrieve(_make_embedding(0.1), "sess-del", top_k=5, client=chroma_client)
    assert results == []


def test_delete_session_nonexistent_returns_zero(chroma_client):
    deleted = delete_session("ghost-session", client=chroma_client)
    assert deleted == 0


def test_delete_session_does_not_affect_other_sessions(chroma_client):
    store(
        ["Keep this."],
        [_make_embedding(0.5)],
        [{"session_id": "sess-keep", "drug_name": "aspirin", "section": "dosage"}],
        client=chroma_client,
    )
    store(
        ["Delete this."],
        [_make_embedding(0.5)],
        [{"session_id": "sess-gone", "drug_name": "aspirin", "section": "dosage"}],
        client=chroma_client,
    )

    delete_session("sess-gone", client=chroma_client)

    kept = retrieve(_make_embedding(0.5), "sess-keep", top_k=5, client=chroma_client)
    assert len(kept) == 1


def test_store_accumulates_across_calls(chroma_client):
    meta = {"session_id": "sess-4", "drug_name": "ibuprofen", "section": "dosage"}
    store(["First chunk."], [_make_embedding(0.1)], [meta], client=chroma_client)
    store(["Second chunk."], [_make_embedding(0.2)], [meta], client=chroma_client)

    results = retrieve(_make_embedding(0.15), "sess-4", top_k=5, client=chroma_client)
    assert len(results) == 2
