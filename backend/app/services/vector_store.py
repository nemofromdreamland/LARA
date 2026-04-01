import chromadb
from chromadb import ClientAPI

from app.config import settings

_COLLECTION_NAME = "leaflets"
_client: ClientAPI | None = None


def _get_client() -> ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_path)
    return _client


def _get_collection(client: ClientAPI | None = None) -> chromadb.Collection:
    c = client or _get_client()
    return c.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def store(
    chunks: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    client: ClientAPI | None = None,
) -> None:
    """Persist *chunks* with their *embeddings* and *metadatas* in Chroma.

    Each metadata dict must contain at minimum: session_id, drug_name, section.
    IDs are derived from session_id + positional index to avoid collisions.
    """
    collection = _get_collection(client)
    session_id = metadatas[0]["session_id"]
    existing = collection.get(where={"session_id": session_id})
    offset = len(existing["ids"])
    ids = [f"{session_id}_{offset + i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )


def delete_session(
    session_id: str,
    client: ClientAPI | None = None,
) -> int:
    """Delete all ChromaDB documents belonging to *session_id*.

    Returns the number of documents deleted.
    """
    collection = _get_collection(client)
    existing = collection.get(where={"session_id": session_id})
    ids = existing["ids"]
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def get_by_section(
    session_id: str,
    section: str,
    drug_name: str | None = None,
    client: ClientAPI | None = None,
) -> list[dict]:
    """Return all stored chunks matching *session_id* + *section*.

    Optionally further filtered by *drug_name*.
    Returns a list of dicts with keys: text, drug_name, section.
    No semantic ranking — pure metadata filter.
    """
    collection = _get_collection(client)
    where: dict = {
        "$and": [
            {"session_id": session_id},
            {"section": section},
        ]
    }
    if drug_name is not None:
        where["$and"].append({"drug_name": drug_name})
    results = collection.get(where=where, include=["documents", "metadatas"])
    return [
        {
            "text": doc,
            "drug_name": meta["drug_name"],
            "section": meta["section"],
        }
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]


def retrieve(
    query_embedding: list[float],
    session_id: str,
    top_k: int = 5,
    client: ClientAPI | None = None,
) -> list[dict]:
    """Query Chroma for the top-*k* chunks scoped to *session_id*.

    Returns a list of dicts with keys: text, drug_name, section.
    """
    collection = _get_collection(client)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"session_id": session_id},
        include=["documents", "metadatas"],
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return [
        {
            "text": doc,
            "drug_name": meta["drug_name"],
            "section": meta["section"],
        }
        for doc, meta in zip(docs, metas)
    ]
