from uuid import uuid4

import chromadb
from chromadb import ClientAPI

from app.config import settings
from app.utils import run_sync

_COLLECTION_NAME = "leaflets"
_client: ClientAPI | None = None


def _get_client() -> ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return _client


def _get_collection(client: ClientAPI | None = None) -> chromadb.Collection:
    c = client or _get_client()
    return c.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


async def ping() -> bool:
    try:
        collection = _get_collection()
        await run_sync(collection.count)
        return True
    except Exception:
        return False


async def store(
    chunks: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    client: ClientAPI | None = None,
) -> None:
    """Persist *chunks* with their *embeddings* and *metadatas* in Chroma.

    Each metadata dict must contain at minimum: session_id, drug_name, section.
    IDs are random UUIDs to guarantee uniqueness across concurrent uploads.
    """
    collection = _get_collection(client)
    ids = [str(uuid4()) for _ in chunks]
    await run_sync(
        collection.add,
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )


async def delete_session(
    session_id: str,
    client: ClientAPI | None = None,
) -> int:
    """Delete all ChromaDB documents belonging to *session_id*.

    Returns the number of documents deleted.
    """
    collection = _get_collection(client)
    existing = await run_sync(
        collection.get, where={"session_id": session_id}, include=[]
    )
    ids = existing["ids"]
    if ids:
        await run_sync(collection.delete, ids=ids)
    return len(ids)


async def list_session_ids(client: ClientAPI | None = None) -> list[str]:
    """Return all distinct session_ids present in the collection."""
    collection = _get_collection(client)
    results = await run_sync(collection.get, include=["metadatas"])
    seen: set[str] = set()
    for meta in results["metadatas"] or []:
        sid = meta.get("session_id")
        if sid:
            seen.add(sid)
    return list(seen)


async def get_by_section(
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
    results = await run_sync(
        collection.get, where=where, include=["documents", "metadatas"]
    )
    return [
        {
            "text": doc,
            "drug_name": meta["drug_name"],
            "section": meta["section"],
        }
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]


async def retrieve_for_drug(
    query_embedding: list[float],
    session_id: str,
    drug_name: str,
    top_k: int = 3,
    client: ClientAPI | None = None,
) -> list[dict]:
    """Query Chroma for top-*k* chunks scoped to *session_id* AND *drug_name*.

    Used by the per-drug retrieval strategy to guarantee each drug gets
    representation in the context regardless of global distance ranking.
    """
    collection = _get_collection(client)
    results = await run_sync(
        collection.query,
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"$and": [{"session_id": session_id}, {"drug_name": drug_name}]},
        include=["documents", "metadatas", "distances"],
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    return [
        {
            "text": doc,
            "drug_name": meta["drug_name"],
            "section": meta["section"],
            "distance": dist,
        }
        for doc, meta, dist in zip(docs, metas, distances)
    ]


async def retrieve(
    query_embedding: list[float],
    session_id: str,
    top_k: int = 5,
    client: ClientAPI | None = None,
) -> list[dict]:
    """Query Chroma for the top-*k* chunks scoped to *session_id*.

    Returns a list of dicts with keys: text, drug_name, section.
    """
    collection = _get_collection(client)
    results = await run_sync(
        collection.query,
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"session_id": session_id},
        include=["documents", "metadatas", "distances"],
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    return [
        {
            "text": doc,
            "drug_name": meta["drug_name"],
            "section": meta["section"],
            "distance": dist,
        }
        for doc, meta, dist in zip(docs, metas, distances)
    ]
