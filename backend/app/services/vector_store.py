from uuid import uuid4

import chromadb
import httpx
from chromadb import ClientAPI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings
from app.utils import run_sync

_client: ClientAPI | None = None

_COLLECTION_PREFIX = "session_"


def _get_client() -> ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.HttpClient(
            host=settings.chroma_host, port=settings.chroma_port
        )
    return _client


def _collection_name(session_id: str) -> str:
    return f"{_COLLECTION_PREFIX}{session_id.replace('-', '_')}"


def session_id_from_collection(name: str) -> str | None:
    """Inverse of _collection_name. Returns None for non-session collections."""
    if not name.startswith(_COLLECTION_PREFIX):
        return None
    return name[len(_COLLECTION_PREFIX) :].replace("_", "-")


def _is_transient(exc: BaseException) -> bool:
    """Network-level errors worth retrying (Chroma's HttpClient is httpx-based).

    TransportError covers ConnectError, timeouts, and RemoteProtocolError;
    HTTP status errors are not retried.
    """
    return isinstance(exc, httpx.TransportError)


# Retry wrapper for idempotent Chroma operations (get/query/count/heartbeat/
# get_or_create_collection). collection.add is deliberately NOT routed through
# this: its IDs are pre-generated, so re-adding after an ambiguous network
# failure is not idempotent (duplicate-ID error) — writes fail fast instead.
@retry(
    retry=retry_if_exception(_is_transient),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def _run_idempotent(fn, *args, **kwargs):
    return await run_sync(fn, *args, **kwargs)


async def _get_session_collection(c: ClientAPI, session_id: str):
    return await _run_idempotent(
        c.get_or_create_collection,
        _collection_name(session_id),
        metadata={"hnsw:space": "cosine"},
    )


async def ping() -> bool:
    try:
        await _run_idempotent(_get_client().heartbeat)
        return True
    except Exception:
        return False


async def store(
    chunks: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    session_id: str,
    client: ClientAPI | None = None,
) -> None:
    """Persist *chunks* with *embeddings* and *metadatas* in the session's collection.

    Each metadata dict must contain at minimum: drug_name, section.
    session_id is implicit in the collection name, not stored as metadata.
    IDs are random UUIDs to guarantee uniqueness across concurrent uploads.
    """
    c = client or _get_client()
    collection = await _get_session_collection(c, session_id)
    ids = [str(uuid4()) for _ in chunks]
    # Strip session_id from metadata — it's implicit in the collection name.
    clean_metas = [{k: v for k, v in m.items() if k != "session_id"} for m in metadatas]
    await run_sync(
        collection.add,
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=clean_metas,
    )


async def delete_session(
    session_id: str,
    client: ClientAPI | None = None,
) -> int:
    """Delete the ChromaDB collection for *session_id*.

    Returns the number of documents that were in the collection, or 0 if it
    did not exist.
    """
    c = client or _get_client()
    name = _collection_name(session_id)
    try:
        collection = await _run_idempotent(c.get_collection, name)
        count = await _run_idempotent(collection.count)
        await run_sync(c.delete_collection, name)
        return count
    except Exception:
        return 0


async def list_session_ids(client: ClientAPI | None = None) -> list[str]:
    """Return all distinct session_ids that have a ChromaDB collection."""
    c = client or _get_client()
    collections = await _run_idempotent(c.list_collections)
    result: list[str] = []
    for col in collections:
        name = col.name if hasattr(col, "name") else str(col)
        sid = session_id_from_collection(name)
        if sid is not None:
            result.append(sid)
    return result


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
    c = client or _get_client()
    collection = await _get_session_collection(c, session_id)
    if drug_name is not None:
        where: dict = {"$and": [{"section": section}, {"drug_name": drug_name}]}
    else:
        where = {"section": section}
    results = await _run_idempotent(
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


async def retrieve(
    query_embedding: list[float],
    session_id: str,
    top_k: int = 5,
    drug_name: str | None = None,
    client: ClientAPI | None = None,
) -> list[dict]:
    """Query Chroma for the top-*k* chunks scoped to *session_id*.

    When *drug_name* is given, results are additionally filtered to that drug —
    used by the per-drug retrieval strategy to guarantee each drug gets
    representation in the context regardless of global distance ranking.

    Returns a list of dicts with keys: text, drug_name, section, distance.
    """
    c = client or _get_client()
    collection = await _get_session_collection(c, session_id)
    if await _run_idempotent(collection.count) == 0:
        return []
    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if drug_name is not None:
        query_kwargs["where"] = {"drug_name": drug_name}
    results = await _run_idempotent(collection.query, **query_kwargs)
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
