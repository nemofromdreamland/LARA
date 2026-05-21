import json
import logging
from collections.abc import AsyncGenerator

from app.config import settings
from app.models.schemas import ChatResponse, PrescriptionEntry, Source
from app.services.embedder import embed
from app.services.llm_client import generate, generate_stream
from app.services.session_store import get_prescription_entries, get_upload_result
from app.services.vector_store import retrieve
from app.utils import get_request_id

logger = logging.getLogger(__name__)

# Cosine distance threshold for retrieval quality filtering.
# Chroma's HNSW uses cosine distance (range 0–2); 0 = identical, 2 = opposite.
# Chunks above this threshold are semantically too distant to be useful context.
_DISTANCE_THRESHOLD = 0.45


def trim_to_budget(chunks_with_scores: list, max_chars: int) -> list:
    """Return the highest-relevance chunks that fit within *max_chars*.

    Sorts ascending by distance (most relevant first), then greedily accumulates
    chunks until the next one would overflow the budget.  The prescription summary
    must be counted separately by the caller before invoking this function.
    """
    rid = get_request_id()
    sorted_chunks = sorted(chunks_with_scores, key=lambda c: c["distance"])
    kept: list = []
    total = 0
    for chunk in sorted_chunks:
        chunk_len = len(chunk["text"])
        if total + chunk_len > max_chars:
            break
        kept.append(chunk)
        total += chunk_len

    trimmed = len(sorted_chunks) - len(kept)
    if trimmed:
        logger.debug(
            "trim_to_budget: dropped %d chunk(s), %d chars kept",
            trimmed,
            total,
            extra={"request_id": rid},
        )
    else:
        logger.debug(
            "trim_to_budget: all %d chunk(s) fit, %d chars kept",
            len(kept),
            total,
            extra={"request_id": rid},
        )
    return kept


def _format_prescription(entries: list[PrescriptionEntry]) -> str:
    """Format structured prescription entries as a numbered bullet-point block."""
    lines = ["[Prescription]"]
    for i, e in enumerate(entries, 1):
        lines.append(f"{i}. {e.drug_name.title()}")
        if e.dosage:
            lines.append(f"   • Dosage: {e.dosage}")
        if e.frequency:
            lines.append(f"   • Frequency: {e.frequency}")
        if e.duration:
            lines.append(f"   • Duration: {e.duration}")
        if e.instructions:
            lines.append(f"   • Instructions: {e.instructions}")
    return "\n".join(lines)


async def answer(session_id: str, question: str) -> ChatResponse:
    """Run the full RAG query pipeline for *question* scoped to *session_id*.

    Steps:
    1. Embed the question.
    2. Retrieve top-5 chunks from Chroma filtered by session_id.
    3. Build context string from prescription + retrieved leaflet chunks.
    4. Generate answer via LLM (hallucination-guarded).
    5. Return ChatResponse with answer + deduplicated source list.
    """
    rid = get_request_id()
    query_embedding = (await embed([question]))[0]
    raw_chunks = await retrieve(query_embedding, session_id, top_k=5)
    chunks = [c for c in raw_chunks if c["distance"] < _DISTANCE_THRESHOLD]

    logger.debug(
        "RAG retrieve: %d raw chunks, %d passed threshold",
        len(raw_chunks),
        len(chunks),
        extra={"request_id": rid},
    )

    if not chunks:
        drugs_found, missing = await get_upload_result(session_id)
        parts: list[str] = [
            "This information is not available in the provided leaflets."
        ]
        if drugs_found:
            parts.append(f"Indexed leaflets: {', '.join(drugs_found)}.")
        if missing:
            parts.append(f"No official leaflet was found for: {', '.join(missing)}.")
        return ChatResponse(answer=" ".join(parts), sources=[])

    context_parts: list[str] = []
    entries = await get_prescription_entries(session_id)
    prescription_text = _format_prescription(entries) if entries else ""
    if prescription_text:
        context_parts.append(prescription_text)

    remaining_budget = settings.max_context_chars - len(prescription_text)
    chunks = trim_to_budget(chunks, max_chars=max(0, remaining_budget))

    context_parts.extend(
        f"[{c['drug_name']} — {c['section']}]\n{c['text']}" for c in chunks
    )
    context = "\n\n".join(context_parts)
    answer_text = await generate(context, question)

    seen: set[tuple[str, str]] = set()
    sources: list[Source] = []
    for c in chunks:
        key = (c["drug_name"], c["section"])
        if key not in seen:
            seen.add(key)
            sources.append(Source(drug_name=c["drug_name"], section=c["section"]))

    return ChatResponse(answer=answer_text, sources=sources)


async def answer_stream(session_id: str, question: str) -> AsyncGenerator[str, None]:
    """Stream the RAG answer as SSE-ready payloads.

    Yields:
      - Raw text tokens while the LLM is generating.
      - A single ``[SOURCES]{json}`` line once generation is complete.
      - A final ``[DONE]`` line.
    """
    rid = get_request_id()
    query_embedding = (await embed([question]))[0]
    raw_chunks = await retrieve(query_embedding, session_id, top_k=5)
    chunks = [c for c in raw_chunks if c["distance"] < _DISTANCE_THRESHOLD]

    logger.debug(
        "RAG stream retrieve: %d raw chunks, %d passed threshold",
        len(raw_chunks),
        len(chunks),
        extra={"request_id": rid},
    )

    if not chunks:
        drugs_found, missing = await get_upload_result(session_id)
        parts: list[str] = [
            "This information is not available in the provided leaflets."
        ]
        if drugs_found:
            parts.append(f"Indexed leaflets: {', '.join(drugs_found)}.")
        if missing:
            parts.append(f"No official leaflet was found for: {', '.join(missing)}.")
        yield " ".join(parts)
        yield "[SOURCES]" + json.dumps({"sources": []})
        yield "[DONE]"
        return

    context_parts: list[str] = []
    entries = await get_prescription_entries(session_id)
    prescription_text = _format_prescription(entries) if entries else ""
    if prescription_text:
        context_parts.append(prescription_text)

    remaining_budget = settings.max_context_chars - len(prescription_text)
    chunks = trim_to_budget(chunks, max_chars=max(0, remaining_budget))

    context_parts.extend(
        f"[{c['drug_name']} — {c['section']}]\n{c['text']}" for c in chunks
    )
    context = "\n\n".join(context_parts)

    async for token in generate_stream(context, question):
        yield token

    seen: set[tuple[str, str]] = set()
    sources: list[dict] = []
    for c in chunks:
        key = (c["drug_name"], c["section"])
        if key not in seen:
            seen.add(key)
            sources.append({"drug_name": c["drug_name"], "section": c["section"]})

    yield "[SOURCES]" + json.dumps({"sources": sources})
    yield "[DONE]"
