import json
from collections.abc import AsyncGenerator

from app.models.schemas import ChatResponse, Source
from app.services.embedder import embed
from app.services.llm_client import generate, generate_stream
from app.services.session_store import get_prescription, get_upload_result
from app.services.vector_store import retrieve


async def answer(session_id: str, question: str) -> ChatResponse:
    """Run the full RAG query pipeline for *question* scoped to *session_id*.

    Steps:
    1. Embed the question.
    2. Retrieve top-5 chunks from Chroma filtered by session_id.
    3. Build context string from prescription + retrieved leaflet chunks.
    4. Generate answer via LLM (hallucination-guarded).
    5. Return ChatResponse with answer + deduplicated source list.
    """
    query_embedding = embed([question])[0]
    chunks = retrieve(query_embedding, session_id, top_k=5)

    if not chunks:
        drugs_found, missing = get_upload_result(session_id)
        parts: list[str] = [
            "This information is not available in the provided leaflets."
        ]
        if drugs_found:
            parts.append(f"Indexed leaflets: {', '.join(drugs_found)}.")
        if missing:
            parts.append(f"No official leaflet was found for: {', '.join(missing)}.")
        return ChatResponse(answer=" ".join(parts), sources=[])

    context_parts: list[str] = []
    prescription = get_prescription(session_id)
    if prescription:
        # Include only the first 600 characters — enough to capture patient name,
        # drug, dosage, and frequency without overwhelming the context with a full
        # formatted document (some PDFs embed the entire patient info sheet).
        snippet = prescription[:600].strip()
        context_parts.append(f"[Prescription excerpt]\n{snippet}")
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
    query_embedding = embed([question])[0]
    chunks = retrieve(query_embedding, session_id, top_k=5)

    if not chunks:
        drugs_found, missing = get_upload_result(session_id)
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
    prescription = get_prescription(session_id)
    if prescription:
        snippet = prescription[:600].strip()
        context_parts.append(f"[Prescription excerpt]\n{snippet}")
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
