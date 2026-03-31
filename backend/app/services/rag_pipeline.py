from app.models.schemas import ChatResponse, Source
from app.services.embedder import embed
from app.services.llm_client import generate
from app.services.vector_store import retrieve


async def answer(session_id: str, question: str) -> ChatResponse:
    """Run the full RAG query pipeline for *question* scoped to *session_id*.

    Steps:
    1. Embed the question.
    2. Retrieve top-5 chunks from Chroma filtered by session_id.
    3. Build context string from retrieved chunks.
    4. Generate answer via LLM (hallucination-guarded).
    5. Return ChatResponse with answer + deduplicated source list.
    """
    query_embedding = embed([question])[0]
    chunks = retrieve(query_embedding, session_id, top_k=5)

    if not chunks:
        return ChatResponse(
            answer="This information is not available in the provided leaflets.",
            sources=[],
        )

    context = "\n\n".join(
        f"[{c['drug_name']} — {c['section']}]\n{c['text']}" for c in chunks
    )
    answer_text = await generate(context, question)

    seen: set[tuple[str, str]] = set()
    sources: list[Source] = []
    for c in chunks:
        key = (c["drug_name"], c["section"])
        if key not in seen:
            seen.add(key)
            sources.append(Source(drug_name=c["drug_name"], section=c["section"]))

    return ChatResponse(answer=answer_text, sources=sources)
