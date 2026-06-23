import asyncio
import json
import logging
import math
import re
from collections.abc import AsyncGenerator

import tiktoken
from prometheus_client import Counter

from app.config import settings
from app.models.schemas import ChatResponse, PrescriptionEntry, Source
from app.security import sanitize_chat_input
from app.services.embedder import embed
from app.services.input_classifier import Route, classify_input
from app.services.llm_client import (
    STREAM_RESET,
    SYSTEM_PROMPT,
    generate,
    generate_stream,
    strip_cited_line,
)
from app.services.reranker import rerank
from app.services.session_store import get_prescription_entries, get_upload_result
from app.services.vector_store import retrieve
from app.utils import get_request_id, whole_word_match

logger = logging.getLogger(__name__)

# cl100k_base is the LLaMA-family proxy encoding; avoids a sentencepiece dependency
_enc = tiktoken.get_encoding("cl100k_base")

# Fixed system-prompt cost, counted against the context budget once per request.
_SYSTEM_PROMPT_TOKENS = len(_enc.encode(SYSTEM_PROMPT))

_RETRIEVAL_CHUNKS = Counter(
    "lara_retrieval_chunks_total",
    "Chunks passing the distance threshold per query",
)

# Per-route classification counter (incl. the safety routes). Labelled by route
# only — never the message content — so no PII is recorded.
_CHAT_ROUTE = Counter(
    "lara_chat_route_total",
    "Chat messages by pre-retrieval classification route",
    ["route"],
)


def trim_to_budget(chunks_with_scores: list, max_tokens: int) -> list:
    """Return the highest-relevance chunks that fit within *max_tokens*.

    Sorts ascending by distance (most relevant first), then uses a full-fill
    strategy: skips a chunk that would overflow but continues checking remaining
    smaller chunks.  The prescription summary must be counted separately by the
    caller before invoking this function.
    """
    rid = get_request_id()
    if chunks_with_scores and "rerank_score" in chunks_with_scores[0]:
        sorted_chunks = sorted(
            chunks_with_scores, key=lambda c: c["rerank_score"], reverse=True
        )
    else:
        sorted_chunks = sorted(chunks_with_scores, key=lambda c: c["distance"])
    kept: list = []
    total = 0
    for chunk in sorted_chunks:
        # Count the "[drug — section]\n" header the context assembler prepends to
        # each chunk, not just the chunk text, so the budget reflects the real prompt.
        header = f"[{chunk['drug_name']} — {chunk['section']}]\n"
        chunk_len = len(_enc.encode(header + chunk["text"]))
        if total + chunk_len <= max_tokens:
            kept.append(chunk)
            total += chunk_len

    trimmed = len(sorted_chunks) - len(kept)
    if trimmed:
        logger.debug(
            "trim_to_budget: dropped %d chunk(s), %d tokens kept",
            trimmed,
            total,
            extra={"request_id": rid},
        )
    else:
        logger.debug(
            "trim_to_budget: all %d chunk(s) fit, %d tokens kept",
            len(kept),
            total,
            extra={"request_id": rid},
        )
    return kept


def _history_token_cost(history: list[dict] | None) -> int:
    """Token cost of the conversation history sent alongside the context."""
    if not history:
        return 0
    text = "\n".join(h.get("content", "") for h in history)
    return len(_enc.encode(text))


def _trim_history_to_budget(history: list[dict], max_tokens: int) -> list[dict]:
    """Keep the most recent whole turns whose cumulative token cost fits max_tokens.

    Drops oldest turns first; never splits a turn. Counting matches
    _history_token_cost so the budget reservation in _prepare_context becomes
    exact (a single most-recent turn over budget yields an empty list).
    """
    if not history:
        return []
    kept: list[dict] = []
    for turn in reversed(history):
        candidate = [turn, *kept]
        if _history_token_cost(candidate) > max_tokens:
            break
        kept = candidate
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


async def _build_fallback_message(session_id: str) -> str:
    """Return the 'no relevant chunks' message enriched with session context."""
    drugs_found, missing = await get_upload_result(session_id)
    indexed = ", ".join(drugs_found) if drugs_found else "none"
    no_leaflet = ", ".join(missing) if missing else "none"
    return (
        "I couldn't find relevant information in the uploaded leaflets "
        "for your question. "
        f"Drugs indexed: {indexed}. "
        f"Drugs with no leaflet found: {no_leaflet}. "
        "Try rephrasing your question or ask about a specific section "
        "(e.g. 'warnings', 'dosage', 'interactions')."
    )


def _deduplicate_sources(chunks: list[dict]) -> list[Source]:
    """Return one Source per (drug_name, section) pair, preserving relevance order."""
    seen: set[tuple[str, str]] = set()
    sources: list[Source] = []
    for c in chunks:
        key = (c["drug_name"], c["section"])
        if key not in seen:
            seen.add(key)
            sources.append(
                Source(
                    drug_name=c["drug_name"],
                    section=c["section"],
                    rerank_score=c.get("rerank_score"),
                )
            )
    return sources


def _filter_sources_by_cited(
    sources: list[Source], cited: list[tuple[str, str]] | None
) -> list[Source]:
    """Keep only sources the LLM explicitly cited.

    - cited is None → no/garbled CITED footer; fall back to all sources.
    - cited == []   → the model wrote 'CITED: none'; return zero sources.
    - otherwise     → keep the cited (drug, section) pairs, falling back to all
      only when none of them match a retrieved source (LLM format drift).
    """
    if cited is None:
        return sources
    if not cited:
        return []
    cited_set = {(drug, section) for drug, section in cited}
    filtered = [
        s for s in sources if (s.drug_name.lower(), s.section.lower()) in cited_set
    ]
    # If citation parsing produced no matches (LLM format drift), return everything.
    return filtered if filtered else sources


# Wording that asks about the whole prescription rather than a named drug —
# forces retrieval across every session drug regardless of which are mentioned.
# Deliberately narrow: a bare "any"/"all" (as in "any problems taking X?") must
# NOT trigger this — only quantifiers that clearly refer to the medications.
_BROAD_RE = re.compile(
    r"\bboth\b"
    r"|\b(?:any|all|each|one|none|some|either|neither) of (?:these|those|them|the|my)\b"
    r"|\b(?:my|these|those|the) "
    r"(?:drugs?|meds?|medications?|medicines?|prescriptions?|pills?|leaflets?)\b",
    re.IGNORECASE,
)
# Wording that asks how drugs relate to one another — needs the other drugs'
# context, so it falls back to all session drugs unless ≥2 are named explicitly.
_INTERACTION_RE = re.compile(
    r"\b(interact\w*|combin\w*|together|mix|compare\w*|versus|vs)\b",
    re.IGNORECASE,
)


def _resolve_drug_scope(
    question: str, history: list[dict] | None, drugs_found: list[str]
) -> list[str]:
    """Which session drugs the question is about. Defaults to all.

    - Broad wording ("any of these") → every drug.
    - Interaction/comparison wording → the named drugs if ≥2, else every drug
      (the other leaflets are needed to assess the relationship).
    - One or more drugs named → just those.
    - Nothing named → inherit the drug(s) named in the previous user turn
      (follow-up like "what about pregnancy?"), else every drug.
    """
    mentioned = [d for d in drugs_found if whole_word_match(d, question)]
    if _BROAD_RE.search(question):
        return drugs_found
    if _INTERACTION_RE.search(question):
        return mentioned if len(mentioned) >= 2 else drugs_found
    if mentioned:
        return mentioned
    # Follow-up: inherit the drug(s) named in the previous user turn.
    last_user = next(
        (h["content"] for h in reversed(history or []) if h.get("role") == "user"),
        None,
    )
    if last_user:
        prior = [d for d in drugs_found if whole_word_match(d, last_user)]
        if prior:
            return prior
    return drugs_found


async def _retrieve_diverse(
    query_embedding: list[float],
    session_id: str,
    drugs: list[str],
) -> list[dict]:
    """Run one retrieval query per drug in parallel, merge, deduplicate,
    sort by distance.

    Guarantees at least one chunk per drug when multiple drugs are in the session,
    preventing a single drug from monopolising all top-k slots.
    """
    per_drug_k = max(4, math.ceil(settings.retrieval_top_k / len(drugs)))
    per_drug_results = await asyncio.gather(
        *[
            retrieve(query_embedding, session_id, top_k=per_drug_k, drug_name=drug)
            for drug in drugs
        ]
    )
    seen_texts: set[str] = set()
    merged: list[dict] = []
    for drug_chunks in per_drug_results:
        for chunk in drug_chunks:
            if chunk["text"] not in seen_texts:
                seen_texts.add(chunk["text"])
                merged.append(chunk)
    merged.sort(key=lambda c: c["distance"])
    return merged


async def _prepare_context(
    session_id: str,
    question: str,
    history: list[dict] | None = None,
    embed_executor=None,
) -> tuple[list[dict], str] | None:
    """Embed the retrieval query, retrieve chunks, assemble context string.

    When *history* is provided, the last user turn is appended to *question*
    to give the embedder richer context for follow-up questions.

    Returns (chunks, context) or None if no chunks pass the distance threshold.
    The caller is responsible for the no-chunks fallback path.
    """
    rid = get_request_id()
    # Enrich the embedding query with the most recent prior exchange so that
    # follow-up questions ("What about pregnancy?") embed in the right direction.
    last_user_turn = next(
        (h["content"] for h in reversed(history or []) if h.get("role") == "user"),
        None,
    )
    retrieval_query = f"{last_user_turn} {question}" if last_user_turn else question
    query_embedding = (await embed([retrieval_query], embed_executor, source="query"))[
        0
    ]

    drugs_found, _ = await get_upload_result(session_id)
    if len(drugs_found) > 1:
        # Scope retrieval to the drug(s) the question is actually about so a
        # single-drug question doesn't pull in every other drug's leaflet.
        scope = _resolve_drug_scope(question, history, drugs_found)
        if len(scope) == 1:
            raw_chunks = await retrieve(
                query_embedding,
                session_id,
                top_k=settings.retrieval_top_k,
                drug_name=scope[0],
            )
        else:
            # Per-drug retrieval: guarantees representation from every scoped drug.
            raw_chunks = await _retrieve_diverse(query_embedding, session_id, scope)
    else:
        raw_chunks = await retrieve(
            query_embedding, session_id, top_k=settings.retrieval_top_k
        )
    chunks = [
        c for c in raw_chunks if c["distance"] < settings.retrieval_distance_threshold
    ]
    _RETRIEVAL_CHUNKS.inc(len(chunks))

    logger.debug(
        "RAG retrieve: %d raw chunks, %d passed threshold (threshold=%.2f)",
        len(raw_chunks),
        len(chunks),
        settings.retrieval_distance_threshold,
        extra={"request_id": rid},
    )

    if not chunks:
        return None

    if settings.reranker_enabled:
        # Embedding and reranking are sequential within a request, so the
        # embed pool doubles as the bounded model-inference pool.
        chunks = await rerank(retrieval_query, chunks, executor=embed_executor)

    context_parts: list[str] = []
    entries = await get_prescription_entries(session_id)
    prescription_text = _format_prescription(entries) if entries else ""
    if prescription_text:
        context_parts.append(prescription_text)

    # Reserve budget for the system prompt and (capped) history so retrieved
    # chunks plus those fixed costs stay within the model's context window.
    history_tokens = min(_history_token_cost(history), settings.max_history_tokens)
    remaining_budget = (
        settings.max_context_tokens
        - len(_enc.encode(prescription_text))
        - _SYSTEM_PROMPT_TOKENS
        - history_tokens
    )
    chunks = trim_to_budget(chunks, max_tokens=max(0, remaining_budget))
    if not chunks:
        return None

    context_parts.extend(
        f"[{c['drug_name']} — {c['section']}]\n{c['text']}" for c in chunks
    )
    context = "\n\n".join(context_parts)
    return chunks, context


async def answer(
    session_id: str,
    question: str,
    history: list[dict] | None = None,
    embed_executor=None,
) -> ChatResponse:
    """Run the full RAG query pipeline for *question* scoped to *session_id*.

    Steps:
    0. Classify the message; non-medical/unsafe routes short-circuit (no
       retrieval, no source chips) with a canned reply.
    1. Embed the question (enriched with last history turn for follow-ups).
    2. Retrieve top-k chunks from Chroma filtered by session_id.
    3. Build context string from prescription + retrieved leaflet chunks.
    4. Generate answer via LLM (hallucination-guarded, with conversation history).
    5. Return ChatResponse with answer + citation-filtered source list.
    """
    drugs_found, _ = await get_upload_result(session_id)
    gate = classify_input(question, drugs_found)
    _CHAT_ROUTE.labels(route=gate.route.value).inc()
    if gate.route is not Route.MEDICAL:
        return ChatResponse(answer=gate.reply, sources=[])

    # Trim history to the reserved budget once, then use the trimmed list
    # everywhere so _prepare_context's reservation is exact. Sanitize the
    # question AFTER classification (the classifier saw the raw text) but
    # BEFORE embed/LLM — defense-in-depth against prompt injection.
    history = _trim_history_to_budget(history or [], settings.max_history_tokens)
    question = sanitize_chat_input(question)

    prepared = await _prepare_context(session_id, question, history, embed_executor)
    if prepared is None:
        fallback = await _build_fallback_message(session_id)
        return ChatResponse(answer=fallback, sources=[])

    chunks, context = prepared
    raw_answer = await generate(context, question, history=history)
    answer_text, cited = strip_cited_line(raw_answer)
    all_sources = _deduplicate_sources(chunks)
    return ChatResponse(
        answer=answer_text,
        sources=_filter_sources_by_cited(all_sources, cited),
    )


async def answer_stream(
    session_id: str,
    question: str,
    history: list[dict] | None = None,
    embed_executor=None,
) -> AsyncGenerator[str, None]:
    """Stream the RAG answer as SSE-ready payloads.

    Yields:
      - Raw text tokens while the LLM is generating.
      - llm_client.STREAM_RESET (passed through) if a mid-stream provider
        failover invalidates the tokens yielded so far.
      - A single ``[SOURCES]{json}`` line once generation is complete.
      - A final ``[DONE]`` line.

    The frontend is responsible for stripping the trailing CITED: line from
    the accumulated text when it receives the sources event.
    """
    drugs_found, _ = await get_upload_result(session_id)
    gate = classify_input(question, drugs_found)
    _CHAT_ROUTE.labels(route=gate.route.value).inc()
    if gate.route is not Route.MEDICAL:
        # Same contract as the no-chunks path: text → empty [SOURCES] → [DONE],
        # so SSE framing and server-side history append are unchanged.
        yield gate.reply
        yield "[SOURCES]" + json.dumps({"sources": []})
        yield "[DONE]"
        return

    # Trim history to the reserved budget once, then use the trimmed list
    # everywhere so _prepare_context's reservation is exact. Sanitize the
    # question AFTER classification (the classifier saw the raw text) but
    # BEFORE embed/LLM — defense-in-depth against prompt injection.
    history = _trim_history_to_budget(history or [], settings.max_history_tokens)
    question = sanitize_chat_input(question)

    prepared = await _prepare_context(session_id, question, history, embed_executor)
    if prepared is None:
        fallback = await _build_fallback_message(session_id)
        yield fallback
        yield "[SOURCES]" + json.dumps({"sources": []})
        yield "[DONE]"
        return

    chunks, context = prepared

    # Accumulate tokens so the streamed answer's citations drive source filtering,
    # exactly as answer() does — keeping /chat and /chat/stream sources in sync.
    accumulated: list[str] = []
    async for token in generate_stream(context, question, history=history):
        if token == STREAM_RESET:
            # Mid-stream failover regenerates from scratch; discard partial tokens
            # so only the final answer's CITED line is parsed.
            accumulated.clear()
        else:
            accumulated.append(token)
        yield token

    _, cited = strip_cited_line("".join(accumulated))
    filtered = _filter_sources_by_cited(_deduplicate_sources(chunks), cited)
    sources = [{"drug_name": s.drug_name, "section": s.section} for s in filtered]
    yield "[SOURCES]" + json.dumps({"sources": sources})
    yield "[DONE]"
