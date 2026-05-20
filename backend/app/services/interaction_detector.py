import re

from app.models.schemas import InteractionFlag
from app.services.session_store import get_upload_result
from app.services.vector_store import get_by_section

# Excerpt window: characters captured before and after the match point.
_EXCERPT_BEFORE = 100
_EXCERPT_AFTER = 200


def _extract_excerpt(text: str, term: str) -> str:
    """Return a ≤300-char excerpt from *text* around the first occurrence of *term*.

    Tries to snap to sentence boundaries; falls back to a plain character
    window if none are found.
    """
    idx = text.lower().find(term.lower())
    if idx == -1:
        return text[:300]

    start = max(0, idx - _EXCERPT_BEFORE)
    end = min(len(text), idx + len(term) + _EXCERPT_AFTER)

    # Snap start forward to the next sentence boundary if possible.
    sentence_start = text.rfind(".", 0, idx)
    if sentence_start != -1 and sentence_start >= start:
        start = sentence_start + 1

    # Snap end back to the nearest sentence boundary if possible.
    sentence_end = text.find(".", idx + len(term))
    if sentence_end != -1 and sentence_end <= end:
        end = sentence_end + 1

    return text[start:end].strip()


def _word_in_text(word: str, text: str) -> bool:
    """Return True if *word* appears as a whole word in *text* (case-insensitive)."""
    pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
    return bool(pattern.search(text))


async def detect_interactions(
    session_id: str,
    client=None,
) -> list[InteractionFlag]:
    """Scan drug_interactions leaflet sections for cross-drug mentions.

    For each pair (drug_a, drug_b) where drug_a != drug_b:
    - Fetch all ``drug_interactions`` chunks stored for drug_a.
    - Check whether drug_b's name appears as a whole word in any chunk.
    - If found, emit an InteractionFlag with a supporting excerpt.

    One flag per (drug_a, drug_b) pair — the excerpt from the highest-quality
    chunk (longest containing the term) is kept.
    """
    drugs_found, _ = await get_upload_result(session_id)

    if len(drugs_found) < 2:
        return []

    # pair key → best excerpt length seen so far (for deduplication)
    seen: dict[tuple[str, str], int] = {}
    flags: list[InteractionFlag] = []

    for drug_a in drugs_found:
        chunks = await get_by_section(
            session_id, "drug_interactions", drug_name=drug_a, client=client
        )
        for chunk in chunks:
            text = chunk["text"]
            for drug_b in drugs_found:
                if drug_b == drug_a:
                    continue
                if not _word_in_text(drug_b, text):
                    continue

                excerpt = _extract_excerpt(text, drug_b)
                pair = (drug_a, drug_b)

                if pair not in seen or len(excerpt) > seen[pair]:
                    seen[pair] = len(excerpt)
                    # Remove any existing flag for this pair and replace.
                    flags = [f for f in flags if (f.drug_a, f.drug_b) != pair]
                    flags.append(
                        InteractionFlag(
                            drug_a=drug_a,
                            drug_b=drug_b,
                            excerpt=excerpt,
                        )
                    )

    return flags
