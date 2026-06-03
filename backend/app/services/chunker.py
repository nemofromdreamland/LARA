import re

# Splits on whitespace following sentence-ending punctuation.
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_RE.split(text) if s.strip()]


def _chunk_large_para(para: str, chunk_size: int, overlap: int, out: list[str]) -> None:
    """Pack sentences into chunks, snapping boundaries to sentence ends.

    Falls back to a raw character sliding window when the paragraph has no
    detectable sentence boundaries (e.g. a long XML table or numeric block).
    """
    sentences = _split_sentences(para)

    # No sentence boundaries detected — raw char sliding window.
    if len(sentences) <= 1:
        step = chunk_size - overlap
        start = 0
        while start < len(para):
            out.append(para[start : start + chunk_size])
            start += step
        return

    step = chunk_size - overlap
    buf: list[str] = []
    buf_len = 0

    for sent in sentences:
        sent_len = len(sent)
        # +1 accounts for the space separator between sentences in the buffer.
        needed = sent_len + (1 if buf else 0)

        if buf_len + needed > chunk_size:
            if buf:
                out.append(" ".join(buf))
                # Carry the tail sentences that fit within the overlap budget.
                carry: list[str] = []
                carry_len = 0
                for s in reversed(buf):
                    slen = len(s) + (1 if carry else 0)
                    if carry_len + slen <= overlap:
                        carry.insert(0, s)
                        carry_len += slen
                    else:
                        break
                # Safety: if carry + new sentence already overflows, drop carry.
                if carry and carry_len + sent_len + 1 > chunk_size:
                    carry = []
                    carry_len = 0
                buf = carry
                buf_len = carry_len

            if sent_len > chunk_size:
                # Single sentence is itself too long: char split.
                start = 0
                while start < len(sent):
                    out.append(sent[start : start + chunk_size])
                    start += step
                continue

        buf.append(sent)
        buf_len += sent_len + (1 if len(buf) > 1 else 0)

    if buf:
        out.append(" ".join(buf))


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
) -> list[str]:
    """Split *text* into chunks, respecting paragraph and sentence boundaries.

    Strategy:
    1. Split on blank lines into paragraphs.
    2. Paragraphs <= chunk_size are emitted as single chunks.
    3. Larger paragraphs are split sentence-by-sentence, with a character-level
       overlap carried from the tail of each chunk into the next.  When no
       sentence boundaries exist the strategy falls back to a raw character
       sliding window.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum characters per chunk (default 1000).
        overlap: Characters shared between consecutive chunks (default 100).

    Returns:
        List of non-empty text chunks. Returns [] for empty input.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []

    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            _chunk_large_para(para, chunk_size, overlap, chunks)

    return [c for c in chunks if c]
