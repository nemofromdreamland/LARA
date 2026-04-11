def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split *text* into chunks, respecting paragraph boundaries where possible.

    Strategy:
    1. Split on blank lines (double newline) into paragraphs.
    2. Paragraphs <= chunk_size are emitted as single chunks.
    3. Paragraphs > chunk_size are split with a character-level sliding window.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters shared between consecutive chunks.

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
    step = chunk_size - overlap

    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            start = 0
            while start < len(para):
                chunks.append(para[start : start + chunk_size])
                start += step

    return chunks
