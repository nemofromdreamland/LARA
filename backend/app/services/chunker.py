def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split *text* into overlapping chunks using a sliding window.

    Args:
        text: Input text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters shared between consecutive chunks.

    Returns:
        List of non-empty text chunks. Returns [] for empty input.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size]
        chunks.append(chunk)
        start += step
    return chunks
