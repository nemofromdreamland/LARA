import pytest

from app.services.chunker import chunk_text


def test_empty_text_returns_empty():
    assert chunk_text("") == []


def test_whitespace_only_returns_empty():
    assert chunk_text("   \n  ") == []


def test_short_text_is_single_chunk():
    text = "Short text."
    result = chunk_text(text, chunk_size=500, overlap=50)
    assert result == [text]


def test_chunk_size_respected():
    text = "x" * 1000
    chunks = chunk_text(text, chunk_size=500, overlap=0)
    assert all(len(c) <= 500 for c in chunks)


def test_overlap_creates_shared_content():
    text = "a" * 100
    chunks = chunk_text(text, chunk_size=60, overlap=20)
    # Each chunk except the first starts 40 chars after the previous one started
    # so 20 chars overlap between consecutive chunks
    assert len(chunks) >= 2
    # Tail of chunk[0] == head of chunk[1] (overlap region)
    assert chunks[0][-20:] == chunks[1][:20]


def test_no_overlap():
    text = "ab" * 300  # 600 chars
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    assert len(chunks) == 6
    assert "".join(chunks) == text


def test_full_text_is_covered():
    text = "hello " * 200  # 1200 chars
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    # First and last char of original text appear somewhere in chunks
    assert chunks[0].startswith(text[:1])
    assert text[-1] in chunks[-1]


def test_single_char_text():
    assert chunk_text("x") == ["x"]


@pytest.mark.parametrize("size,overlap", [(100, 10), (200, 50), (500, 50)])
def test_parametrized_sizes(size: int, overlap: int):
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=size, overlap=overlap)
    assert all(len(c) <= size for c in chunks)
    assert len(chunks) >= 1
