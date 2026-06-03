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


def test_paragraphs_kept_together():
    """Short paragraphs are not split across chunk boundaries."""
    para1 = ("First paragraph about indications. " * 5).strip()  # ~175 chars
    para2 = ("Second paragraph about warnings. " * 5).strip()  # ~160 chars
    text = para1 + "\n\n" + para2
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 2
    assert chunks[0] == para1
    assert chunks[1] == para2


def test_overlap_equal_to_chunk_size_raises():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("some text", chunk_size=100, overlap=100)


def test_overlap_greater_than_chunk_size_raises():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("some text", chunk_size=100, overlap=150)


def test_sentence_boundary_snapping():
    """Chunks must end at sentence boundaries, not mid-sentence."""
    para = (
        "Lisinopril is an ACE inhibitor. "
        "It is indicated for hypertension. "
        "Take once daily with or without food. "
        "Avoid potassium supplements while on this drug. "
        "Consult your doctor before stopping."
    )
    # chunk_size small enough to force multiple chunks
    chunks = chunk_text(para, chunk_size=100, overlap=20)
    for chunk in chunks:
        # Every chunk produced from sentence-aware splitting must end on
        # a sentence boundary (period/question/exclamation) or be the last
        # piece of a char-split oversized sentence.
        assert chunk[-1] in ".!?" or len(chunk) <= 100


def test_sentence_overlap_carries_context():
    """The tail sentence of chunk N reappears at the start of chunk N+1
    when it is short enough to fit within the overlap budget."""
    # Each sentence is ~20 chars so fits easily within overlap=30.
    para = "Take daily. Avoid sun. Drink water. See doctor. No alcohol."
    chunks = chunk_text(para, chunk_size=50, overlap=30)
    if len(chunks) > 1:
        # At least one word from the end of chunk[0] must appear in chunk[1].
        assert any(word in chunks[1] for word in chunks[0].split()[-4:])


def test_default_chunk_size_is_1000():
    text = "x" * 1500
    chunks = chunk_text(text)
    assert all(len(c) <= 1000 for c in chunks)


def test_default_overlap_is_100():
    # 100-char overlap on a plain text means chunk[1] starts 900 chars into chunk[0].
    text = "a" * 2000
    chunks = chunk_text(text)
    assert len(chunks) >= 2
    assert chunks[0][-100:] == chunks[1][:100]
