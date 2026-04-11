import fitz
import pytest

from app.services.pdf_parser import PDFExtractionError, extract_text


def _make_pdf(text: str) -> bytes:
    """Create a minimal single-page PDF containing *text*."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    return doc.tobytes()


def test_extract_text_returns_string():
    pdf = _make_pdf("Lisinopril 10mg daily")
    result = extract_text(pdf)
    assert isinstance(result, str)


def test_extract_text_contains_content():
    pdf = _make_pdf("Metformin 500mg twice daily")
    result = extract_text(pdf)
    assert "Metformin" in result


def test_extract_text_multipage():
    doc = fitz.open()
    for word in ("Atorvastatin", "Amlodipine"):
        page = doc.new_page()
        page.insert_text((72, 72), word, fontsize=12)
    pdf_bytes = doc.tobytes()

    result = extract_text(pdf_bytes)
    assert "Atorvastatin" in result
    assert "Amlodipine" in result


def test_extract_text_empty_pdf():
    doc = fitz.open()
    doc.new_page()
    result = extract_text(doc.tobytes())
    assert result == ""


def test_extract_text_invalid_bytes_raises_extraction_error():
    with pytest.raises(PDFExtractionError, match="Could not read PDF"):
        extract_text(b"not a pdf")
