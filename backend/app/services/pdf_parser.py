import fitz  # PyMuPDF


class PDFExtractionError(ValueError):
    """Raised when PyMuPDF cannot parse or decode a PDF."""


def extract_text(file_bytes: bytes) -> str:
    """Extract plain text from a PDF given its raw bytes."""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            pages = [page.get_text() for page in doc]
    except Exception as exc:
        raise PDFExtractionError(f"Could not read PDF: {exc}") from exc
    return "\n".join(pages).strip()
