import fitz  # PyMuPDF


def extract_text(file_bytes: bytes) -> str:
    """Extract plain text from a PDF given its raw bytes."""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        pages = [page.get_text() for page in doc]
    return "\n".join(pages).strip()
