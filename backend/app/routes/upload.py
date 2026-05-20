import asyncio
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import UploadResponse
from app.services.chunker import chunk_text
from app.services.dailymed import fetch_leaflet_sections
from app.services.embedder import embed
from app.services.pdf_parser import PDFExtractionError, extract_text
from app.services.prescription_parser import parse_prescription
from app.services.session_store import (
    save_prescription,
    save_prescription_entries,
    save_upload_result,
)
from app.services.vector_store import store
from app.utils import get_request_id, run_sync

logger = logging.getLogger(__name__)

router = APIRouter()


async def _process_drug(
    drug: str, session_id: str
) -> tuple[str, list[str], list[dict]]:
    """Fetch leaflet sections for one drug and produce chunks + metadata.

    Returns (drug, chunks, metas). Both lists are empty when no leaflet is found.
    """
    sections = await fetch_leaflet_sections(drug)
    if not sections:
        return drug, [], []
    chunks: list[str] = []
    metas: list[dict] = []
    for section in sections:
        for chunk in chunk_text(section.text):
            chunks.append(chunk)
            metas.append(
                {
                    "session_id": session_id,
                    "drug_name": drug,
                    "section": section.section,
                }
            )
    return drug, chunks, metas


@router.post("/upload", response_model=UploadResponse)
async def upload(
    session_id: str = Form(...),
    file: UploadFile = File(...),
) -> UploadResponse:
    rid = get_request_id()

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    raw_bytes = await file.read()
    if len(raw_bytes) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail="File too large. Maximum size is 20 MB."
        )
    if not raw_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400, detail="File does not appear to be a valid PDF."
        )

    logger.info(
        "upload started for session %s", session_id, extra={"request_id": rid}
    )

    try:
        text = await run_sync(extract_text, raw_bytes)
    except PDFExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if not text:
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    # LLM-based structured extraction (falls back to regex if LLM is unavailable).
    entries = await parse_prescription(text)
    if not entries:
        raise HTTPException(
            status_code=422, detail="No drug names found in prescription."
        )

    save_prescription(session_id, text)
    save_prescription_entries(session_id, entries)

    # Derive clean drug names for DailyMed lookup.
    drug_names = [e.drug_name for e in entries]

    # Fetch all drug leaflets concurrently; a single drug failure won't abort the rest.
    results = await asyncio.gather(
        *[_process_drug(drug, session_id) for drug in drug_names],
        return_exceptions=True,
    )

    stored_drugs: list[str] = []
    missing_drugs: list[str] = []

    for drug, result in zip(drug_names, results):
        if isinstance(result, Exception):
            logger.warning(
                "DailyMed fetch failed for %s: %s", drug, result,
                extra={"request_id": rid},
            )
            missing_drugs.append(drug)
            continue
        _, chunks, metas = result
        if not chunks:
            logger.warning(
                "No DailyMed leaflet found for drug: %s", drug,
                extra={"request_id": rid},
            )
            missing_drugs.append(drug)
            continue
        embeddings = await embed(chunks)
        await store(chunks, embeddings, metas)
        stored_drugs.append(drug)

    save_upload_result(session_id, stored_drugs, missing_drugs)

    logger.info(
        "drugs found: %d stored, %d missing", len(stored_drugs), len(missing_drugs),
        extra={"request_id": rid},
    )

    return UploadResponse(
        session_id=session_id,
        drugs_found=stored_drugs,
        missing_leaflets=missing_drugs,
        status="ok" if stored_drugs else "no_leaflets_found",
    )
