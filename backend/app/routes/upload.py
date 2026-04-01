import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import UploadResponse
from app.services.chunker import chunk_text
from app.services.dailymed import fetch_leaflet_sections
from app.services.drug_extractor import extract_drug_names
from app.services.embedder import embed
from app.services.pdf_parser import extract_text
from app.services.session_store import save_prescription, save_upload_result
from app.services.vector_store import store

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload(
    session_id: str = Form(...),
    file: UploadFile = File(...),
) -> UploadResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    raw_bytes = await file.read()
    text = extract_text(raw_bytes)
    if not text:
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    save_prescription(session_id, text)

    drug_names = extract_drug_names(text)
    if not drug_names:
        raise HTTPException(
            status_code=422, detail="No drug names found in prescription."
        )

    stored_drugs: list[str] = []
    missing_drugs: list[str] = []

    for drug in drug_names:
        sections = await fetch_leaflet_sections(drug)
        if not sections:
            logger.warning("No DailyMed leaflet found for drug: %s", drug)
            missing_drugs.append(drug)
            continue

        all_chunks: list[str] = []
        all_metas: list[dict] = []

        for section in sections:
            for chunk in chunk_text(section.text):
                all_chunks.append(chunk)
                all_metas.append(
                    {
                        "session_id": session_id,
                        "drug_name": drug,
                        "section": section.section,
                    }
                )

        if all_chunks:
            embeddings = embed(all_chunks)
            store(all_chunks, embeddings, all_metas)
            stored_drugs.append(drug)

    save_upload_result(session_id, stored_drugs, missing_drugs)

    return UploadResponse(
        session_id=session_id,
        drugs_found=stored_drugs,
        missing_leaflets=missing_drugs,
        status="ok" if stored_drugs else "no_leaflets_found",
    )
