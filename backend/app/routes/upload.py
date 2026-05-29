import asyncio
import logging
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)

from app.config import settings
from app.limiter import limiter
from app.models.schemas import JobStatusResponse, UploadJobResponse
from app.services.embedder import embed
from app.services.ingestion import process_drug
from app.services.pdf_parser import PDFExtractionError, extract_text
from app.services.prescription_parser import parse_prescription
from app.services.session_store import (
    get_job_status,
    save_job_status,
    save_prescription,
    save_prescription_entries,
    save_upload_result,
)
from app.services.vector_store import store
from app.utils import get_request_id, run_sync

logger = logging.getLogger(__name__)

router = APIRouter()


async def _run_ingestion(
    job_id: str,
    session_id: str,
    text: str,
    rid: str,
    embed_executor,
) -> None:
    """Background task: parse prescription → fetch leaflets → embed → store."""
    try:
        entries = await parse_prescription(text)
        if not entries:
            await save_job_status(
                job_id,
                session_id,
                "failed",
                error="No drug names found in prescription.",
            )
            return

        await save_prescription(session_id, text)
        await save_prescription_entries(session_id, entries)

        drug_names = [e.drug_name for e in entries]

        results = await asyncio.gather(
            *[process_drug(drug, session_id) for drug in drug_names],
            return_exceptions=True,
        )

        stored_drugs: list[str] = []
        missing_drugs: list[str] = []

        for drug, result in zip(drug_names, results):
            if isinstance(result, Exception):
                logger.warning(
                    "DailyMed fetch failed for %s: %s",
                    drug,
                    result,
                    extra={"request_id": rid},
                )
                missing_drugs.append(drug)
                continue
            _, chunks, metas = result
            if not chunks:
                logger.warning(
                    "No DailyMed leaflet found for drug: %s",
                    drug,
                    extra={"request_id": rid},
                )
                missing_drugs.append(drug)
                continue
            embeddings = await embed(chunks, embed_executor)
            await store(chunks, embeddings, metas)
            stored_drugs.append(drug)

        await save_upload_result(session_id, stored_drugs, missing_drugs)
        await save_job_status(
            job_id,
            session_id,
            "done",
            drugs_found=stored_drugs,
            missing_leaflets=missing_drugs,
        )

        logger.info(
            "ingestion done: %d stored, %d missing",
            len(stored_drugs),
            len(missing_drugs),
            extra={"request_id": rid},
        )

    except Exception as exc:
        logger.exception(
            "ingestion failed for job %s: %s", job_id, exc, extra={"request_id": rid}
        )
        await save_job_status(job_id, session_id, "failed", error=str(exc))


@router.post("/upload", response_model=UploadJobResponse, status_code=202)
@limiter.limit(settings.upload_rate_limit)
async def upload(
    request: Request,
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    file: UploadFile = File(...),
) -> UploadJobResponse:
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

    try:
        text = await run_sync(extract_text, raw_bytes)
    except PDFExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if not text:
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    job_id = str(uuid.uuid4())
    await save_job_status(job_id, session_id, "processing")

    embed_executor = getattr(request.app.state, "embed_executor", None)
    background_tasks.add_task(
        _run_ingestion, job_id, session_id, text, rid, embed_executor
    )

    logger.info(
        "upload accepted, job %s started for session %s",
        job_id,
        session_id,
        extra={"request_id": rid},
    )

    return UploadJobResponse(job_id=job_id, session_id=session_id, status="processing")


@router.get("/upload/status/{job_id}", response_model=JobStatusResponse)
async def upload_status(job_id: str) -> JobStatusResponse:
    data = await get_job_status(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatusResponse(
        job_id=job_id,
        session_id=data["session_id"],
        status=data["status"],
        drugs_found=data.get("drugs_found") or [],
        missing_leaflets=data.get("missing_leaflets") or [],
        error=data.get("error"),
    )
