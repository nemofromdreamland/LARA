import logging
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)

from app.config import settings
from app.dependencies import require_api_key, verify_session_owner
from app.limiter import limiter
from app.models.schemas import JobStatusResponse, UploadJobResponse
from app.services.ingestion import run_ingestion
from app.services.pdf_parser import PDFExtractionError, extract_text
from app.services.session_store import get_job_status, save_job_status
from app.utils import get_request_id, run_sync

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload", response_model=UploadJobResponse, status_code=202)
@limiter.limit(settings.upload_rate_limit)
async def upload(
    request: Request,
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    file: UploadFile = File(...),
    caller_hash: str = Depends(require_api_key),
) -> UploadJobResponse:
    await verify_session_owner(session_id, caller_hash)
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
        run_ingestion, job_id, session_id, text, rid, embed_executor
    )

    logger.info(
        "upload accepted, job %s started for session %s",
        job_id,
        session_id,
        extra={"request_id": rid},
    )

    return UploadJobResponse(job_id=job_id, session_id=session_id, status="processing")


@router.get("/upload/status/{job_id}", response_model=JobStatusResponse)
async def upload_status(
    job_id: str,
    session_id: str = Query(...),
    caller_hash: str = Depends(require_api_key),
) -> JobStatusResponse:
    await verify_session_owner(session_id, caller_hash)
    data = await get_job_status(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if data["session_id"] != session_id:
        raise HTTPException(
            status_code=403, detail="Job belongs to a different session."
        )
    return JobStatusResponse(
        job_id=job_id,
        session_id=data["session_id"],
        status=data["status"],
        drugs_found=data.get("drugs_found") or [],
        missing_leaflets=data.get("missing_leaflets") or [],
        error=data.get("error"),
    )
