import logging
import uuid

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from app.config import settings
from app.dependencies import require_api_key, verify_session_owner
from app.limiter import limiter
from app.models.schemas import (
    SampleInfo,
    SampleListResponse,
    SampleLoadRequest,
    UploadJobResponse,
)
from app.services.ingestion_queue import enqueue_ingestion
from app.services.pdf_parser import PDFExtractionError, extract_text
from app.services.samples import (
    load_manifest,
    sample_pdf_path,
    seed_sample_leaflet_cache,
)
from app.services.session_store import save_job_status
from app.utils import get_request_id, run_sync

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/samples", response_model=SampleListResponse)
async def list_samples() -> SampleListResponse:
    return SampleListResponse(
        samples=[SampleInfo(**entry) for entry in load_manifest().values()]
    )


@router.post("/samples/{sample_id}", response_model=UploadJobResponse, status_code=202)
@limiter.limit(settings.upload_rate_limit)
async def load_sample(
    request: Request,
    sample_id: str,
    body: SampleLoadRequest,
    _api_key: str = Depends(require_api_key),
    x_session_token: str | None = Header(default=None, alias="X-Session-Token"),
) -> UploadJobResponse:
    """Start ingestion of a bundled sample prescription.

    Same 202 job contract as /upload; poll GET /upload/status/{job_id}.
    """
    await verify_session_owner(body.session_id, x_session_token)
    rid = get_request_id()

    sample = load_manifest().get(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Unknown sample.")

    raw_bytes = await run_sync(sample_pdf_path(sample_id).read_bytes)
    try:
        text = await run_sync(extract_text, raw_bytes)
    except PDFExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Re-seed this sample's leaflets so ingestion hits the cache even if the
    # startup seed's TTL has lapsed — the demo never needs live DailyMed.
    await seed_sample_leaflet_cache(drugs=sample["drugs"])

    job_id = str(uuid.uuid4())
    await save_job_status(job_id, body.session_id, "processing")
    await enqueue_ingestion(job_id, body.session_id, text, rid)

    logger.info(
        "sample %s accepted, job %s started for session %s",
        sample_id,
        job_id,
        body.session_id,
        extra={"request_id": rid},
    )

    return UploadJobResponse(
        job_id=job_id, session_id=body.session_id, status="processing"
    )
