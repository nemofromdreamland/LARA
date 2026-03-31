from fastapi import APIRouter

from app.models.schemas import UploadResponse

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload() -> UploadResponse:
    # Implemented in Step 6
    raise NotImplementedError
