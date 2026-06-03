import math

from fastapi import APIRouter, Depends

from app.dependencies import require_api_key, verify_session_owner
from app.models.schemas import (
    InteractionFlag,
    InteractionsRequest,
    InteractionsResponse,
)
from app.services.interaction_detector import detect_interactions
from app.services.session_store import get_upload_result

router = APIRouter()


@router.post("/interactions", response_model=InteractionsResponse)
async def interactions(
    body: InteractionsRequest,
    caller_hash: str = Depends(require_api_key),
) -> InteractionsResponse:
    """Return drug-interaction flags for all drugs in *session_id*.

    Checks every (drug_a, drug_b) pair — where drug_a's official
    Drug Interactions leaflet section mentions drug_b by name.
    """
    await verify_session_owner(body.session_id, caller_hash)
    drugs_found, _ = await get_upload_result(body.session_id)
    n = len(drugs_found)
    pairs_checked = math.comb(n, 2)  # n-choose-2 unordered pairs

    flags: list[InteractionFlag] = await detect_interactions(body.session_id)

    return InteractionsResponse(
        session_id=body.session_id,
        pairs_checked=pairs_checked,
        interactions=flags,
    )
