import logging
from dataclasses import dataclass

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

DAILYMED_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

# LOINC code → human-readable section name stored in metadata
LOINC_SECTIONS: dict[str, str] = {
    "34066-1": "boxed_warnings",
    "34067-9": "indications",
    "34068-7": "dosage",
    "34070-3": "contraindications",
    "34073-7": "drug_interactions",
    "34084-4": "adverse_reactions",
    "34071-1": "warnings",
}


@dataclass
class LeafletSection:
    drug_name: str
    section: str
    text: str


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
async def _fetch_set_id(drug_name: str, client: httpx.AsyncClient) -> str | None:
    """Return the first SPL set-id for *drug_name*, or None if not found."""
    response = await client.get(
        f"{DAILYMED_BASE}/spls.json",
        params={"drug_name": drug_name, "pagesize": 1},
    )
    response.raise_for_status()
    items = response.json().get("data", [])
    if not items:
        logger.warning("DailyMed: no SPL found for drug %r", drug_name)
        return None
    return items[0].get("setid")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
async def _fetch_sections_raw(
    set_id: str, client: httpx.AsyncClient
) -> list[dict]:
    """Return the raw sections list from a DailyMed SPL document."""
    response = await client.get(f"{DAILYMED_BASE}/spls/{set_id}.json")
    response.raise_for_status()
    return response.json().get("data", {}).get("sections", [])


def _parse_sections(raw: list[dict], drug_name: str) -> list[LeafletSection]:
    """Filter and map raw DailyMed sections to LeafletSection dataclasses."""
    results: list[LeafletSection] = []
    for section in raw:
        code = section.get("loinc_code", "")
        text = (section.get("text") or "").strip()
        if code in LOINC_SECTIONS and text:
            results.append(
                LeafletSection(
                    drug_name=drug_name,
                    section=LOINC_SECTIONS[code],
                    text=text,
                )
            )
    return results


async def fetch_leaflet_sections(drug_name: str) -> list[LeafletSection]:
    """Fetch and parse official leaflet sections for *drug_name* from DailyMed.

    Returns an empty list (with a warning) if the drug is not found.
    Raises httpx.HTTPError after 3 retries if the API is unavailable.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        set_id = await _fetch_set_id(drug_name, client)
        if set_id is None:
            return []
        raw = await _fetch_sections_raw(set_id, client)
        return _parse_sections(raw, drug_name)
