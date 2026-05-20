import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings
from app.utils import get_request_id

logger = logging.getLogger(__name__)

DAILYMED_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

_cache: dict[str, tuple[list, float]] = {}  # normalized_drug_name → (sections, timestamp)


def _cache_get(drug_name: str) -> list | None:
    entry = _cache.get(drug_name)
    if entry and (time.time() - entry[1]) < settings.dailymed_cache_ttl_seconds:
        return entry[0]
    return None


def _cache_set(drug_name: str, sections: list) -> None:
    _cache[drug_name] = (sections, time.time())


def clear_dailymed_cache() -> None:
    _cache.clear()

# Matches trailing dosage info: "50 mg", "10mg", "0.5 mcg/ml", etc.
_DOSAGE_RE = re.compile(r"\s+\d[\d.,]*\s*(?:mg|mcg|ml|g|iu|%|units?)\S*", re.IGNORECASE)


def _is_retryable(exc: BaseException) -> bool:
    """Return True only for errors worth retrying.

    Retries 5xx, 429 (rate limit), and network-level failures.
    Does NOT retry 404 (drug not found — permanent) or other 4xx.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(
        exc, (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError)
    )


def _normalize_drug_name(name: str) -> str:
    """Strip trailing dosage/strength info and lowercase.

    Examples:
      "Sertraline 50mg"   → "sertraline"
      "Lisinopril 10 mg"  → "lisinopril"
      "Metformin"         → "metformin"
    """
    return _DOSAGE_RE.sub("", name).strip().lower()


# LOINC code → human-readable section name stored in metadata
LOINC_SECTIONS: dict[str, str] = {
    "34066-1": "boxed_warnings",
    "34067-9": "indications",
    "34068-7": "dosage",
    "34070-3": "contraindications",
    "34073-7": "drug_interactions",
    "34084-4": "adverse_reactions",
    "34071-1": "warnings",
    "43685-7": "warnings_and_precautions",
    "42228-7": "pregnancy",
    "34077-8": "teratogenic_effects",
    "34078-6": "nonteratogenic_effects",
    "34080-2": "nursing_mothers",
    "34081-0": "pediatric_use",
    "34083-6": "geriatric_use",
}


@dataclass
class LeafletSection:
    drug_name: str
    section: str
    text: str


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_is_retryable),
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
        logger.warning(
            "DailyMed: no SPL found for drug %r", drug_name,
            extra={"request_id": get_request_id()},
        )
        return None
    return items[0].get("setid")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
async def _fetch_sections_raw(set_id: str, client: httpx.AsyncClient) -> list[dict]:
    """Return the raw sections list from a DailyMed SPL XML document.

    DailyMed v2 only provides full section content via XML (the JSON API has
    no sections endpoint). We parse the HL7 v3 XML and return dicts with
    loinc_code and text keys — the same shape _parse_sections expects.
    """
    response = await client.get(f"{DAILYMED_BASE}/spls/{set_id}.xml")
    response.raise_for_status()
    return _parse_spl_xml(response.text)


_SPL_NS = "urn:hl7-org:v3"


def _parse_spl_xml(xml_text: str) -> list[dict]:
    """Extract sections with LOINC codes from an SPL XML document."""
    root = ET.fromstring(xml_text)
    ns = {"h": _SPL_NS}
    results: list[dict] = []
    for section in root.findall(".//h:section", ns):
        code_el = section.find("h:code", ns)
        if code_el is None:
            continue
        loinc_code = code_el.get("code", "")
        text_el = section.find("h:text", ns)
        text = " ".join(text_el.itertext()).strip() if text_el is not None else ""
        results.append({"loinc_code": loinc_code, "text": text})
    return results


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

    If the original name returns no results, retries once with a normalized
    form (lowercase, dosage info stripped) before giving up.

    Results are cached in-process by normalized drug name for
    settings.dailymed_cache_ttl_seconds (default 24 h) to avoid redundant
    API calls when the same drug appears across multiple uploads.

    Returns an empty list (with a warning) if the drug is not found.
    Raises httpx.HTTPError after 3 retries if the API is unavailable.
    """
    normalized_name = _normalize_drug_name(drug_name)

    cached = _cache_get(normalized_name)
    if cached is not None:
        logger.debug("DailyMed cache hit: %s", drug_name)
        return cached

    rid = get_request_id()
    async with httpx.AsyncClient(timeout=15.0) as client:
        set_id = await _fetch_set_id(drug_name, client)

        if set_id is None:
            if normalized_name and normalized_name != drug_name.lower():
                logger.info(
                    "DailyMed: retrying %r with normalized name %r",
                    drug_name,
                    normalized_name,
                    extra={"request_id": rid},
                )
                set_id = await _fetch_set_id(normalized_name, client)

        if set_id is None:
            sections: list[LeafletSection] = []
        else:
            raw = await _fetch_sections_raw(set_id, client)
            sections = _parse_sections(raw, drug_name)

    _cache_set(normalized_name, sections)
    return sections
