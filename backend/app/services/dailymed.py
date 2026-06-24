import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import settings
from app.utils import get_request_id

logger = logging.getLogger(__name__)

DAILYMED_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

_CACHE_PREFIX = "dailymed:"

# Singleton httpx client — reuses the underlying connection pool across calls,
# avoiding a fresh TCP+TLS handshake on every DailyMed request. Mirrors the
# Cerebras client pattern in llm_client.py. When None (e.g. in tests that don't
# run the app lifespan) callers fall back to a temporary per-call client.
_dailymed_client: httpx.AsyncClient | None = None


async def init_dailymed_client() -> None:
    """Create the shared DailyMed httpx client. Call from app lifespan startup."""
    global _dailymed_client
    _dailymed_client = httpx.AsyncClient(timeout=15.0)


async def close_dailymed_client() -> None:
    """Close the shared DailyMed httpx client. Call from app lifespan teardown."""
    global _dailymed_client
    if _dailymed_client is not None:
        await _dailymed_client.aclose()
        _dailymed_client = None


async def _cache_get(drug_name: str) -> list | None:
    from app.services.session_store import get_redis

    try:
        r = get_redis()
        raw = await r.get(f"{_CACHE_PREFIX}{drug_name}")
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.warning("DailyMed cache read failed for %r: %s", drug_name, exc)
        return None


async def _cache_set(drug_name: str, sections: list) -> None:
    from app.services.session_store import get_redis

    try:
        r = get_redis()
        payload = json.dumps([asdict(s) for s in sections])
        await r.setex(
            f"{_CACHE_PREFIX}{drug_name}",
            settings.dailymed_cache_ttl_seconds,
            payload,
        )
    except Exception as exc:
        logger.warning("DailyMed cache write failed for %r: %s", drug_name, exc)


async def clear_dailymed_cache() -> None:
    from app.services.session_store import get_redis

    try:
        r = get_redis()
        keys = await r.keys(f"{_CACHE_PREFIX}*")
        if keys:
            await r.delete(*keys)
    except Exception:
        pass


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


def _sections_from_cache(raw: list[dict]) -> list[LeafletSection]:
    return [LeafletSection(**item) for item in raw]


def _title_matches(drug_name: str, title: str) -> bool:
    """Return True if *drug_name* appears as a whole word in the SPL title."""
    pattern = re.compile(r"\b" + re.escape(drug_name) + r"\b", re.IGNORECASE)
    return bool(pattern.search(title))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
async def _fetch_set_id(drug_name: str, client: httpx.AsyncClient) -> str | None:
    """Return the best SPL set-id for *drug_name*, or None if not found.

    Prefers the first result whose title mentions the drug name as a whole
    word; the search API ranks loosely, so the top hit can be a combination
    product or an unrelated formulation. Falls back to the first result when
    no title matches.
    """
    response = await client.get(
        f"{DAILYMED_BASE}/spls.json",
        params={"drug_name": drug_name, "pagesize": 5},
    )
    response.raise_for_status()
    items = response.json().get("data", [])
    if not items:
        logger.warning(
            "DailyMed: no SPL found for drug %r",
            drug_name,
            extra={"request_id": get_request_id()},
        )
        return None
    for item in items:
        if _title_matches(drug_name, item.get("title", "")):
            return item.get("setid")
    logger.warning(
        "DailyMed: no SPL title matched %r — falling back to first result %r",
        drug_name,
        items[0].get("title", ""),
        extra={"request_id": get_request_id()},
    )
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
# Collapses any run of whitespace (spaces, tabs, newlines) to a single space.
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_spl_text(raw: str) -> str:
    """Normalise whitespace in text extracted from SPL XML.

    HL7 SPL documents embed table markup, list numbering, and cross-reference
    artefacts that produce long runs of whitespace when itertext() is joined.
    Collapsing these to single spaces keeps chunks coherent.
    """
    return _WHITESPACE_RE.sub(" ", raw).strip()


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
        raw_text = " ".join(text_el.itertext()) if text_el is not None else ""
        results.append({"loinc_code": loinc_code, "text": _clean_spl_text(raw_text)})
    return results


def _parse_sections(raw: list[dict], drug_name: str) -> list[LeafletSection]:
    """Filter and map raw DailyMed sections to LeafletSection dataclasses."""
    results: list[LeafletSection] = []
    for section in raw:
        code = section.get("loinc_code", "")
        text = (section.get("text") or "").strip()
        if not text:
            continue
        if code not in LOINC_SECTIONS:
            logger.debug(
                "DailyMed: skipping unknown LOINC code %r for drug %r",
                code,
                drug_name,
            )
            continue
        results.append(
            LeafletSection(
                drug_name=drug_name,
                section=LOINC_SECTIONS[code],
                text=text,
            )
        )
    return results


async def _resolve_and_fetch(
    client: httpx.AsyncClient, drug_name: str, normalized_name: str
) -> list[LeafletSection]:
    """Resolve the SPL set-id for *drug_name* and fetch its leaflet sections.

    Tries the original name first, then the normalized form if the original
    returns no results. Uses *client* for every request so the caller controls
    whether it's the pooled singleton or a temporary per-call client.
    """
    set_id = await _fetch_set_id(drug_name, client)

    if set_id is None and normalized_name and normalized_name != drug_name.lower():
        logger.info(
            "DailyMed: retrying %r with normalized name %r",
            drug_name,
            normalized_name,
            extra={"request_id": get_request_id()},
        )
        set_id = await _fetch_set_id(normalized_name, client)

    if set_id is None:
        return []
    raw = await _fetch_sections_raw(set_id, client)
    return _parse_sections(raw, drug_name)


async def fetch_leaflet_sections(drug_name: str) -> list[LeafletSection]:
    """Fetch and parse official leaflet sections for *drug_name* from DailyMed.

    If the original name returns no results, retries once with a normalized
    form (lowercase, dosage info stripped) before giving up.

    Results are cached in Redis by normalized drug name for
    settings.dailymed_cache_ttl_seconds (default 24 h) to avoid redundant
    API calls when the same drug appears across multiple uploads or workers.

    Uses the pooled `_dailymed_client` when the app lifespan has initialised it;
    otherwise falls back to a temporary per-call client.

    Returns an empty list (with a warning) if the drug is not found.
    Raises httpx.HTTPError after 3 retries if the API is unavailable.
    """
    normalized_name = _normalize_drug_name(drug_name)

    cached = await _cache_get(normalized_name)
    if cached is not None:
        logger.debug("DailyMed cache hit: %s", drug_name)
        return _sections_from_cache(cached)

    if _dailymed_client is not None:
        sections = await _resolve_and_fetch(
            _dailymed_client, drug_name, normalized_name
        )
    else:
        async with httpx.AsyncClient(timeout=15.0) as client:
            sections = await _resolve_and_fetch(client, drug_name, normalized_name)

    await _cache_set(normalized_name, sections)
    return sections
