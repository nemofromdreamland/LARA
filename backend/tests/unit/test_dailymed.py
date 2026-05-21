from unittest.mock import patch

import httpx
import pytest
import respx

from app.config import settings
from app.services.dailymed import (
    LeafletSection,
    _fetch_set_id,
    _normalize_drug_name,
    _parse_sections,
    _parse_spl_xml,
    clear_dailymed_cache,
    fetch_leaflet_sections,
)


@pytest.fixture(autouse=True)
def reset_cache():
    clear_dailymed_cache()
    yield
    clear_dailymed_cache()


DAILYMED_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

SEARCH_URL = f"{DAILYMED_BASE}/spls.json"
SPL_URL = f"{DAILYMED_BASE}/spls/test-set-id-123.xml"

MOCK_SEARCH_RESPONSE = {
    "data": [{"setid": "test-set-id-123", "title": "LISINOPRIL tablet"}],
    "metadata": {"total_elements": 1},
}

# Minimal SPL XML with 3 known sections + 1 unknown code + 1 empty text
MOCK_SPL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<document xmlns="urn:hl7-org:v3">
  <component>
    <structuredBody>
      <component>
        <section>
          <code code="34067-9" codeSystem="2.16.840.1.113883.6.1"
                displayName="INDICATIONS &amp; USAGE SECTION"/>
          <text><paragraph>Lisinopril is indicated for hypertension.</paragraph></text>
        </section>
      </component>
      <component>
        <section>
          <code code="34068-7" codeSystem="2.16.840.1.113883.6.1"
                displayName="DOSAGE &amp; ADMINISTRATION SECTION"/>
          <text><paragraph>Take 10mg once daily.</paragraph></text>
        </section>
      </component>
      <component>
        <section>
          <code code="34071-1" codeSystem="2.16.840.1.113883.6.1"
                displayName="WARNINGS SECTION"/>
          <text><paragraph>Do not use in pregnancy.</paragraph></text>
        </section>
      </component>
      <component>
        <section>
          <code code="99999-9" codeSystem="2.16.840.1.113883.6.1"
                displayName="UNKNOWN SECTION"/>
          <text><paragraph>Some other section.</paragraph></text>
        </section>
      </component>
      <component>
        <section>
          <code code="34070-3" codeSystem="2.16.840.1.113883.6.1"
                displayName="CONTRAINDICATIONS SECTION"/>
          <text></text>
        </section>
      </component>
    </structuredBody>
  </component>
</document>"""

# Raw dicts as returned by _parse_spl_xml — used by _parse_sections tests
_RAW_SECTIONS = [
    {"loinc_code": "34067-9", "text": "Lisinopril is indicated for hypertension."},
    {"loinc_code": "34068-7", "text": "Take 10mg once daily."},
    {"loinc_code": "34071-1", "text": "Do not use in pregnancy."},
    {"loinc_code": "99999-9", "text": "Some other section."},
    {"loinc_code": "34070-3", "text": ""},
]


# ── _fetch_set_id ─────────────────────────────────────────────────────────────


@respx.mock
async def test_fetch_set_id_returns_setid():
    respx.get(SEARCH_URL).mock(
        return_value=httpx.Response(200, json=MOCK_SEARCH_RESPONSE)
    )
    async with httpx.AsyncClient() as client:
        result = await _fetch_set_id("lisinopril", client)
    assert result == "test-set-id-123"


@respx.mock
async def test_fetch_set_id_returns_none_when_not_found():
    respx.get(SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": [], "metadata": {}})
    )
    async with httpx.AsyncClient() as client:
        result = await _fetch_set_id("unknowndrug", client)
    assert result is None


@respx.mock
async def test_fetch_set_id_raises_on_http_error():
    respx.get(SEARCH_URL).mock(return_value=httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        async with httpx.AsyncClient() as client:
            await _fetch_set_id("lisinopril", client)


# ── _parse_spl_xml ────────────────────────────────────────────────────────────


def test_parse_spl_xml_returns_list_of_dicts():
    result = _parse_spl_xml(MOCK_SPL_XML)
    assert isinstance(result, list)
    assert all(isinstance(r, dict) for r in result)


def test_parse_spl_xml_extracts_loinc_codes():
    result = _parse_spl_xml(MOCK_SPL_XML)
    codes = [r["loinc_code"] for r in result]
    assert "34067-9" in codes
    assert "34068-7" in codes
    assert "34071-1" in codes


def test_parse_spl_xml_extracts_text():
    result = _parse_spl_xml(MOCK_SPL_XML)
    indications = next(r for r in result if r["loinc_code"] == "34067-9")
    assert "hypertension" in indications["text"]


def test_parse_spl_xml_dict_has_required_keys():
    result = _parse_spl_xml(MOCK_SPL_XML)
    for item in result:
        assert "loinc_code" in item
        assert "text" in item


# ── _parse_sections ───────────────────────────────────────────────────────────


def test_parse_sections_maps_known_loinc_codes():
    sections = _parse_sections(_RAW_SECTIONS, "lisinopril")
    section_names = {s.section for s in sections}
    assert "indications" in section_names
    assert "dosage" in section_names
    assert "warnings" in section_names


def test_parse_sections_ignores_unknown_loinc():
    sections = _parse_sections(_RAW_SECTIONS, "lisinopril")
    assert len(sections) == 3  # 3 valid non-empty sections


def test_parse_sections_ignores_empty_text():
    sections = _parse_sections(_RAW_SECTIONS, "lisinopril")
    assert all(s.text for s in sections)


def test_parse_sections_sets_drug_name():
    raw = [{"loinc_code": "34067-9", "text": "Indicated for hypertension."}]
    sections = _parse_sections(raw, "lisinopril")
    assert all(s.drug_name == "lisinopril" for s in sections)


def test_parse_sections_returns_leaflet_section_dataclass():
    raw = [{"loinc_code": "34068-7", "text": "Take once daily."}]
    sections = _parse_sections(raw, "lisinopril")
    assert isinstance(sections[0], LeafletSection)


# ── fetch_leaflet_sections (end-to-end) ───────────────────────────────────────


@respx.mock
async def test_fetch_leaflet_sections_end_to_end():
    respx.get(SEARCH_URL).mock(
        return_value=httpx.Response(200, json=MOCK_SEARCH_RESPONSE)
    )
    respx.get(SPL_URL).mock(return_value=httpx.Response(200, text=MOCK_SPL_XML))
    sections = await fetch_leaflet_sections("lisinopril")
    assert len(sections) == 3
    assert all(isinstance(s, LeafletSection) for s in sections)


@respx.mock
async def test_fetch_leaflet_sections_returns_empty_for_unknown_drug():
    respx.get(SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": [], "metadata": {}})
    )
    sections = await fetch_leaflet_sections("notadrug")
    assert sections == []


# ── _normalize_drug_name ──────────────────────────────────────────────────────


def test_normalize_strips_mg_dosage():
    assert _normalize_drug_name("Sertraline 50mg") == "sertraline"


def test_normalize_strips_mg_with_space():
    assert _normalize_drug_name("Lisinopril 10 mg") == "lisinopril"


def test_normalize_strips_mcg():
    assert _normalize_drug_name("Levothyroxine 25 mcg") == "levothyroxine"


def test_normalize_no_dosage_still_lowercases():
    assert _normalize_drug_name("Metformin") == "metformin"


def test_normalize_already_lowercase_no_dosage():
    assert _normalize_drug_name("aspirin") == "aspirin"


# ── fetch_leaflet_sections normalization fallback ─────────────────────────────


@respx.mock
async def test_fetch_leaflet_sections_retries_with_normalized_name():
    """If the original name returns no results, retry with the normalized form."""
    empty = httpx.Response(200, json={"data": [], "metadata": {}})
    found = httpx.Response(200, json=MOCK_SEARCH_RESPONSE)

    # First call (original name) → empty; second call (normalized) → found
    respx.get(SEARCH_URL).mock(side_effect=[empty, found])
    respx.get(SPL_URL).mock(return_value=httpx.Response(200, text=MOCK_SPL_XML))

    sections = await fetch_leaflet_sections("Sertraline 50mg")
    assert len(sections) > 0


@respx.mock
async def test_fetch_leaflet_sections_no_retry_when_already_normalized():
    """If name has no dosage, only one search call is made."""
    respx.get(SEARCH_URL).mock(
        return_value=httpx.Response(200, json={"data": [], "metadata": {}})
    )
    sections = await fetch_leaflet_sections("notadrug")
    assert sections == []
    assert respx.calls.call_count == 1


# ── TTL cache ─────────────────────────────────────────────────────────────────


@respx.mock
async def test_cache_hit_calls_http_exactly_once():
    """Two calls for the same drug name → only one round-trip to DailyMed."""
    respx.get(SEARCH_URL).mock(
        return_value=httpx.Response(200, json=MOCK_SEARCH_RESPONSE)
    )
    respx.get(SPL_URL).mock(return_value=httpx.Response(200, text=MOCK_SPL_XML))

    first = await fetch_leaflet_sections("lisinopril")
    second = await fetch_leaflet_sections("lisinopril")

    assert first == second
    # search + xml = 2 calls total; second call is served from cache
    assert respx.calls.call_count == 2


@respx.mock
async def test_expired_cache_calls_http_again():
    """An entry past its TTL is evicted and the API is called again."""
    respx.get(SEARCH_URL).mock(
        return_value=httpx.Response(200, json=MOCK_SEARCH_RESPONSE)
    )
    respx.get(SPL_URL).mock(return_value=httpx.Response(200, text=MOCK_SPL_XML))

    with patch("app.services.dailymed.time.time", return_value=0.0):
        await fetch_leaflet_sections("lisinopril")

    past_ttl = float(settings.dailymed_cache_ttl_seconds + 1)
    with patch("app.services.dailymed.time.time", return_value=past_ttl):
        await fetch_leaflet_sections("lisinopril")

    # 2 HTTP calls per fetch × 2 fetches = 4
    assert respx.calls.call_count == 4


@respx.mock
async def test_clear_dailymed_cache_resets_state():
    """After clearing the cache, the next call hits the API again."""
    respx.get(SEARCH_URL).mock(
        return_value=httpx.Response(200, json=MOCK_SEARCH_RESPONSE)
    )
    respx.get(SPL_URL).mock(return_value=httpx.Response(200, text=MOCK_SPL_XML))

    await fetch_leaflet_sections("lisinopril")
    clear_dailymed_cache()
    await fetch_leaflet_sections("lisinopril")

    assert respx.calls.call_count == 4
