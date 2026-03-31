import httpx
import pytest
import respx

from app.services.dailymed import (
    LeafletSection,
    _fetch_set_id,
    _parse_sections,
    fetch_leaflet_sections,
)

DAILYMED_BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

SEARCH_URL = f"{DAILYMED_BASE}/spls.json"
SPL_URL = f"{DAILYMED_BASE}/spls/test-set-id-123.json"

MOCK_SEARCH_RESPONSE = {
    "data": [{"setid": "test-set-id-123", "title": "LISINOPRIL tablet"}],
    "metadata": {"total_elements": 1},
}

MOCK_SPL_RESPONSE = {
    "data": {
        "setid": "test-set-id-123",
        "sections": [
            {
                "loinc_code": "34067-9",
                "text": "Lisinopril is indicated for hypertension.",
            },
            {"loinc_code": "34068-7", "text": "Take 10mg once daily."},
            {"loinc_code": "34071-1", "text": "Do not use in pregnancy."},
            # Unknown LOINC code — should be ignored
            {"loinc_code": "99999-9", "text": "Some other section."},
            # Empty text — should be ignored
            {"loinc_code": "34070-3", "text": ""},
        ],
    }
}


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


def test_parse_sections_maps_known_loinc_codes():
    raw = MOCK_SPL_RESPONSE["data"]["sections"]
    sections = _parse_sections(raw, "lisinopril")
    section_names = {s.section for s in sections}
    assert "indications" in section_names
    assert "dosage" in section_names
    assert "warnings" in section_names


def test_parse_sections_ignores_unknown_loinc():
    raw = MOCK_SPL_RESPONSE["data"]["sections"]
    sections = _parse_sections(raw, "lisinopril")
    # The unknown code "99999-9" must not appear
    assert len(sections) == 3  # 3 valid non-empty sections


def test_parse_sections_ignores_empty_text():
    raw = MOCK_SPL_RESPONSE["data"]["sections"]
    sections = _parse_sections(raw, "lisinopril")
    # contraindications section has empty text — must be excluded
    assert all(s.text for s in sections)


def test_parse_sections_sets_drug_name():
    raw = [{"loinc_code": "34067-9", "text": "Indicated for hypertension."}]
    sections = _parse_sections(raw, "lisinopril")
    assert all(s.drug_name == "lisinopril" for s in sections)


def test_parse_sections_returns_leaflet_section_dataclass():
    raw = [{"loinc_code": "34068-7", "text": "Take once daily."}]
    sections = _parse_sections(raw, "lisinopril")
    assert isinstance(sections[0], LeafletSection)


@respx.mock
async def test_fetch_leaflet_sections_end_to_end():
    respx.get(SEARCH_URL).mock(
        return_value=httpx.Response(200, json=MOCK_SEARCH_RESPONSE)
    )
    respx.get(SPL_URL).mock(return_value=httpx.Response(200, json=MOCK_SPL_RESPONSE))
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
