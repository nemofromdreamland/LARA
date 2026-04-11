import json
from unittest.mock import AsyncMock, patch

from app.services.prescription_parser import _strip_markdown, parse_prescription

# ---------------------------------------------------------------------------
# _strip_markdown helper
# ---------------------------------------------------------------------------


def test_strip_markdown_removes_json_fence():
    assert _strip_markdown("```json\n[]\n```") == "[]"


def test_strip_markdown_removes_plain_fence():
    assert _strip_markdown("```\n[]\n```") == "[]"


def test_strip_markdown_leaves_plain_json():
    assert _strip_markdown('[{"a": 1}]') == '[{"a": 1}]'


# ---------------------------------------------------------------------------
# parse_prescription — happy path
# ---------------------------------------------------------------------------

_VALID_JSON = json.dumps(
    [
        {
            "drug_name": "Ibuprofen",
            "dosage": "400mg",
            "frequency": "three times daily",
            "duration": "14 days",
            "instructions": "Take with food",
        },
        {
            "drug_name": "Azithromycin",
            "dosage": "500mg",
            "frequency": "once daily",
            "duration": "3 days",
            "instructions": None,
        },
    ]
)


@patch("app.services.prescription_parser.extract_medications", new_callable=AsyncMock)
async def test_parse_returns_structured_entries(mock_extract):
    mock_extract.return_value = _VALID_JSON
    entries = await parse_prescription("some prescription text")

    assert len(entries) == 2
    assert entries[0].drug_name == "ibuprofen"
    assert entries[0].dosage == "400mg"
    assert entries[0].frequency == "three times daily"
    assert entries[0].duration == "14 days"
    assert entries[0].instructions == "Take with food"
    assert entries[1].drug_name == "azithromycin"
    assert entries[1].instructions is None


@patch("app.services.prescription_parser.extract_medications", new_callable=AsyncMock)
async def test_parse_strips_markdown_fences(mock_extract):
    mock_extract.return_value = f"```json\n{_VALID_JSON}\n```"
    entries = await parse_prescription("prescription")
    assert len(entries) == 2
    assert entries[0].drug_name == "ibuprofen"


@patch("app.services.prescription_parser.extract_medications", new_callable=AsyncMock)
async def test_parse_lowercases_drug_names(mock_extract):
    mock_extract.return_value = json.dumps(
        [{"drug_name": "IBUPROFEN", "dosage": "400mg"}]
    )
    entries = await parse_prescription("prescription")
    assert entries[0].drug_name == "ibuprofen"


@patch("app.services.prescription_parser.extract_medications", new_callable=AsyncMock)
async def test_parse_ignores_entries_without_drug_name(mock_extract):
    mock_extract.return_value = json.dumps(
        [
            {"drug_name": "", "dosage": "400mg"},
            {"dosage": "400mg"},  # no drug_name key
            {"drug_name": "aspirin", "dosage": "81mg"},
        ]
    )
    entries = await parse_prescription("prescription")
    assert len(entries) == 1
    assert entries[0].drug_name == "aspirin"


# ---------------------------------------------------------------------------
# parse_prescription — fallback paths
# ---------------------------------------------------------------------------


@patch("app.services.prescription_parser.extract_drug_names")
@patch("app.services.prescription_parser.extract_medications", new_callable=AsyncMock)
async def test_parse_falls_back_on_invalid_json(mock_extract, mock_regex):
    mock_extract.return_value = "not valid json at all"
    mock_regex.return_value = ["ibuprofen", "azithromycin"]

    entries = await parse_prescription("prescription")
    assert len(entries) == 2
    assert entries[0].drug_name == "ibuprofen"
    assert entries[0].dosage is None  # fallback entries have no structured fields


@patch("app.services.prescription_parser.extract_drug_names")
@patch("app.services.prescription_parser.extract_medications", new_callable=AsyncMock)
async def test_parse_falls_back_on_llm_exception(mock_extract, mock_regex):
    mock_extract.side_effect = RuntimeError("LLM is down")
    mock_regex.return_value = ["lisinopril"]

    entries = await parse_prescription("prescription")
    assert len(entries) == 1
    assert entries[0].drug_name == "lisinopril"
    assert entries[0].dosage is None


@patch("app.services.prescription_parser.extract_drug_names")
@patch("app.services.prescription_parser.extract_medications", new_callable=AsyncMock)
async def test_parse_falls_back_when_llm_returns_empty_list(mock_extract, mock_regex):
    mock_extract.return_value = "[]"
    mock_regex.return_value = ["metformin"]

    entries = await parse_prescription("prescription")
    assert len(entries) == 1
    assert entries[0].drug_name == "metformin"


@patch("app.services.prescription_parser.extract_drug_names")
@patch("app.services.prescription_parser.extract_medications", new_callable=AsyncMock)
async def test_parse_returns_empty_when_both_methods_find_nothing(
    mock_extract, mock_regex
):
    mock_extract.return_value = "[]"
    mock_regex.return_value = []

    entries = await parse_prescription("prescription")
    assert entries == []
