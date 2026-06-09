import pytest

from app.services.drug_extractor import (
    _extract_regex,
    _extract_spacy,
    extract_drug_names,
    extract_prescription_entries,
)

_SAMPLE_RX = """
Patient: John Doe
Date: 2024-01-15

Lisinopril 10mg once daily
Metformin 500mg twice daily
Atorvastatin 20mg at bedtime
"""

_SAMPLE_SUFFIX = """
The patient is prescribed Omeprazole for gastric issues.
She is also taking Amoxicillin and Sertraline.
"""

_SAMPLE_NUMBERED = """
1. Acetaminophen
 • Dosage: 500mg
 • Frequency: Every 6 hours as needed
2. Tylenol
 • Dosage: 500mg
 • Frequency: Twice daily
"""


# ---------------------------------------------------------------------------
# extract_drug_names — integration (spaCy + regex combined)
# ---------------------------------------------------------------------------


def test_rx_line_extracts_known_drugs():
    names = extract_drug_names(_SAMPLE_RX)
    assert "lisinopril" in names
    assert "metformin" in names
    assert "atorvastatin" in names


def test_suffix_heuristic_extracts_drugs():
    names = extract_drug_names(_SAMPLE_SUFFIX)
    assert "omeprazole" in names
    assert "amoxicillin" in names
    assert "sertraline" in names


def test_results_are_lowercase():
    names = extract_drug_names(_SAMPLE_RX)
    assert all(n == n.lower() for n in names)


def test_no_duplicates():
    text = "Lisinopril 10mg daily\nLisinopril 20mg daily"
    names = extract_drug_names(text)
    assert names.count("lisinopril") == 1


def test_empty_text_returns_empty_list():
    assert extract_drug_names("") == []


def test_no_drugs_returns_empty_list():
    assert extract_drug_names("The patient feels well today.") == []


def test_mixed_rx_and_suffix():
    text = "Metformin 500mg daily\nShe also uses Sertraline for anxiety."
    names = extract_drug_names(text)
    assert "metformin" in names
    assert "sertraline" in names


@pytest.mark.parametrize(
    "drug,text",
    [
        ("amlodipine", "Amlodipine 5mg daily"),
        ("omeprazole", "Omeprazole 20mg daily"),
        ("atorvastatin", "Atorvastatin 40mg daily"),
    ],
)
def test_parametrized_rx_line(drug: str, text: str):
    assert drug in extract_drug_names(text)


def test_numbered_list_extracts_acetaminophen():
    names = extract_drug_names(_SAMPLE_NUMBERED)
    assert "acetaminophen" in names


def test_numbered_list_extracts_tylenol():
    names = extract_drug_names(_SAMPLE_NUMBERED)
    assert "tylenol" in names


def test_numbered_list_no_duplicates():
    names = extract_drug_names(_SAMPLE_NUMBERED)
    assert names.count("acetaminophen") == 1
    assert names.count("tylenol") == 1


def test_suffix_extracts_acetaminophen_via_phen():
    names = extract_drug_names("The patient takes Acetaminophen for pain relief.")
    assert "acetaminophen" in names


# ---------------------------------------------------------------------------
# _extract_spacy — unit tests (skipped gracefully if model absent)
# ---------------------------------------------------------------------------


def test_spacy_returns_list_for_inline_drugs():
    # _extract_spacy now only returns named entities that are not PERSON/ORG/GPE/etc.
    # Drug names not recognised as entities by en_core_web_sm return nothing here;
    # they are caught downstream by _extract_regex via pharmaceutical suffix patterns.
    result = _extract_spacy("The patient takes Omeprazole, Amoxicillin and Sertraline.")
    assert isinstance(result, list)
    # Verify the combined extractor still finds them (via _SUFFIX_RE).
    from app.services.drug_extractor import extract_drug_names

    names = extract_drug_names(
        "The patient takes Omeprazole, Amoxicillin and Sertraline."
    )
    assert "omeprazole" in names
    assert "amoxicillin" in names
    assert "sertraline" in names


def test_spacy_does_not_include_patient_name():
    # Common patient names should be filtered by the stoplist
    names = _extract_spacy("Patient: John Smith\nSertraline 50mg daily")
    assert "john" not in names
    assert "smith" not in names


def test_spacy_returns_lowercase():
    names = _extract_spacy("Omeprazole 20mg daily")
    assert all(n == n.lower() for n in names)


def test_spacy_returns_list():
    result = _extract_spacy("Lisinopril 10mg")
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# _extract_regex — unit tests (pure regex, no spaCy dependency)
# ---------------------------------------------------------------------------


def test_regex_rx_line():
    names = _extract_regex("Lisinopril 10mg once daily")
    assert "lisinopril" in names


def test_regex_numbered_list():
    names = _extract_regex("1. Acetaminophen\n • Dosage: 500mg")
    assert "acetaminophen" in names


def test_regex_suffix():
    names = _extract_regex("She takes Sertraline for depression.")
    assert "sertraline" in names


def test_regex_empty():
    assert _extract_regex("") == []


# ---------------------------------------------------------------------------
# Graceful degradation when spaCy model is unavailable
# ---------------------------------------------------------------------------


def test_extract_drug_names_works_without_spacy(monkeypatch):
    """If spaCy fails to load, regex results should still be returned."""
    import app.services.drug_extractor as de

    # Simulate a failed spaCy load: model is None and load was already attempted.
    monkeypatch.setattr(de, "_nlp", None)
    monkeypatch.setattr(de, "_nlp_load_attempted", True)

    names = extract_drug_names("Sertraline 50mg daily\nLisinopril 10mg once daily")
    assert "sertraline" in names
    assert "lisinopril" in names


# ---------------------------------------------------------------------------
# extract_prescription_entries — structured bullet-format extraction (Fix 1)
# ---------------------------------------------------------------------------

_BULLET_RX = (
    "1. Warfarin\n"
    "• Dosage: 5mg\n"
    "• Frequency: Once daily\n"
    "• Duration: 90 days\n"
    "• Instructions: Take at same time daily; avoid NSAIDs\n"
    "2. Furosemide\n"
    "• Dosage: 40mg\n"
    "• Frequency: Once daily in the morning\n"
    "• Duration: 90 days\n"
    "• Instructions: Monitor weight daily\n"
)


def test_extract_entries_finds_both_drugs():
    entries, tier = extract_prescription_entries(_BULLET_RX)
    assert tier == "regex"
    assert len(entries) == 2
    assert entries[0].drug_name == "warfarin"
    assert entries[1].drug_name == "furosemide"


def test_extract_entries_full_fields_first_drug():
    entries, _tier = extract_prescription_entries(_BULLET_RX)
    assert entries[0].dosage == "5mg"
    assert entries[0].frequency == "Once daily"
    assert entries[0].duration == "90 days"
    assert entries[0].instructions == "Take at same time daily; avoid NSAIDs"


def test_extract_entries_full_fields_second_drug():
    entries, _tier = extract_prescription_entries(_BULLET_RX)
    assert entries[1].dosage == "40mg"
    assert entries[1].frequency == "Once daily in the morning"
    assert entries[1].duration == "90 days"
    assert entries[1].instructions == "Monitor weight daily"


def test_extract_entries_five_drugs_polypharmacy():
    text = (
        "1. Warfarin\n• Dosage: 5mg\n• Frequency: Once daily\n"
        "• Duration: 90 days\n• Instructions: Avoid NSAIDs\n"
        "2. Furosemide\n• Dosage: 40mg\n• Frequency: Once daily in the morning\n"
        "• Duration: 90 days\n• Instructions: Monitor weight daily\n"
        "3. Carvedilol\n• Dosage: 6.25mg\n• Frequency: Twice daily\n"
        "• Duration: 90 days\n• Instructions: Take with food\n"
        "4. Spironolactone\n• Dosage: 25mg\n• Frequency: Once daily\n"
        "• Duration: 90 days\n• Instructions: Monitor potassium levels\n"
        "5. Digoxin\n• Dosage: 0.125mg\n• Frequency: Once daily\n"
        "• Duration: 90 days\n• Instructions: Report nausea immediately\n"
    )
    entries, tier = extract_prescription_entries(text)
    assert tier == "regex"
    assert len(entries) == 5
    assert [e.drug_name for e in entries] == [
        "warfarin",
        "furosemide",
        "carvedilol",
        "spironolactone",
        "digoxin",
    ]
    assert all(e.dosage is not None for e in entries)
    assert all(e.frequency is not None for e in entries)


def test_extract_entries_falls_back_to_name_only_for_non_bullet():
    text = "Patient takes Sertraline 50mg daily and Zolpidem 10mg at bedtime."
    entries, tier = extract_prescription_entries(text)
    assert tier == "ner"
    names = [e.drug_name for e in entries]
    assert "sertraline" in names


def test_extract_entries_missing_fields_are_none():
    text = "1. Digoxin\n• Dosage: 0.125mg\n"
    entries, _tier = extract_prescription_entries(text)
    assert entries[0].drug_name == "digoxin"
    assert entries[0].dosage == "0.125mg"
    assert entries[0].frequency is None
    assert entries[0].duration is None
    assert entries[0].instructions is None


def test_extract_entries_lowercases_drug_name():
    text = "1. Atorvastatin\n• Dosage: 40mg\n"
    entries, _tier = extract_prescription_entries(text)
    assert entries[0].drug_name == "atorvastatin"


# ---------------------------------------------------------------------------
# Extended _SUFFIX_RE — covers drugs missed before Fix 3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Levothyroxine 75mcg daily", "levothyroxine"),
        ("Furosemide 40mg once daily", "furosemide"),
        ("Warfarin 5mg once daily", "warfarin"),
        ("Spironolactone 25mg daily", "spironolactone"),
        ("Zolpidem 10mg at bedtime", "zolpidem"),
    ],
)
def test_suffix_re_extended_drugs(text: str, expected: str):
    assert expected in extract_drug_names(text)
