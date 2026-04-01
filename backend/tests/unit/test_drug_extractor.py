import pytest

from app.services.drug_extractor import (
    _extract_regex,
    _extract_spacy,
    extract_drug_names,
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


def test_spacy_extracts_inline_drugs():
    names = _extract_spacy("The patient takes Omeprazole, Amoxicillin and Sertraline.")
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

    monkeypatch.setattr(de, "_nlp", False)  # sentinel: simulate missing model

    names = extract_drug_names("Sertraline 50mg daily\nLisinopril 10mg once daily")
    assert "sertraline" in names
    assert "lisinopril" in names
