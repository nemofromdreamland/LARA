import pytest

from app.services.drug_extractor import extract_drug_names

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


_SAMPLE_NUMBERED = """
1. Acetaminophen
 • Dosage: 500mg
 • Frequency: Every 6 hours as needed
2. Tylenol
 • Dosage: 500mg
 • Frequency: Twice daily
"""


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
    # "nophen" suffix catches Acetaminophen even without numbered list context
    names = extract_drug_names("The patient takes Acetaminophen for pain relief.")
    assert "acetaminophen" in names
