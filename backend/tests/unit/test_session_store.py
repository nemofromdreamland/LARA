import time
from unittest.mock import patch

from app.models.schemas import PrescriptionEntry
from app.services.session_store import (
    expire_sessions,
    get_prescription,
    get_prescription_entries,
    get_upload_result,
    save_prescription,
    save_prescription_entries,
    save_upload_result,
)


def _fresh_state():
    import app.services.session_store as ss

    ss._sessions.clear()


def test_save_and_get_prescription():
    _fresh_state()
    save_prescription("s1", "Drug A 50mg")
    assert get_prescription("s1") == "Drug A 50mg"


def test_get_prescription_unknown_session():
    _fresh_state()
    assert get_prescription("nonexistent") is None


def test_save_and_get_upload_result():
    _fresh_state()
    save_prescription("s2", "text")
    save_upload_result("s2", ["lisinopril"], ["tylenol"])
    found, missing = get_upload_result("s2")
    assert found == ["lisinopril"]
    assert missing == ["tylenol"]


def test_get_upload_result_unknown_session():
    _fresh_state()
    found, missing = get_upload_result("ghost")
    assert found == []
    assert missing == []


def test_upload_result_defaults_to_empty_lists():
    _fresh_state()
    save_prescription("s3", "text")
    found, missing = get_upload_result("s3")
    assert found == []
    assert missing == []


def test_expire_sessions_removes_old_entry():
    _fresh_state()
    save_prescription("old", "text")

    future = time.monotonic() + 10_800
    with patch("app.services.session_store.time.monotonic", return_value=future):
        expired = expire_sessions(ttl_seconds=7200)

    assert "old" in expired
    assert get_prescription("old") is None


def test_expire_sessions_keeps_fresh_entry():
    _fresh_state()
    save_prescription("fresh", "text")
    expired = expire_sessions(ttl_seconds=7200)
    assert "fresh" not in expired
    assert get_prescription("fresh") == "text"


def test_expire_sessions_returns_only_expired():
    _fresh_state()
    save_prescription("keep", "text")
    save_prescription("evict", "text")

    future = time.monotonic() + 10_800
    with patch("app.services.session_store.time.monotonic", return_value=future):
        expired = expire_sessions(ttl_seconds=7200)

    assert set(expired) == {"keep", "evict"}


def test_expire_sessions_empty_store():
    _fresh_state()
    assert expire_sessions(ttl_seconds=7200) == []


def test_save_and_get_prescription_entries():
    _fresh_state()
    save_prescription("s5", "text")
    entries = [
        PrescriptionEntry(drug_name="ibuprofen", dosage="400mg", frequency="TID"),
        PrescriptionEntry(drug_name="azithromycin", dosage="500mg"),
    ]
    save_prescription_entries("s5", entries)
    result = get_prescription_entries("s5")
    assert len(result) == 2
    assert result[0].drug_name == "ibuprofen"
    assert result[0].dosage == "400mg"
    assert result[1].drug_name == "azithromycin"
    assert result[1].frequency is None


def test_get_prescription_entries_returns_empty_for_missing_session():
    _fresh_state()
    assert get_prescription_entries("nonexistent") == []
