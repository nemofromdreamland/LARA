import logging
import re

from app.models.schemas import PrescriptionEntry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns (fallback / supplement to spaCy)
# ---------------------------------------------------------------------------

# Drug name preceding a dosage on the same line: "Lisinopril 10mg daily"
_RX_LINE_RE = re.compile(
    r"^([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)?)"
    r"\s+\d+\s*(?:mg|mcg|g|ml|units?|IU)\b",
    re.MULTILINE,
)

# Capitalised words with common pharmaceutical suffixes.
_SUFFIX_RE = re.compile(
    r"\b([A-Z][a-z]{2,}"
    r"(?:pril|sartan|statin|mab|nib|afil|xaban|parin"
    r"|olol|oxin|oxine|azole|mycin|cycline|cillin|dronate"
    r"|amine|idine|tidine|zepam|zolam|oxetine|prazole"
    r"|raline|dipine|triptan|setron|pram|phen|nophen"
    r"|semide|farin|lactone|pidem))\b"
)

# Numbered list items: "1. Acetaminophen" or "1. Hydroxyzine (NOT Hydralazine)"
# The (?:\([^)]*\))* at the end ignores any trailing parenthetical annotations
# so the capture group still yields only the intended drug name.
_NUMBERED_ITEM_RE = re.compile(
    r"^\d+\.\s+([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})?)\s*(?:\([^)]*\))*\s*$",
    re.MULTILINE,
)

# Bullet field lines: "• Dosage: 5mg"
_BULLET_FIELD_RE = re.compile(
    r"[•·\-]\s*(Dosage|Frequency|Duration|Instructions):\s*(.+)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# spaCy — lazy-loaded so startup is not blocked if the model is missing
# ---------------------------------------------------------------------------

# Entity labels that are definitely NOT drug names.
# We explicitly skip people, organisations, and geographic entities —
# these are the main source of false positives from prescription headers.
_SKIP_ENT_LABELS = {
    "DATE",
    "TIME",
    "CARDINAL",
    "ORDINAL",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "LANGUAGE",
    # Entities that appear in prescription headers and metadata:
    "PERSON",  # Dr. Mitchell, Sarah, patient names
    "ORG",  # Wellness Healthcare, Mitchell Clinic
    "GPE",  # cities, countries — "City" in clinic addresses
    "FAC",  # buildings, facilities
    "LOC",  # geographic locations
    "NORP",  # nationalities, religious or political groups
    "EVENT",  # named events
    "WORK_OF_ART",  # books, titles
    "LAW",  # laws and acts
}

# Words that appear capitalised in prescriptions but are never drug names.
# This covers: column headers, metadata labels, clinic name fragments,
# credentials, instruction words, and common personal names.
_STOPWORDS: set[str] = {
    # --- prescription structure and metadata ---
    "patient",
    "doctor",
    "physician",
    "prescriber",
    "hospital",
    "pharmacy",
    "clinic",
    "institute",
    "institution",
    "date",
    "name",
    "address",
    "phone",
    "fax",
    "email",
    "signature",
    "prescription",
    "prescribed",
    # --- column headers that appear in tabular prescriptions ---
    "frequency",
    "duration",
    "instructions",
    "dosage",
    "quantity",
    "refills",
    "dispense",
    "notes",
    "diagnosis",
    # --- healthcare / clinic name fragments ---
    "wellness",
    "healthcare",
    "medicine",
    "medical",
    "health",
    "care",
    "center",
    "centre",
    "service",
    "services",
    "general",
    "internal",
    "specialty",
    "practice",
    "associates",
    "group",
    "city",
    "community",
    # --- credentials (appear after doctor names) ---
    "facp",
    "facs",
    "frcpc",
    "frcs",
    "mbbs",
    "bchir",
    # --- patient demographics ---
    "gender",
    "male",
    "female",
    "dob",
    "weight",
    "height",
    "allergy",
    "allergies",
    # --- dosing instructions ---
    "daily",
    "twice",
    "once",
    "every",
    "each",
    "take",
    "oral",
    "tablet",
    "capsule",
    "solution",
    "injection",
    "morning",
    "evening",
    "night",
    "food",
    "water",
    # --- administrative ---
    "rx",
    "sig",
    "qty",
    # --- calendar ---
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    # --- common patient / doctor name fragments ---
    "john",
    "jane",
    "mary",
    "james",
    "robert",
    "michael",
    "william",
    "david",
    "richard",
    "thomas",
    "sarah",
    "emily",
    "lisa",
    "karen",
    "patricia",
    "mitchell",
    "smith",
    "jones",
    "brown",
    "davis",
    "wilson",
    "johnson",
    "taylor",
    "anderson",
    "martin",
    "white",
    "harris",
    "clark",
    "lewis",
    "robinson",
    "walker",
    "hall",
    "allen",
    "young",
    "king",
    "wright",
    "scott",
    "green",
    "baker",
    "adams",
    "nelson",
    "carter",
    "hill",
}

_nlp = None
_nlp_load_attempted = False


def _get_nlp():
    global _nlp, _nlp_load_attempted
    if not _nlp_load_attempted:
        _nlp_load_attempted = True
        try:
            import spacy

            _nlp = spacy.load("en_core_web_sm")
            logger.debug("spaCy en_core_web_sm loaded")
        except Exception as exc:
            logger.warning("spaCy model unavailable (%s) — using regex only", exc)
    return _nlp


def _extract_spacy(text: str) -> list[str]:
    """Return lowercased drug name candidates using spaCy NER.

    Only tokens belonging to named entities NOT in _SKIP_ENT_LABELS are
    considered. We intentionally do NOT include bare PROPN tokens, because
    that catches clinic names, doctor names, and column headers like
    'Frequency' and 'Duration' which are tagged as proper nouns by spaCy
    but are definitely not medications.
    """
    nlp = _get_nlp()
    if nlp is None:
        return []

    doc = nlp(text)

    # Collect token indices that belong to a qualifying named entity.
    ent_token_ids: set[int] = set()
    for ent in doc.ents:
        if ent.label_ not in _SKIP_ENT_LABELS:
            for tok in ent:
                ent_token_ids.add(tok.i)

    results: list[str] = []
    for tok in doc:
        # Require membership in a qualifying named entity — PROPN alone is
        # too broad and is the main source of false positives.
        if tok.i not in ent_token_ids:
            continue
        if not tok.is_alpha or len(tok.text) < 4:
            continue
        if not tok.text[0].isupper():
            continue
        word = tok.text.lower()
        if word not in _STOPWORDS:
            results.append(word)

    return results


def _extract_regex(text: str) -> list[str]:
    """Return lowercased drug name candidates using regex heuristics."""
    rx = [m.group(1).strip().lower() for m in _RX_LINE_RE.finditer(text)]
    numbered = [m.group(1).strip().lower() for m in _NUMBERED_ITEM_RE.finditer(text)]
    suffix = [m.lower() for m in _SUFFIX_RE.findall(text)]
    return rx + numbered + suffix


def extract_drug_names(text: str) -> list[str]:
    """Return a deduplicated, lowercase list of drug names found in *text*.

    Strategy (union of two complementary methods):
    1. spaCy NER — catches drug names tagged as named entities (PRODUCT, etc.);
       explicitly excludes PERSON, ORG, GPE, and other non-drug entity types.
    2. Regex heuristics — RX-line dosage pattern, numbered-list pattern,
       pharmaceutical suffix pattern; these are the primary reliable source
       when the LLM extraction fallback is active.
    """
    candidates = _extract_spacy(text) + _extract_regex(text)
    seen: set[str] = set()
    result: list[str] = []
    for name in candidates:
        name = name.strip()
        if name and name not in seen and name not in _STOPWORDS:
            seen.add(name)
            result.append(name)
    return result


def extract_prescription_entries(
    text: str,
) -> tuple[list[PrescriptionEntry], str]:
    """Extract full PrescriptionEntry objects from bullet-format prescriptions.

    Walks the text line-by-line. Each numbered drug line (matched by
    _NUMBERED_ITEM_RE) starts a new entry; subsequent bullet lines
    (• Dosage: / • Frequency: / • Duration: / • Instructions:) are
    consumed until the next numbered item or end of text.

    Falls back to name-only extraction (extract_drug_names) when no
    numbered items are found, preserving the previous behaviour for
    non-bullet prescription formats.

    Returns (entries, tier) where tier is "regex" when numbered items were
    matched or "ner" when the unstructured name extractor was used.
    """
    lines = text.splitlines()
    entries: list[PrescriptionEntry] = []
    i = 0
    while i < len(lines):
        m = _NUMBERED_ITEM_RE.match(lines[i].strip())
        if m:
            drug = m.group(1).strip().lower()
            fields: dict[str, str | None] = {
                "dosage": None,
                "frequency": None,
                "duration": None,
                "instructions": None,
            }
            j = i + 1
            while j < len(lines):
                bf = _BULLET_FIELD_RE.match(lines[j].strip())
                if bf:
                    key = bf.group(1).lower()
                    val = bf.group(2).strip()
                    if key in fields:
                        fields[key] = val
                    j += 1
                elif _NUMBERED_ITEM_RE.match(lines[j].strip()):
                    break
                else:
                    j += 1
            entries.append(PrescriptionEntry(drug_name=drug, **fields))
            i = j
        else:
            i += 1

    if not entries:
        logger.warning(
            "extract_prescription_entries: no numbered drug items matched — "
            "falling back to unstructured name extraction. "
            "Check for non-standard prescription formatting."
        )
        return [PrescriptionEntry(drug_name=n) for n in extract_drug_names(text)], "ner"
    return entries, "regex"
