import logging
import re

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
    r"|olol|oxin|azole|mycin|cycline|cillin|dronate"
    r"|amine|idine|tidine|zepam|zolam|oxetine|prazole"
    r"|raline|dipine|triptan|setron|pram|phen|nophen))\b"
)

# Numbered list items: "1. Acetaminophen"
_NUMBERED_ITEM_RE = re.compile(
    r"^\d+\.\s+([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})?)\s*$",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# spaCy — lazy-loaded so startup is not blocked if the model is missing
# ---------------------------------------------------------------------------

# Entity labels that are purely numeric/temporal — never drug names.
_SKIP_ENT_LABELS = {
    "DATE",
    "TIME",
    "CARDINAL",
    "ORDINAL",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "LANGUAGE",
}

# Common words that appear capitalized in prescriptions but are not drugs.
_STOPWORDS: set[str] = {
    "patient",
    "doctor",
    "hospital",
    "pharmacy",
    "clinic",
    "institute",
    "date",
    "name",
    "address",
    "phone",
    "fax",
    "email",
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
    "general",
    "medical",
    "health",
    "care",
    "center",
    "service",
    "rx",
    "sig",
    "refills",
    "qty",
    "dispense",
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
    # common patient/doctor name parts
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
}

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy

            _nlp = spacy.load("en_core_web_sm")
            logger.debug("spaCy en_core_web_sm loaded")
        except Exception as exc:
            logger.warning("spaCy model unavailable (%s) — using regex only", exc)
            _nlp = False  # sentinel: don't retry
    return _nlp if _nlp is not False else None


def _extract_spacy(text: str) -> list[str]:
    """Return lowercased drug name candidates using spaCy NER + POS tagging."""
    nlp = _get_nlp()
    if nlp is None:
        return []

    doc = nlp(text)

    # Collect token indices that belong to a non-numeric named entity.
    ent_token_ids: set[int] = set()
    for ent in doc.ents:
        if ent.label_ not in _SKIP_ENT_LABELS:
            for tok in ent:
                ent_token_ids.add(tok.i)

    results: list[str] = []
    for tok in doc:
        is_ne = tok.i in ent_token_ids
        is_propn = tok.pos_ == "PROPN"
        if not (is_ne or is_propn):
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
    1. spaCy NER + POS — catches names with no known suffix and unusual
       formats; falls back gracefully if the model is not installed.
    2. Regex heuristics — RX-line dosage pattern, numbered-list pattern,
       pharmaceutical suffix pattern; supplements spaCy for names tagged
       with generic POS labels (NOUN/ADV) and no entity span.
    """
    candidates = _extract_spacy(text) + _extract_regex(text)
    seen: set[str] = set()
    result: list[str] = []
    for name in candidates:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result
