import re

# Matches a drug name at the start of an Rx line followed by a dosage.
# Example: "Lisinopril 10mg daily" → "Lisinopril"
_RX_LINE_RE = re.compile(
    r"^([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)?)"
    r"\s+\d+\s*(?:mg|mcg|g|ml|units?|IU)\b",
    re.MULTILINE,
)

# Matches capitalised words ending in common pharmaceutical suffixes.
# Catches brand/generic names not captured by the Rx-line pattern.
_SUFFIX_RE = re.compile(
    r"\b([A-Z][a-z]{2,}"
    r"(?:pril|sartan|statin|mab|nib|afil|xaban|parin"
    r"|olol|oxin|azole|mycin|cycline|cillin|dronate"
    r"|amine|idine|tidine|zepam|zolam|oxetine|prazole"
    r"|raline|dipine|triptan|setron|pram))\b"
)


def extract_drug_names(text: str) -> list[str]:
    """Return a deduplicated, lower-cased list of drug names found in *text*.

    Uses two complementary heuristics:
    1. Rx-line pattern — drug name preceding a dose on the same line.
    2. Pharmaceutical suffix pattern — capitalised words with known endings.
    """
    rx_matches = [m.group(1).strip() for m in _RX_LINE_RE.finditer(text)]
    suffix_matches = _SUFFIX_RE.findall(text)

    combined = rx_matches + suffix_matches
    seen: set[str] = set()
    result: list[str] = []
    for name in combined:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result
