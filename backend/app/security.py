"""Shared prompt-injection defenses for untrusted text.

Holds the ASCII-folded denylist used by both prescription ingestion and chat:
- ``sanitize_prescription_text`` (in prescription_parser) quarantines — a hit
  there can reject the whole upload.
- ``sanitize_chat_input`` (here) only cleans — it drops suspicious lines and
  keeps the rest, since the pre-retrieval classifier already routes obvious
  injection away and this is defense-in-depth before embed/LLM.
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_MAX_TEXT_LENGTH = 8000

# Denylist: patterns that indicate prompt-injection attempts.
# Matched against an ASCII-normalised copy of each line so Unicode lookalikes
# (e.g. Cyrillic 'а' ≈ Latin 'a') don't bypass the check.
# "ignore" is narrowed to require an injection keyword so that legitimate
# clinical instructions ("do not ignore blurred vision") are not flagged.
_INJECTION_PATTERNS = re.compile(
    r"\bignore\s+(?:previous|prior|above|all|the\s+above|instructions?|prompts?)\b"
    r"|\bforget\s+(?:previous|prior|above|all|instructions?|everything)\b"
    r"|\bdisregard\s+(?:previous|prior|above|all|instructions?)\b"
    r"|\boverride\s+(?:previous|prior|above|all|instructions?|settings?)\b"
    r"|\bbypass\s+(?:previous|prior|above|all|instructions?|filters?|restrictions?)\b"
    r"|system\s*:"
    r"|assistant\s*:"
    r"|human\s*:"
    r"|user\s*:"
    r"|\bprompt\s*:"
    r"|\bnew\s+instruction"
    r"|\bact\s+as\b"
    r"|\bpretend\s+(?:to\s+be|you\s+are)\b"
    r"|\bdo\s+not\s+follow\b"
    r"|\bdo\s+not\s+obey\b"
    r"|<\|"
    r"|</?(?:s|system|user|assistant|inst|instruction)>"
    r"|^\[/?INST\]"
    r"|\{\{.*?\}\}"
    r"|%7[Bb]%7[Bb]",  # URL-encoded {{ }}
    re.IGNORECASE | re.MULTILINE,
)


def _ascii_fold(text: str) -> str:
    """NFKD-normalise + strip non-ASCII so Unicode lookalikes match ASCII patterns."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def sanitize_chat_input(text: str) -> str:
    """Drop injection-pattern lines from a chat question; return the cleaned text.

    Defense-in-depth after the classifier gate. Unlike sanitize_prescription_text
    this NEVER rejects — it only removes suspicious lines (matched on an
    ASCII-folded copy of each line) and logs each drop, then truncates to a sane
    maximum length.
    """
    clean_lines: list[str] = []
    for line in text.splitlines():
        if _INJECTION_PATTERNS.search(_ascii_fold(line)):
            logger.warning(
                "Dropped suspicious injection pattern in chat input: %r", line
            )
        else:
            clean_lines.append(line)
    return "\n".join(clean_lines)[:_MAX_TEXT_LENGTH]
