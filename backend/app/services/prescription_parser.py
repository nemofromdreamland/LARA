import json
import logging
import re

from app.models.schemas import PrescriptionEntry
from app.services.drug_extractor import extract_drug_names
from app.services.llm_client import extract_medications

logger = logging.getLogger(__name__)

# Matches optional ```json ... ``` or ``` ... ``` fences that some LLMs add.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_markdown(text: str) -> str:
    """Remove markdown code fences if the LLM wrapped its JSON response in them."""
    return _FENCE_RE.sub("", text).strip()


async def parse_prescription(text: str) -> list[PrescriptionEntry]:
    """Extract structured medication entries from raw prescription text.

    Primary path:
      1. Call the LLM with EXTRACTION_SYSTEM_PROMPT.
      2. Parse the returned JSON array into PrescriptionEntry objects.

    Fallback (LLM unavailable or JSON parse error):
      Use the regex/spaCy drug_extractor to get drug names only
      (no dosage/frequency/duration/instructions).

    Returns an empty list only when no drugs can be found by either method.
    """
    try:
        raw_json = await extract_medications(text)
        logger.debug("LLM extraction raw response: %s", raw_json[:200])
        clean_json = _strip_markdown(raw_json)
        items: list[dict] = json.loads(clean_json)
        entries = [
            PrescriptionEntry(
                drug_name=item["drug_name"].lower().strip(),
                dosage=item.get("dosage"),
                frequency=item.get("frequency"),
                duration=item.get("duration"),
                instructions=item.get("instructions"),
            )
            for item in items
            if item.get("drug_name")
        ]
        if entries:
            logger.info(
                "LLM extraction succeeded: %s",
                [e.drug_name for e in entries],
            )
            return entries
        logger.error(
            "LLM extraction returned empty list — activating regex fallback. "
            "Check that the LLM is receiving the prescription text correctly."
        )
    except Exception as exc:
        logger.error(
            "LLM extraction failed (%s: %s) — activating regex fallback. "
            "Verify GROQ_API_KEY / CEREBRAS_API_KEY is set in .env.",
            type(exc).__name__,
            exc,
        )

    # Fallback: regex/spaCy pipeline (drug names only, no structured fields)
    names = extract_drug_names(text)
    return [PrescriptionEntry(drug_name=name) for name in names]
