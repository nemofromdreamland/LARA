"""Bundled sample prescriptions: manifest access and leaflet cache seeding.

Sample assets live in app/samples/ — a manifest, one PDF per sample, and
pre-fetched DailyMed leaflet JSON under leaflets/, stored in the exact
shape dailymed._cache_set writes so seeding is a plain setex.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

from app.config import settings
from app.services.dailymed import _CACHE_PREFIX, _get_redis, _normalize_drug_name

logger = logging.getLogger(__name__)

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"
LEAFLETS_DIR = SAMPLES_DIR / "leaflets"


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, dict]:
    """Return the sample manifest as {sample_id: entry}."""
    raw = json.loads((SAMPLES_DIR / "manifest.json").read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in raw["samples"]}


def sample_pdf_path(sample_id: str) -> Path:
    return SAMPLES_DIR / f"{sample_id}.pdf"


async def seed_sample_leaflet_cache(drugs: list[str] | None = None) -> int:
    """Seed the DailyMed Redis cache from the bundled leaflet fixtures.

    Seeds every fixture by default, or only those matching *drugs* when
    given. Overwrites unconditionally (idempotent; refreshes the TTL).
    Failure-tolerant: a missing/broken fixture or an unavailable Redis is
    logged and skipped — ingestion then self-heals via the live fetch path.
    Returns the number of leaflets seeded.
    """
    if drugs is None:
        paths = sorted(LEAFLETS_DIR.glob("*.json"))
    else:
        wanted = {_normalize_drug_name(d) for d in drugs}
        paths = [LEAFLETS_DIR / f"{name}.json" for name in sorted(wanted)]

    seeded = 0
    for path in paths:
        try:
            payload = path.read_text(encoding="utf-8")
            json.loads(payload)  # refuse to seed corrupt fixtures
            r = _get_redis()
            await r.setex(
                f"{_CACHE_PREFIX}{path.stem}",
                settings.dailymed_cache_ttl_seconds,
                payload,
            )
            seeded += 1
        except Exception as exc:
            logger.warning("sample leaflet seed failed for %s: %s", path.name, exc)
    return seeded
