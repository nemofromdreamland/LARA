"""One-off fixture generator for the bundled sample prescriptions.

Fetches the real DailyMed leaflet for every drug listed in
app/samples/manifest.json and writes it to
app/samples/leaflets/<normalized_drug_name>.json in the exact JSON shape
that dailymed._cache_set stores in Redis, so startup seeding is a plain
setex of the file contents.

Run locally (network required) from backend/:

    uv run python scripts/generate_sample_fixtures.py

Not part of the application; re-run only when samples change or fixtures
need refreshing. Redis is not required — the DailyMed cache layer degrades
gracefully when uninitialized.
"""

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.dailymed import (  # noqa: E402
    _normalize_drug_name,
    fetch_leaflet_sections,
)

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "app" / "samples"
LEAFLETS_DIR = SAMPLES_DIR / "leaflets"


async def main() -> int:
    manifest = json.loads((SAMPLES_DIR / "manifest.json").read_text(encoding="utf-8"))
    drugs = {
        drug for sample in manifest["samples"] for drug in sample["drugs"]
    }
    LEAFLETS_DIR.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    for drug in sorted(drugs):
        normalized = _normalize_drug_name(drug)
        print(f"fetching {drug!r} -> {normalized}.json ...")
        sections = await fetch_leaflet_sections(drug)
        if not sections:
            print(f"  ERROR: DailyMed returned no sections for {drug!r}")
            failed.append(drug)
            continue
        payload = json.dumps([asdict(s) for s in sections], indent=2)
        out = LEAFLETS_DIR / f"{normalized}.json"
        out.write_text(payload, encoding="utf-8")
        print(f"  wrote {out.name}: {len(sections)} sections, {len(payload)} bytes")

    if failed:
        print(f"FAILED for: {', '.join(failed)}")
        return 1
    print(f"done: {len(drugs)} leaflet fixtures written to {LEAFLETS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
