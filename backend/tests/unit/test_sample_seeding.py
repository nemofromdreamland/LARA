import json

import respx

import app.services.session_store as _ss
from app.services.dailymed import (
    _normalize_drug_name,
    fetch_leaflet_sections,
)
from app.services.samples import (
    load_manifest,
    sample_pdf_path,
    seed_sample_leaflet_cache,
)


def _all_manifest_drugs() -> list[str]:
    return [drug for entry in load_manifest().values() for drug in entry["drugs"]]


def test_manifest_lists_three_samples_with_pdfs():
    manifest = load_manifest()
    assert len(manifest) == 3
    for sample_id, entry in manifest.items():
        assert entry["label"]
        assert entry["description"]
        assert entry["drugs"]
        assert sample_pdf_path(sample_id).is_file()


async def test_seed_all_writes_one_key_per_manifest_drug():
    drugs = _all_manifest_drugs()
    seeded = await seed_sample_leaflet_cache()
    assert seeded == len(drugs) == 9

    for drug in drugs:
        raw = await _ss._redis.get(f"dailymed:{_normalize_drug_name(drug)}")
        assert raw is not None, f"missing cache key for {drug}"
        sections = json.loads(raw)
        assert isinstance(sections, list) and sections
        for item in sections:
            assert set(item) == {"drug_name", "section", "text"}


async def test_seed_filtered_by_drugs_seeds_only_those():
    seeded = await seed_sample_leaflet_cache(drugs=["Warfarin 5mg"])
    assert seeded == 1
    assert await _ss._redis.get("dailymed:warfarin") is not None
    assert await _ss._redis.get("dailymed:sertraline") is None


async def test_seed_with_unavailable_redis_returns_zero_without_raising():
    _ss._redis = None
    assert await seed_sample_leaflet_cache() == 0


@respx.mock  # no routes registered: any HTTP request would fail the test
async def test_seeded_fetch_is_served_offline():
    await seed_sample_leaflet_cache(drugs=["Sertraline 50mg"])
    sections = await fetch_leaflet_sections("Sertraline 50mg")
    assert sections
    assert all(s.section and s.text for s in sections)
