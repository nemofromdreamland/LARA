from app.services.chunker import chunk_text
from app.services.dailymed import fetch_leaflet_sections


async def process_drug(drug: str, session_id: str) -> tuple[str, list[str], list[dict]]:
    """Fetch leaflet sections for one drug and produce chunks + metadata.

    Returns (drug, chunks, metas). Both lists are empty when no leaflet is found.
    """
    sections = await fetch_leaflet_sections(drug)
    if not sections:
        return drug, [], []
    chunks: list[str] = []
    metas: list[dict] = []
    for section in sections:
        for chunk in chunk_text(section.text):
            chunks.append(chunk)
            metas.append(
                {
                    "session_id": session_id,
                    "drug_name": drug,
                    "section": section.section,
                }
            )
    return drug, chunks, metas
