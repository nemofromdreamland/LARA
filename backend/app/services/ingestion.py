import asyncio
import logging

from app.services.chunker import chunk_text
from app.services.dailymed import fetch_leaflet_sections
from app.services.embedder import embed
from app.services.prescription_parser import parse_prescription
from app.services.session_store import (
    save_job_status,
    save_prescription,
    save_prescription_entries,
    save_upload_result,
)
from app.services.vector_store import delete_session, store

logger = logging.getLogger(__name__)


async def process_drug(drug: str) -> tuple[str, list[str], list[dict]]:
    """Fetch leaflet sections for one drug and produce chunks + metadata.

    Returns (drug, chunks, metas). Both lists are empty when no leaflet is found.
    session_id is NOT in metas — it is implicit in the ChromaDB collection name.
    """
    sections = await fetch_leaflet_sections(drug)
    if not sections:
        return drug, [], []
    chunks: list[str] = []
    metas: list[dict] = []
    for section in sections:
        for chunk in chunk_text(section.text):
            chunks.append(chunk)
            metas.append({"drug_name": drug, "section": section.section})
    return drug, chunks, metas


async def run_ingestion(
    job_id: str,
    session_id: str,
    text: str,
    rid: str,
    embed_executor,
) -> None:
    """Parse prescription → fetch leaflets → embed → store.

    Unexpected/infra errors propagate to the queue consumer (ingestion_queue),
    which owns retry and dead-lettering — this function no longer swallows them.
    Terminal logical outcomes (no drugs found, per-drug missing leaflets) are
    handled here and return normally so the message is ACKed, not retried.
    """
    entries = await parse_prescription(text, session_id=session_id)
    if not entries:
        await save_job_status(
            job_id,
            session_id,
            "failed",
            error="No drug names found in prescription.",
        )
        return

    await save_prescription(session_id, text)
    await save_prescription_entries(session_id, entries)

    drug_names = [e.drug_name for e in entries]

    results = await asyncio.gather(
        *[process_drug(drug) for drug in drug_names],
        return_exceptions=True,
    )

    # Idempotency: at-least-once delivery may redeliver this job (e.g. a worker
    # crashed after storing some drugs but before ACK). store() uses random
    # UUID ids, so replaying would double-store. Clear the session's collection
    # before (re)populating it. This also matches the existing replace semantics
    # of save_prescription/save_upload_result (a new upload replaces, not appends).
    await delete_session(session_id)

    stored_drugs: list[str] = []
    missing_drugs: list[str] = []

    for drug, result in zip(drug_names, results):
        if isinstance(result, Exception):
            logger.warning(
                "DailyMed fetch failed for %s: %s",
                drug,
                result,
                extra={"request_id": rid},
            )
            missing_drugs.append(drug)
            continue
        _, chunks, metas = result
        if not chunks:
            logger.warning(
                "No DailyMed leaflet found for drug: %s",
                drug,
                extra={"request_id": rid},
            )
            missing_drugs.append(drug)
            continue
        try:
            embeddings = await embed(chunks, embed_executor)
            await store(chunks, embeddings, metas, session_id=session_id)
            stored_drugs.append(drug)
        except Exception as store_exc:
            logger.warning(
                "Failed to embed/store chunks for %s: %s",
                drug,
                store_exc,
                extra={"request_id": rid},
            )
            missing_drugs.append(drug)

    await save_upload_result(session_id, stored_drugs, missing_drugs)
    await save_job_status(
        job_id,
        session_id,
        "done",
        drugs_found=stored_drugs,
        missing_leaflets=missing_drugs,
    )

    logger.info(
        "ingestion done: %d stored, %d missing",
        len(stored_drugs),
        len(missing_drugs),
        extra={"request_id": rid},
    )
