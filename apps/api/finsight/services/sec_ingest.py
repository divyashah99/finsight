"""SEC ingestion pipeline.

End-to-end:
    ticker → CIK → recent 10-K/10-Q filings → fetch HTML → parse + chunk →
    embed in batches → upsert to Qdrant → record metadata in Postgres

Idempotent: filings already in `sec_docs` (matched by accession number) are
skipped, so calling this repeatedly is safe and cheap.

Runs as a FastAPI BackgroundTask from the `/ingest/sec/{ticker}` endpoint and
nightly from APScheduler for any ticker that has a recent run.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from finsight.db.client import session_scope
from finsight.db.models import SECDoc
from finsight.logging_setup import get_logger
from finsight.services import llm, vectorstore
from finsight.services.chunker import Chunk, chunk_filing
from finsight.tools import edgar

log = get_logger(__name__)

_EMBED_BATCH = 64
_MAX_CHUNKS_PER_FILING = 200  # cap cost; 10-Ks can be huge


async def ingest_ticker(ticker: str, max_filings: int = 2) -> dict[str, Any]:
    """Ingest the most recent N filings (default 2: latest 10-K + latest 10-Q)."""
    ticker = ticker.upper()
    log.info("ingest.start ticker=%s", ticker)
    await vectorstore.ensure_collection()

    cik = await edgar.lookup_cik(ticker)
    if not cik:
        return {"ok": False, "error": f"unknown ticker: {ticker}"}

    filings = await edgar.recent_filings(cik, forms=("10-K", "10-Q"), limit=max_filings)
    if not filings:
        return {"ok": False, "error": "no filings found"}

    ingested: list[dict[str, Any]] = []
    skipped: list[str] = []

    for f in filings:
        async with session_scope() as s:
            existing = (
                await s.execute(select(SECDoc).where(SECDoc.accession_number == f.accession))
            ).scalar_one_or_none()
            if existing:
                skipped.append(f.accession)
                continue

        log.info("ingest.filing ticker=%s form=%s accession=%s", ticker, f.form_type, f.accession)
        html = await edgar.fetch_document(f.url)
        chunks = chunk_filing(html)
        if len(chunks) > _MAX_CHUNKS_PER_FILING:
            log.info("ingest.cap chunks=%d -> %d", len(chunks), _MAX_CHUNKS_PER_FILING)
            chunks = chunks[:_MAX_CHUNKS_PER_FILING]

        if not chunks:
            log.warning("ingest.empty accession=%s", f.accession)
            continue

        count = await _embed_and_store(ticker, cik, f, chunks)

        async with session_scope() as s:
            s.add(
                SECDoc(
                    id=uuid.uuid4(),
                    ticker=ticker,
                    cik=cik,
                    accession_number=f.accession,
                    form_type=f.form_type,
                    filed_date=f.filed_date.replace(tzinfo=timezone.utc),
                    url=f.url,
                    chunk_count=count,
                    bytes_size=len(html),
                    ingested_at=datetime.now(timezone.utc),
                )
            )

        ingested.append(
            {
                "accession": f.accession,
                "form_type": f.form_type,
                "filed_date": f.filed_date.isoformat(),
                "chunks": count,
            }
        )

    log.info("ingest.done ticker=%s ingested=%d skipped=%d", ticker, len(ingested), len(skipped))
    return {"ok": True, "ticker": ticker, "ingested": ingested, "skipped": skipped}


async def _embed_and_store(
    ticker: str, cik: str, filing: edgar.Filing, chunks: list[Chunk]
) -> int:
    total = 0
    for i in range(0, len(chunks), _EMBED_BATCH):
        batch = chunks[i : i + _EMBED_BATCH]
        vectors = await llm.embed([c.text for c in batch])
        points = []
        for c, v in zip(batch, vectors, strict=True):
            payload = {
                "ticker": ticker,
                "cik": cik,
                "accession": filing.accession,
                "form_type": filing.form_type,
                "section": c.section,
                "filed_date": filing.filed_date.date().isoformat(),
                "url": filing.url,
                "chunk_index": c.chunk_index,
                "text": c.text,
            }
            points.append((v, payload))
        total += await vectorstore.upsert_chunks(points)
        await asyncio.sleep(0)  # cooperate with the event loop
    return total
