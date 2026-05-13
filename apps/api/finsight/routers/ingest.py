"""POST /ingest/sec/{ticker} — kick off SEC ingestion as a background task.

Returns 202 immediately. Clients poll `/ingest/sec/{ticker}/status` (cheap —
just counts rows in `sec_docs`) to know when retrieval will be useful.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException
from sqlalchemy import select

from finsight.db.client import session_scope
from finsight.db.models import SECDoc
from finsight.logging_setup import get_logger
from finsight.services import sec_ingest, vectorstore

log = get_logger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/sec/{ticker}", status_code=202)
async def ingest_sec(ticker: str, bg: BackgroundTasks) -> dict:
    ticker = ticker.upper().strip()
    if not ticker.isalnum() or len(ticker) > 8:
        raise HTTPException(status_code=400, detail="invalid ticker")
    bg.add_task(sec_ingest.ingest_ticker, ticker)
    return {"status": "queued", "ticker": ticker}


@router.get("/sec/{ticker}/status")
async def ingest_status(ticker: str) -> dict:
    ticker = ticker.upper().strip()
    async with session_scope() as s:
        rows = (
            await s.execute(
                select(SECDoc).where(SECDoc.ticker == ticker).order_by(SECDoc.filed_date.desc())
            )
        ).scalars().all()
    chunk_count = await vectorstore.count_for_ticker(ticker)
    return {
        "ticker": ticker,
        "filings": [
            {
                "accession": r.accession_number,
                "form_type": r.form_type,
                "filed_date": r.filed_date.isoformat(),
                "chunk_count": r.chunk_count,
                "ingested_at": r.ingested_at.isoformat(),
            }
            for r in rows
        ],
        "qdrant_chunks": chunk_count,
    }
