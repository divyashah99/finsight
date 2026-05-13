"""SEC agent — RAG over ingested 10-K / 10-Q chunks.

Strategy: fire three semantic queries against Qdrant — one each targeting risk
factors, MD&A, and recent guidance — then dedupe by chunk_index/accession and
return the top citations. This gives the writer a balanced evidence base
instead of just the closest matches to one query.

If no chunks exist yet for the ticker, we trigger an on-demand ingestion and
return empty evidence (the run still completes; the memo notes missing SEC
context).
"""

from __future__ import annotations

import asyncio
from typing import Any

from finsight.agents.state import AgentError, ResearchState, SECCitation, SECEvidence
from finsight.logging_setup import get_logger
from finsight.services import llm, sec_ingest, vectorstore

log = get_logger(__name__)

QUERIES = [
    ("risk_factors", "What are the most material business and operational risk factors?"),
    ("mdna", "Management's discussion of revenue drivers, margins, and forward outlook"),
    ("market_risk", "Quantitative market risks, interest rate exposure, FX exposure"),
]

TOP_K_PER_QUERY = 4
MAX_CITATIONS = 8


async def run(state: ResearchState) -> dict[str, Any]:
    ticker = state["ticker"]
    log.info("sec.start ticker=%s", ticker)

    available = await vectorstore.count_for_ticker(ticker)
    if available == 0:
        log.info("sec.no_chunks triggering_ingest ticker=%s", ticker)
        try:
            await sec_ingest.ingest_ticker(ticker, max_filings=1)
        except Exception as e:  # noqa: BLE001
            return {
                "sec": SECEvidence(),
                "errors": [AgentError(agent="sec", error=f"ingest_failed: {e}")],
            }
        available = await vectorstore.count_for_ticker(ticker)
        if available == 0:
            return {
                "sec": SECEvidence(),
                "errors": [AgentError(agent="sec", error="no chunks after ingest")],
            }

    # Embed all 3 queries in one OpenAI call (batched)
    embeddings = await llm.embed([q for _, q in QUERIES])

    async def _search(emb: list[float], section_hint: str):
        return await vectorstore.search(
            embedding=emb,
            ticker=ticker,
            limit=TOP_K_PER_QUERY,
            sections=[section_hint] if section_hint in {"risk_factors", "mdna", "market_risk"} else None,
        )

    hits_per_query = await asyncio.gather(
        *[_search(emb, section_hint) for emb, (section_hint, _) in zip(embeddings, QUERIES, strict=True)]
    )

    # Dedupe by (accession, chunk_index), keep best score
    best: dict[tuple[str, int], tuple[float, dict[str, Any]]] = {}
    for hits in hits_per_query:
        for h in hits:
            key = (h.payload.get("accession", "?"), h.payload.get("chunk_index", -1))
            existing = best.get(key)
            if existing is None or h.score > existing[0]:
                best[key] = (h.score, h.payload)

    ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)[:MAX_CITATIONS]

    citations = [
        SECCitation(
            accession=p.get("accession", ""),
            form_type=p.get("form_type", ""),
            section=p.get("section"),
            filed_date=p.get("filed_date"),
            url=p.get("url"),
            excerpt=(p.get("text") or "")[:600],
        )
        for _, p in ranked
    ]

    risk_excerpts = " ".join(c.excerpt for c in citations if c.section == "risk_factors")[:1200]
    mdna_excerpts = " ".join(c.excerpt for c in citations if c.section == "mdna")[:1200]

    evidence = SECEvidence(
        citations=citations,
        risk_summary=risk_excerpts or None,
        mdna_summary=mdna_excerpts or None,
    )
    log.info("sec.done ticker=%s citations=%d", ticker, len(citations))
    return {"sec": evidence}
