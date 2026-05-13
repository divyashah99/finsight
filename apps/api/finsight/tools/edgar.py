"""SEC EDGAR client.

EDGAR is free and unauthenticated but requires a `User-Agent` header that
identifies you. We use the JSON endpoints (preferred over scraping HTML index
pages) to find recent filings, then download the document HTML for parsing.

Two public functions:
- `lookup_cik(ticker)`           — ticker → 10-digit zero-padded CIK
- `recent_filings(cik, forms)`   — list metadata for last N filings of given forms
- `fetch_document(url)`          — raw HTML of a primary filing document
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import httpx

from finsight.logging_setup import get_logger
from finsight.settings import settings

log = get_logger(__name__)

_HEADERS = {"User-Agent": settings.sec_user_agent, "Accept-Encoding": "gzip, deflate"}
_TICKER_INDEX_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_TIMEOUT = httpx.Timeout(20.0)


@dataclass
class Filing:
    cik: str
    accession: str
    form_type: str
    filed_date: datetime
    primary_document: str
    url: str


_ticker_cache: dict[str, str] = {}


async def lookup_cik(ticker: str) -> str | None:
    """Resolve a ticker to its 10-digit CIK (zero-padded)."""
    global _ticker_cache
    if not _ticker_cache:
        async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT) as c:
            res = await c.get(_TICKER_INDEX_URL)
            res.raise_for_status()
            data = res.json()
        # data is a dict keyed by integer indices
        for row in data.values():
            _ticker_cache[row["ticker"].upper()] = str(row["cik_str"]).zfill(10)

    return _ticker_cache.get(ticker.upper())


async def recent_filings(cik: str, forms: Iterable[str] = ("10-K", "10-Q"), limit: int = 4) -> list[Filing]:
    """Return the most recent `limit` filings of the requested form types."""
    url = _SUBMISSIONS_URL.format(cik=cik)
    async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT) as c:
        res = await c.get(url)
        res.raise_for_status()
        data = res.json()

    filings = data.get("filings", {}).get("recent", {})
    accessions = filings.get("accessionNumber", [])
    forms_arr = filings.get("form", [])
    dates_arr = filings.get("filingDate", [])
    docs_arr = filings.get("primaryDocument", [])

    want = set(forms)
    out: list[Filing] = []
    for i, form in enumerate(forms_arr):
        if form not in want:
            continue
        accession = accessions[i]
        primary = docs_arr[i]
        no_dash = accession.replace("-", "")
        doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{no_dash}/{primary}"
        out.append(
            Filing(
                cik=cik,
                accession=accession,
                form_type=form,
                filed_date=datetime.fromisoformat(dates_arr[i]),
                primary_document=primary,
                url=doc_url,
            )
        )
        if len(out) >= limit:
            break
    log.info("edgar.filings cik=%s n=%d", cik, len(out))
    return out


async def fetch_document(url: str) -> str:
    async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT) as c:
        res = await c.get(url)
        res.raise_for_status()
        return res.text
