"""Alpha Vantage HTTP client.

Plain async HTTP — caching/rate-limiting/retry are added by the decorators in
`tools.base` so the MCP server can apply them uniformly to every endpoint.

We expose four endpoints; each returns a JSON-serializable dict. The MCP server
re-exports these as MCP tools.
"""

from __future__ import annotations

from typing import Any

import httpx

from finsight.logging_setup import get_logger
from finsight.settings import settings
from finsight.tools.base import ToolResult, cached, rate_limited, with_retry

log = get_logger(__name__)

_BASE_URL = "https://www.alphavantage.co/query"
_PROVIDER = "alpha_vantage"
_PER_MIN = settings.alphavantage_rate_per_min
_PER_DAY = settings.alphavantage_rate_per_day


async def _request(params: dict[str, Any]) -> ToolResult:
    """Single transport call. Alpha Vantage uses HTTP 200 even on auth/quota
    errors and signals them inside the body — we have to inspect the JSON.
    """
    qp = {**params, "apikey": settings.alphavantage_api_key}
    try:
        async with httpx.AsyncClient(timeout=20.0) as c:
            res = await c.get(_BASE_URL, params=qp)
    except httpx.HTTPError as e:
        return ToolResult.failure(f"http_error: {e}", status="network")

    if res.status_code >= 500:
        return ToolResult.failure(f"upstream_5xx: {res.status_code}", status="5xx")
    if res.status_code >= 400:
        return ToolResult.failure(f"client_{res.status_code}: {res.text[:200]}", status="4xx")

    body = res.json()
    # AV's three failure modes — all returned as HTTP 200 :(
    if "Error Message" in body:
        return ToolResult.failure(f"av_error: {body['Error Message']}", status="4xx")
    if "Note" in body and "API call frequency" in body["Note"]:
        return ToolResult.failure("av_rate_limited (note)", status="429")
    if "Information" in body and "rate limit" in str(body["Information"]).lower():
        return ToolResult.failure("av_rate_limited (info)", status="429")
    return ToolResult.success(body)


# ─── Public endpoints ──────────────────────────────────────────────────────


@cached(prefix="av:overview", ttl_seconds=86400)
@rate_limited(_PROVIDER, _PER_MIN, _PER_DAY)
@with_retry(attempts=3)
async def overview(symbol: str) -> ToolResult:
    """Company fundamentals snapshot: sector, P/E, EPS, market cap, etc."""
    return await _request({"function": "OVERVIEW", "symbol": symbol})


@cached(prefix="av:daily", ttl_seconds=43200)
@rate_limited(_PROVIDER, _PER_MIN, _PER_DAY)
@with_retry(attempts=3)
async def daily(symbol: str, outputsize: str = "compact") -> ToolResult:
    """Daily OHLCV. `compact` = last 100 trading days."""
    return await _request(
        {"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": outputsize}
    )


@cached(prefix="av:income", ttl_seconds=86400)
@rate_limited(_PROVIDER, _PER_MIN, _PER_DAY)
@with_retry(attempts=3)
async def income_statement(symbol: str) -> ToolResult:
    """Quarterly + annual income statements."""
    return await _request({"function": "INCOME_STATEMENT", "symbol": symbol})


@cached(prefix="av:news", ttl_seconds=3600)
@rate_limited(_PROVIDER, _PER_MIN, _PER_DAY)
@with_retry(attempts=3)
async def news_sentiment(tickers: str, limit: int = 20) -> ToolResult:
    """News + sentiment for one or more tickers (comma-separated)."""
    return await _request(
        {"function": "NEWS_SENTIMENT", "tickers": tickers, "limit": str(limit)}
    )
