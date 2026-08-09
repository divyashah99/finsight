"""Market-data client — Yahoo Finance (yfinance) backed.

Historically this wrapped Alpha Vantage, but its free tier (25 requests/day)
made the demo unusable. We now fetch from Yahoo Finance via `yfinance` — keyless
and without a hard daily cap. To avoid churn, each function keeps its name and
returns the **same dict shape** the parse helpers expect (`market._parse_overview`,
`market._parse_daily`, `news._parse`), so the MCP server, research tools, and
agents are unchanged. `yfinance` is synchronous, so calls run in a thread.

Caching + retry decorators still apply; the per-key rate limiter is dropped
(no API key / per-key quota anymore).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
import yfinance as yf

from finsight.logging_setup import get_logger
from finsight.tools.base import ToolResult, cached, with_retry

log = get_logger(__name__)


def _f(v: Any) -> Any:
    """None-safe float for pandas/NaN values."""
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# ─── Sync fetchers (run in a thread) ───────────────────────────────────────


def _fetch_overview(symbol: str) -> dict[str, Any]:
    info = yf.Ticker(symbol).get_info() or {}
    if not info or not (info.get("longName") or info.get("shortName")):
        return {}
    # Map to the Alpha-Vantage field names the parser expects.
    return {
        "Name": info.get("longName") or info.get("shortName"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
        "MarketCapitalization": info.get("marketCap"),
        "PERatio": info.get("trailingPE"),
        "EPS": info.get("trailingEps"),
        "ProfitMargin": info.get("profitMargins"),
        "RevenueTTM": info.get("totalRevenue"),
        "52WeekHigh": info.get("fiftyTwoWeekHigh"),
        "52WeekLow": info.get("fiftyTwoWeekLow"),
        "DividendYield": info.get("dividendYield"),
        "Description": info.get("longBusinessSummary"),
    }


def _fetch_daily(symbol: str) -> dict[str, Any]:
    # 1y so SMA-200 is computable (better than AV's 100-day "compact").
    df = yf.Ticker(symbol).history(period="1y", interval="1d", auto_adjust=False)
    if df is None or df.empty:
        return {}
    series: dict[str, Any] = {}
    for idx, row in df.iterrows():
        day = idx.date().isoformat()
        series[day] = {
            "1. open": _f(row.get("Open")),
            "2. high": _f(row.get("High")),
            "3. low": _f(row.get("Low")),
            "4. close": _f(row.get("Close")),
            "5. volume": _f(row.get("Volume")) or 0,
        }
    return {"Time Series (Daily)": series}


def _income_reports(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    reports: list[dict[str, Any]] = []
    for col in list(df.columns)[:2]:
        def g(label: str) -> Any:
            try:
                return _f(df.loc[label, col])
            except Exception:  # noqa: BLE001
                return None

        reports.append(
            {
                "fiscalDateEnding": col.date().isoformat() if hasattr(col, "date") else str(col),
                "totalRevenue": g("Total Revenue"),
                "costOfRevenue": g("Cost Of Revenue"),
                "operatingIncome": g("Operating Income"),
                "netIncome": g("Net Income"),
            }
        )
    return reports


def _fetch_income(symbol: str) -> dict[str, Any]:
    t = yf.Ticker(symbol)
    return {
        "annualReports": _income_reports(getattr(t, "income_stmt", None)),
        "quarterlyReports": _income_reports(getattr(t, "quarterly_income_stmt", None)),
    }


def _fetch_news(symbol: str, limit: int) -> dict[str, Any]:
    items = yf.Ticker(symbol).news or []
    feed: list[dict[str, Any]] = []
    for it in items[:limit]:
        # yfinance changed shape across versions: flat vs nested "content".
        content = it.get("content") if isinstance(it.get("content"), dict) else None
        title = it.get("title") or (content.get("title") if content else None)
        if not title:
            continue
        if content:
            provider = (content.get("provider") or {}).get("displayName")
            url = (content.get("canonicalUrl") or content.get("clickThroughUrl") or {}).get("url")
            published = content.get("pubDate")
            summary = content.get("summary")
        else:
            provider = it.get("publisher")
            url = it.get("link")
            published = it.get("providerPublishTime")
            summary = it.get("summary")
        feed.append(
            {
                "title": title,
                "source": provider,
                "url": url,
                "time_published": published,
                "summary": summary,
            }
        )
    return {"feed": feed}


async def _run(fn, *args) -> ToolResult:
    try:
        data = await asyncio.to_thread(fn, *args)
    except Exception as e:  # noqa: BLE001
        return ToolResult.failure(f"yfinance_error: {e}", status="network")
    if not data:
        return ToolResult.failure("no data returned", status="empty")
    return ToolResult.success(data)


# ─── Public endpoints (names + shapes preserved) ───────────────────────────


@cached(prefix="av:overview", ttl_seconds=86400)
@with_retry(attempts=3)
async def overview(symbol: str) -> ToolResult:
    """Company fundamentals snapshot: sector, P/E, EPS, market cap, etc."""
    return await _run(_fetch_overview, symbol)


@cached(prefix="av:daily", ttl_seconds=43200)
@with_retry(attempts=3)
async def daily(symbol: str, outputsize: str = "compact") -> ToolResult:
    """Daily OHLCV time series (~1y of history)."""
    return await _run(_fetch_daily, symbol)


@cached(prefix="av:income", ttl_seconds=86400)
@with_retry(attempts=3)
async def income_statement(symbol: str) -> ToolResult:
    """Quarterly + annual income statements."""
    return await _run(_fetch_income, symbol)


@cached(prefix="av:news", ttl_seconds=3600)
@with_retry(attempts=3)
async def news_sentiment(tickers: str, limit: int = 20) -> ToolResult:
    """Recent news headlines for a ticker. (Yahoo has no sentiment scores, so the
    NewsAnalyst reports headlines; aggregate sentiment is neutral/unknown.)"""
    symbol = tickers.split(",")[0].strip()
    return await _run(_fetch_news, symbol, limit)
