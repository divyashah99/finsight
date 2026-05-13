"""Market agent.

Fetches fundamentals + daily price history via the Alpha Vantage MCP server.
Pure data plumbing: parses MCP responses into typed `MarketSnapshot` and
`PriceBar` lists. No LLM call here — the Quant and Writer agents reason about
this data downstream.
"""

from __future__ import annotations

from typing import Any

from finsight.agents.state import AgentError, MarketSnapshot, PriceBar, ResearchState
from finsight.logging_setup import get_logger
from finsight.tools.mcp_client import mcp_session

log = get_logger(__name__)


def _to_float(v: Any) -> float | None:
    try:
        if v is None or v == "None" or v == "-":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_overview(d: dict[str, Any], ticker: str) -> MarketSnapshot:
    return MarketSnapshot(
        ticker=ticker,
        name=d.get("Name"),
        sector=d.get("Sector"),
        industry=d.get("Industry"),
        market_cap=_to_float(d.get("MarketCapitalization")),
        pe_ratio=_to_float(d.get("PERatio")),
        eps=_to_float(d.get("EPS")),
        profit_margin=_to_float(d.get("ProfitMargin")),
        revenue_ttm=_to_float(d.get("RevenueTTM")),
        week52_high=_to_float(d.get("52WeekHigh")),
        week52_low=_to_float(d.get("52WeekLow")),
        dividend_yield=_to_float(d.get("DividendYield")),
        description=d.get("Description"),
    )


def _parse_daily(d: dict[str, Any]) -> list[PriceBar]:
    series = d.get("Time Series (Daily)", {}) or {}
    bars: list[PriceBar] = []
    for date, row in series.items():
        bars.append(
            PriceBar(
                date=date,
                open=_to_float(row.get("1. open")) or 0.0,
                high=_to_float(row.get("2. high")) or 0.0,
                low=_to_float(row.get("3. low")) or 0.0,
                close=_to_float(row.get("4. close")) or 0.0,
                volume=int(_to_float(row.get("5. volume")) or 0),
            )
        )
    bars.sort(key=lambda b: b.date)  # oldest first
    return bars


async def run(state: ResearchState) -> dict[str, Any]:
    ticker = state["ticker"]
    log.info("market.start ticker=%s", ticker)
    errors: list[AgentError] = []
    snapshot: MarketSnapshot | None = None
    bars: list[PriceBar] = []

    async with mcp_session("alpha_vantage") as mcp:
        ov = await mcp.call("av_overview", symbol=ticker)
        if ov.ok and isinstance(ov.data, dict) and ov.data:
            snapshot = _parse_overview(ov.data, ticker)
        else:
            errors.append(AgentError(agent="market", error=f"overview: {ov.error or 'empty'}"))

        dl = await mcp.call("av_daily", symbol=ticker, outputsize="compact")
        if dl.ok and isinstance(dl.data, dict):
            bars = _parse_daily(dl.data)
            if not bars:
                errors.append(AgentError(agent="market", error="daily: empty series"))
        else:
            errors.append(AgentError(agent="market", error=f"daily: {dl.error or 'empty'}"))

    log.info("market.done ticker=%s bars=%d errors=%d", ticker, len(bars), len(errors))
    return {
        "market": snapshot,
        "price_bars": bars,
        "errors": errors,
    }
