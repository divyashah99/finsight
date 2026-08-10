"""Market data parsing helpers.

Parse Yahoo Finance MCP responses into typed `MarketSnapshot` / `PriceBar`.
These are pure functions reused by the `get_fundamentals` / `get_price_history`
research tools (`tools/research_tools.py`).
"""

from __future__ import annotations

from typing import Any

from finsight.agents.state import MarketSnapshot, PriceBar


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
