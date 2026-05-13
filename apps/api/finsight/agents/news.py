"""News agent.

Wraps the Alpha Vantage NEWS_SENTIMENT endpoint via MCP. AV returns per-article
ticker-specific sentiment scores, so we filter to the requested ticker's
relevance and compute an aggregate score from there.

Aggregate label rules:
    score >  0.15 → bullish
    score < -0.15 → bearish
    otherwise     → neutral
"""

from __future__ import annotations

from typing import Any

from finsight.agents.state import AgentError, NewsBundle, NewsItem, ResearchState
from finsight.logging_setup import get_logger
from finsight.tools.mcp_client import mcp_session

log = get_logger(__name__)


def _bucket(score: float) -> str:
    if score > 0.15:
        return "bullish"
    if score < -0.15:
        return "bearish"
    return "neutral"


def _parse(data: dict[str, Any], ticker: str) -> NewsBundle:
    feed = data.get("feed") or []
    items: list[NewsItem] = []
    weighted: list[tuple[float, float]] = []  # (relevance, sentiment)

    for entry in feed[:25]:
        ts = next(
            (t for t in entry.get("ticker_sentiment", []) if t.get("ticker") == ticker),
            None,
        )
        try:
            sentiment = float(ts["ticker_sentiment_score"]) if ts else float(entry.get("overall_sentiment_score", 0))
            relevance = float(ts["relevance_score"]) if ts else 1.0
        except (ValueError, TypeError, KeyError):
            continue
        label = ts.get("ticker_sentiment_label") if ts else entry.get("overall_sentiment_label")

        items.append(
            NewsItem(
                title=entry.get("title", "")[:240],
                source=entry.get("source"),
                url=entry.get("url"),
                published=entry.get("time_published"),
                sentiment_score=sentiment,
                sentiment_label=label,
                summary=(entry.get("summary") or "")[:600] or None,
            )
        )
        weighted.append((relevance, sentiment))

    if weighted:
        total_w = sum(w for w, _ in weighted) or 1.0
        agg = sum(w * s for w, s in weighted) / total_w
    else:
        agg = None

    return NewsBundle(
        items=items,
        aggregate_sentiment=agg,
        aggregate_label=_bucket(agg) if agg is not None else None,
    )


async def run(state: ResearchState) -> dict[str, Any]:
    ticker = state["ticker"]
    log.info("news.start ticker=%s", ticker)
    async with mcp_session("alpha_vantage") as mcp:
        res = await mcp.call("av_news_sentiment", tickers=ticker, limit=25)

    if not res.ok or not isinstance(res.data, dict):
        return {
            "news": NewsBundle(),
            "errors": [AgentError(agent="news", error=res.error or "empty")],
        }

    bundle = _parse(res.data, ticker)
    log.info("news.done ticker=%s items=%d label=%s", ticker, len(bundle.items), bundle.aggregate_label)
    return {"news": bundle}
