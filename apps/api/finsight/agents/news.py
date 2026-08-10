"""News sentiment parsing helpers.

Parse the Yahoo Finance news MCP response into a typed `NewsBundle`
(filter to the ticker's per-article relevance, aggregate a weighted score).
Pure functions reused by the `get_news_sentiment` research tool.

Aggregate label rules:
    score >  0.15 → bullish
    score < -0.15 → bearish
    otherwise     → neutral
"""

from __future__ import annotations

from typing import Any

from finsight.agents.state import NewsBundle, NewsItem


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
