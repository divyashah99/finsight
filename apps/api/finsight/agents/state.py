"""Shared graph state + agent output models.

LangGraph passes a single `ResearchState` dict through the DAG. Each agent node
returns a partial dict that LangGraph merges in. We use Pydantic models for the
nested payloads so type errors surface immediately and the writer can be handed
strongly-typed inputs.

The `errors` field uses a list reducer so multiple agents can each append
without clobbering each other when they run in parallel.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


# ─── Per-agent output payloads ─────────────────────────────────────────────


class MarketSnapshot(BaseModel):
    ticker: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    eps: float | None = None
    profit_margin: float | None = None
    revenue_ttm: float | None = None
    week52_high: float | None = None
    week52_low: float | None = None
    dividend_yield: float | None = None
    description: str | None = None


class PriceBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class QuantSignals(BaseModel):
    """Technical indicators derived from the daily series.

    Fields default to None so the writer can render gracefully when we have
    insufficient history (e.g. recently-IPO'd ticker).
    """

    last_close: float | None = None
    return_1m: float | None = None
    return_3m: float | None = None
    return_12m: float | None = None
    volatility_30d: float | None = None
    volatility_90d: float | None = None
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    above_sma_200: bool | None = None
    summary: str | None = None  # human-readable one-liner


class NewsItem(BaseModel):
    title: str
    source: str | None = None
    url: str | None = None
    published: str | None = None
    sentiment_score: float | None = None
    sentiment_label: str | None = None
    summary: str | None = None


class NewsBundle(BaseModel):
    items: list[NewsItem] = Field(default_factory=list)
    aggregate_sentiment: float | None = None
    aggregate_label: Literal["bullish", "neutral", "bearish"] | None = None


class SECCitation(BaseModel):
    accession: str
    form_type: str
    section: str | None = None
    filed_date: str | None = None
    url: str | None = None
    excerpt: str


class SECEvidence(BaseModel):
    citations: list[SECCitation] = Field(default_factory=list)
    risk_summary: str | None = None
    mdna_summary: str | None = None


class AgentError(BaseModel):
    agent: str
    error: str


# ─── Top-level graph state ─────────────────────────────────────────────────


class ResearchState(TypedDict, total=False):
    """LangGraph state.

    `total=False` means every key is optional — agents fill in their slice as
    they run. The `errors` reducer is `operator.add` so parallel agents can
    each append without overwriting.
    """

    ticker: str
    price_bars: list[PriceBar]
    market: MarketSnapshot | None
    quant: QuantSignals | None
    news: NewsBundle | None
    sec: SECEvidence | None
    draft_memo: dict[str, Any] | None
    critique: dict[str, Any] | None
    final_memo: dict[str, Any] | None
    revision_count: int
    errors: Annotated[list[AgentError], operator.add]
