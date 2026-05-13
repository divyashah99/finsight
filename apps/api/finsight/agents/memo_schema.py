"""Structured memo schema.

The output of the writer + critic pipeline. Strict JSON-schema so the LLM is
forced to emit valid `Memo` objects via `chat_structured`.

Citation IDs are the index (1-based) into the SEC evidence list passed in to
the writer; this avoids the model hallucinating URLs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Recommendation = Literal["buy", "hold", "sell", "no_opinion"]


class Argument(BaseModel):
    claim: str = Field(description="One-sentence claim")
    evidence: str = Field(description="Specific supporting data — numbers, quotes")
    citation_ids: list[int] = Field(
        default_factory=list,
        description="Indices (1-based) into the SEC citation list. Empty if claim is from fundamentals/news.",
    )


class Risk(BaseModel):
    title: str
    detail: str
    severity: Literal["low", "medium", "high"]
    citation_ids: list[int] = Field(default_factory=list)


class KeyMetric(BaseModel):
    name: str = Field(description="Metric name, e.g. 'P/E', 'Revenue TTM'")
    value: str = Field(description="Formatted value, e.g. '28.3', '$394B'")


class Memo(BaseModel):
    ticker: str
    as_of: str = Field(description="ISO date the memo was generated")
    recommendation: Recommendation
    conviction: int = Field(ge=1, le=5, description="1=low, 5=high")
    headline: str = Field(description="<= 120 chars, the one-liner thesis")

    thesis_bull: list[Argument] = Field(min_length=2, max_length=5)
    thesis_bear: list[Argument] = Field(min_length=2, max_length=5)

    key_metrics: list[KeyMetric] = Field(description="Headline numerical metrics")
    catalysts: list[str] = Field(description="Near-term catalysts to watch")
    risks: list[Risk] = Field(min_length=1, max_length=6)


class Critique(BaseModel):
    needs_revision: bool
    issues: list[str] = Field(default_factory=list)
    confidence: int = Field(ge=1, le=5, description="Critic's confidence in the memo")
    suggestion: str | None = None
