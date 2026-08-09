"""Unit tests for the LLM reranker — the retrieve-then-rerank second stage.

These mock the LLM so they run with no network / API key.
"""

from __future__ import annotations

import pytest

from finsight.services import reranker
from finsight.services.reranker import _RankItem, _RankResult
from finsight.services.vectorstore import SearchHit


def _hit(chunk_index: int, text: str, section: str = "risk_factors") -> SearchHit:
    return SearchHit(
        score=0.0,
        payload={"accession": "acc-1", "chunk_index": chunk_index, "section": section, "text": text},
    )


@pytest.mark.asyncio
async def test_rerank_promotes_relevant_chunk_over_fusion_order(monkeypatch):
    """Fusion puts an off-topic chunk first; the reranker must reorder it below
    the on-topic chunk."""
    query = "supply chain concentration risk"
    hits = [
        _hit(1, "The company maintains a quarterly dividend policy."),   # fusion #1, irrelevant
        _hit(2, "A significant portion of components is sourced from a single supplier in Asia."),
    ]

    async def fake_chat_structured(messages, schema, **kwargs):
        # Model judges candidate [2] far more relevant than [1].
        return _RankResult(rankings=[_RankItem(index=2, score=0.95), _RankItem(index=1, score=0.05)])

    monkeypatch.setattr(reranker.llm, "chat_structured", fake_chat_structured)

    out = await reranker.rerank(query, hits, top_n=2)
    assert [h.payload["chunk_index"] for h in out] == [2, 1]
    assert out[0].score == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_rerank_falls_back_to_input_order_on_error(monkeypatch):
    hits = [_hit(1, "a"), _hit(2, "b"), _hit(3, "c")]

    async def boom(messages, schema, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(reranker.llm, "chat_structured", boom)

    out = await reranker.rerank("q", hits, top_n=2)
    assert [h.payload["chunk_index"] for h in out] == [1, 2]


@pytest.mark.asyncio
async def test_rerank_truncates_to_top_n(monkeypatch):
    hits = [_hit(i, f"text {i}") for i in range(1, 6)]

    async def fake(messages, schema, **kwargs):
        # ascending relevance → chunk 5 best
        return _RankResult(rankings=[_RankItem(index=i, score=i / 10) for i in range(1, 6)])

    monkeypatch.setattr(reranker.llm, "chat_structured", fake)

    out = await reranker.rerank("q", hits, top_n=2)
    assert [h.payload["chunk_index"] for h in out] == [5, 4]


@pytest.mark.asyncio
async def test_rerank_single_hit_short_circuits():
    hits = [_hit(1, "only")]
    out = await reranker.rerank("q", hits, top_n=8)
    assert len(out) == 1
