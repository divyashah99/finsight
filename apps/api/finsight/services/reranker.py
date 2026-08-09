"""LLM reranker.

Hybrid retrieval (dense + BM25 fusion) is good at *recall* but its fused rank is
coarse. A second-stage reranker reorders the candidate pool by judged relevance
to the actual query — the standard retrieve-then-rerank pattern.

We use the existing `gpt-4o-mini` client (no extra API key, no heavy local
cross-encoder to ship on a free-tier box) in a listwise setup: show the model
the query + numbered candidate excerpts, get back a relevance score per index,
then sort. On any failure we fall back to the input (fusion) order so retrieval
never hard-fails on the reranker.
"""

from __future__ import annotations

from langsmith import traceable
from pydantic import BaseModel, Field

from finsight.logging_setup import get_logger
from finsight.services import llm
from finsight.services.vectorstore import SearchHit

log = get_logger(__name__)

_MAX_CANDIDATES = 20  # cap tokens: rerank at most this many per call
_EXCERPT_CHARS = 500

_SYSTEM = (
    "You are a retrieval reranker for equity-research over SEC filings. "
    "Given a query and numbered candidate passages, score how relevant each "
    "passage is to answering the query, from 0.0 (irrelevant) to 1.0 (directly "
    "answers it). Judge only relevance to the query, not writing quality. "
    "Return a score for every candidate index shown."
)


class _RankItem(BaseModel):
    index: int = Field(description="1-based index of the candidate passage")
    score: float = Field(description="relevance 0.0-1.0")


class _RankResult(BaseModel):
    rankings: list[_RankItem]


def _build_user_message(query: str, hits: list[SearchHit]) -> str:
    lines = [f"Query: {query}", "", "Candidates:"]
    for i, h in enumerate(hits, start=1):
        text = (h.payload.get("text") or "")[:_EXCERPT_CHARS].replace("\n", " ")
        section = h.payload.get("section") or "?"
        lines.append(f"[{i}] (section={section}) {text}")
    return "\n".join(lines)


@traceable(run_type="chain", name="reranker.rerank")
async def rerank(query: str, hits: list[SearchHit], *, top_n: int) -> list[SearchHit]:
    """Reorder `hits` by LLM-judged relevance to `query`; return the top `top_n`.

    Degrades gracefully to the input order on any error.
    """
    if len(hits) <= 1:
        return hits[:top_n]

    candidates = hits[:_MAX_CANDIDATES]
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _build_user_message(query, candidates)},
    ]
    try:
        result = await llm.chat_structured(messages, schema=_RankResult, temperature=0.0)
    except Exception as e:  # noqa: BLE001
        log.warning("rerank.failed falling_back error=%s", e)
        return hits[:top_n]

    # Map 1-based indices back to hits; unseen candidates keep a floor score so
    # they still trail anything the model scored.
    scored: list[tuple[float, SearchHit]] = []
    seen: set[int] = set()
    for item in result.rankings:
        idx = item.index - 1
        if 0 <= idx < len(candidates) and idx not in seen:
            seen.add(idx)
            scored.append((item.score, SearchHit(score=item.score, payload=candidates[idx].payload)))
    for i, h in enumerate(candidates):
        if i not in seen:
            scored.append((-1.0, SearchHit(score=0.0, payload=h.payload)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored[:top_n]]
