"""Qdrant client wrapper — hybrid (dense + sparse BM25) retrieval.

Single collection (`settings.qdrant_collection`) holds chunks from every
filing across every ticker. We rely on payload filtering for per-ticker
retrieval rather than per-ticker collections — keeps the collection count
constant as we add tickers.

Each point carries two named vectors:
    - "dense": OpenAI `text-embedding-3-small` (1536-d, cosine) — semantic recall
    - "bm25":  FastEmbed `Qdrant/bm25` sparse vector — exact-lexical recall

`hybrid_search()` runs both and fuses with Reciprocal Rank Fusion (RRF) server-
side. Dense alone misses exact terms common in 10-Ks (defined terms, "Item 1A",
segment/product names); the sparse branch catches those.

Payload schema:
    {
        "ticker":          "AAPL",
        "cik":             "0000320193",
        "accession":       "0000320193-24-000123",
        "form_type":       "10-K",
        "section":         "risk_factors",
        "filed_date":      "2024-11-01",
        "url":             "https://...",
        "chunk_index":     7,
        "text":            "..."
    }
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from finsight.logging_setup import get_logger
from finsight.settings import settings

log = get_logger(__name__)

_EMBED_DIM = 1536  # text-embedding-3-small
_DENSE = "dense"
_SPARSE = "bm25"

# A sparse vector as (indices, values) — parallel lists.
SparseVec = tuple[list[int], list[float]]


_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return _client


# ─── Sparse (BM25) embeddings via FastEmbed ────────────────────────────────
#
# The model is CPU-only and downloaded once on first use. It's synchronous, so
# we run it in a thread to avoid blocking the event loop.

_sparse_model: Any = None


def _get_sparse_model() -> Any:
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding

        log.info("fastembed.load model=Qdrant/bm25")
        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_model


def _sparse_embed_sync(texts: list[str], *, query: bool) -> list[SparseVec]:
    model = _get_sparse_model()
    # query_embed skips document-side term weighting; IDF is applied by Qdrant
    # (the "bm25" vector is created with Modifier.IDF).
    gen = model.query_embed(texts) if query else model.embed(texts)
    out: list[SparseVec] = []
    for emb in gen:
        out.append((emb.indices.tolist(), emb.values.tolist()))
    return out


async def sparse_embed(texts: list[str], *, query: bool = False) -> list[SparseVec]:
    """Batch BM25 sparse embeddings. Set `query=True` for search queries."""
    if not texts:
        return []
    return await asyncio.to_thread(_sparse_embed_sync, texts, query=query)


def _to_sparse_vector(sv: SparseVec) -> qm.SparseVector:
    indices, values = sv
    return qm.SparseVector(indices=indices, values=values)


# ─── Collection lifecycle ──────────────────────────────────────────────────


async def ensure_collection() -> None:
    client = get_client()
    existing = {c.name for c in (await client.get_collections()).collections}
    if settings.qdrant_collection in existing:
        return
    log.info("qdrant.create_collection name=%s", settings.qdrant_collection)
    await client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config={_DENSE: qm.VectorParams(size=_EMBED_DIM, distance=qm.Distance.COSINE)},
        sparse_vectors_config={_SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
    )
    # Payload indexes — make filters fast
    for field, schema in [
        ("ticker", qm.PayloadSchemaType.KEYWORD),
        ("form_type", qm.PayloadSchemaType.KEYWORD),
        ("section", qm.PayloadSchemaType.KEYWORD),
        ("accession", qm.PayloadSchemaType.KEYWORD),
    ]:
        try:
            await client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=field,
                field_schema=schema,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("qdrant.index_skip field=%s reason=%s", field, e)


async def upsert_chunks(
    points: list[tuple[list[float], SparseVec, dict[str, Any]]],
) -> int:
    """Upsert points carrying both dense and sparse vectors.

    Each element is `(dense_vector, (sparse_indices, sparse_values), payload)`.
    """
    if not points:
        return 0
    client = get_client()
    qpoints = [
        qm.PointStruct(
            id=str(uuid.uuid4()),
            vector={_DENSE: dense, _SPARSE: _to_sparse_vector(sparse)},
            payload=payload,
        )
        for dense, sparse, payload in points
    ]
    await client.upsert(collection_name=settings.qdrant_collection, points=qpoints)
    return len(qpoints)


@dataclass
class SearchHit:
    score: float
    payload: dict[str, Any]


def _ticker_filter(
    ticker: str,
    form_types: list[str] | None,
    sections: list[str] | None,
) -> qm.Filter:
    must: list[qm.Condition] = [
        qm.FieldCondition(key="ticker", match=qm.MatchValue(value=ticker))
    ]
    if form_types:
        must.append(qm.FieldCondition(key="form_type", match=qm.MatchAny(any=form_types)))
    if sections:
        must.append(qm.FieldCondition(key="section", match=qm.MatchAny(any=sections)))
    return qm.Filter(must=must)


async def search(
    embedding: list[float],
    ticker: str,
    *,
    limit: int = 8,
    form_types: list[str] | None = None,
    sections: list[str] | None = None,
) -> list[SearchHit]:
    """Dense-only search (kept for callers that don't need hybrid)."""
    client = get_client()
    res = await client.query_points(
        collection_name=settings.qdrant_collection,
        query=embedding,
        using=_DENSE,
        limit=limit,
        query_filter=_ticker_filter(ticker, form_types, sections),
        with_payload=True,
    )
    return [SearchHit(score=p.score, payload=p.payload or {}) for p in res.points]


async def hybrid_search(
    dense_embedding: list[float],
    sparse_embedding: SparseVec,
    ticker: str,
    *,
    limit: int = 12,
    prefetch_limit: int | None = None,
    form_types: list[str] | None = None,
    sections: list[str] | None = None,
) -> list[SearchHit]:
    """Dense + sparse retrieval fused with Reciprocal Rank Fusion (RRF).

    Runs two `prefetch` branches (each with the same ticker/section payload
    filter) and lets Qdrant fuse them, so a chunk ranked highly by *either*
    signal surfaces. `score` on the returned hits is the RRF score.
    """
    client = get_client()
    filt = _ticker_filter(ticker, form_types, sections)
    pool = prefetch_limit or max(limit * 2, 20)
    res = await client.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            qm.Prefetch(query=dense_embedding, using=_DENSE, filter=filt, limit=pool),
            qm.Prefetch(
                query=_to_sparse_vector(sparse_embedding),
                using=_SPARSE,
                filter=filt,
                limit=pool,
            ),
        ],
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return [SearchHit(score=p.score, payload=p.payload or {}) for p in res.points]


async def count_for_ticker(ticker: str) -> int:
    client = get_client()
    res = await client.count(
        collection_name=settings.qdrant_collection,
        count_filter=qm.Filter(
            must=[qm.FieldCondition(key="ticker", match=qm.MatchValue(value=ticker))]
        ),
        exact=True,
    )
    return res.count
