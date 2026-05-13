"""Qdrant client wrapper.

Single collection (`settings.qdrant_collection`) holds chunks from every
filing across every ticker. We rely on payload filtering for per-ticker
retrieval rather than per-ticker collections — keeps the collection count
constant as we add tickers.

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

import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from finsight.logging_setup import get_logger
from finsight.settings import settings

log = get_logger(__name__)

_EMBED_DIM = 1536  # text-embedding-3-small


_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return _client


async def ensure_collection() -> None:
    client = get_client()
    existing = {c.name for c in (await client.get_collections()).collections}
    if settings.qdrant_collection in existing:
        return
    log.info("qdrant.create_collection name=%s", settings.qdrant_collection)
    await client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=qm.VectorParams(size=_EMBED_DIM, distance=qm.Distance.COSINE),
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
    points: list[tuple[list[float], dict[str, Any]]],
) -> int:
    if not points:
        return 0
    client = get_client()
    qpoints = [
        qm.PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload)
        for vec, payload in points
    ]
    await client.upsert(collection_name=settings.qdrant_collection, points=qpoints)
    return len(qpoints)


@dataclass
class SearchHit:
    score: float
    payload: dict[str, Any]


async def search(
    embedding: list[float],
    ticker: str,
    *,
    limit: int = 8,
    form_types: list[str] | None = None,
    sections: list[str] | None = None,
) -> list[SearchHit]:
    client = get_client()
    must: list[qm.Condition] = [qm.FieldCondition(key="ticker", match=qm.MatchValue(value=ticker))]
    if form_types:
        must.append(qm.FieldCondition(key="form_type", match=qm.MatchAny(any=form_types)))
    if sections:
        must.append(qm.FieldCondition(key="section", match=qm.MatchAny(any=sections)))

    res = await client.query_points(
        collection_name=settings.qdrant_collection,
        query=embedding,
        limit=limit,
        query_filter=qm.Filter(must=must),
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
