"""SQLAlchemy ORM models.

Five tables cover the whole backend:

- `reports`     — final memos returned to users
- `runs`        — one row per `/research` invocation (status, ticker, timings)
- `cache`       — generic KV cache used by the `@cached` tool decorator
- `rate_limit_buckets` — persisted token-bucket state per provider
- `sec_docs`    — metadata for ingested SEC filings (chunks live in Qdrant)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, BigInteger, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    status: Mapped[str] = mapped_column(String(32), default="running")  # running|done|error
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    state_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    memo: Mapped[dict] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


class CacheEntry(Base):
    __tablename__ = "cache"

    key: Mapped[str] = mapped_column(String(512), primary_key=True)
    value: Mapped[dict] = mapped_column(JSONB)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class RateLimitBucket(Base):
    """Token-bucket state, one row per (provider, window) pair.

    `tokens` is fractional (Float) so we can refill at sub-second precision.
    `capacity` and `refill_per_sec` are persisted with the row so config changes
    take effect on next acquisition.
    """

    __tablename__ = "rate_limit_buckets"

    provider: Mapped[str] = mapped_column(String(64), primary_key=True)
    window: Mapped[str] = mapped_column(String(16), primary_key=True)  # "min" | "day"
    tokens: Mapped[float] = mapped_column(Float)
    capacity: Mapped[float] = mapped_column(Float)
    refill_per_sec: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class SECDoc(Base):
    """One row per ingested filing. Vector chunks live in Qdrant; this table is
    the system-of-record for what's been ingested and when.
    """

    __tablename__ = "sec_docs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    cik: Mapped[str] = mapped_column(String(16), index=True)
    accession_number: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    form_type: Mapped[str] = mapped_column(String(16))  # "10-K" | "10-Q" | ...
    filed_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    url: Mapped[str] = mapped_column(Text)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    bytes_size: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    meta: Mapped[dict | None] = mapped_column(JSON, nullable=True)
