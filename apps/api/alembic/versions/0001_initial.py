"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-11
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="running"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("state_snapshot", postgresql.JSONB(), nullable=True),
    )
    op.create_index("ix_runs_ticker", "runs", ["ticker"])

    op.create_table(
        "reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("memo", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_reports_run_id", "reports", ["run_id"])
    op.create_index("ix_reports_ticker", "reports", ["ticker"])
    op.create_index("ix_reports_created_at", "reports", ["created_at"])

    op.create_table(
        "cache",
        sa.Column("key", sa.String(512), primary_key=True),
        sa.Column("value", postgresql.JSONB(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_cache_expires_at", "cache", ["expires_at"])

    op.create_table(
        "rate_limit_buckets",
        sa.Column("provider", sa.String(64), primary_key=True),
        sa.Column("window", sa.String(16), primary_key=True),
        sa.Column("tokens", sa.Float(), nullable=False),
        sa.Column("capacity", sa.Float(), nullable=False),
        sa.Column("refill_per_sec", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "sec_docs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("ticker", sa.String(16), nullable=False),
        sa.Column("cik", sa.String(16), nullable=False),
        sa.Column("accession_number", sa.String(32), nullable=False, unique=True),
        sa.Column("form_type", sa.String(16), nullable=False),
        sa.Column("filed_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("bytes_size", sa.BigInteger(), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=True),
    )
    op.create_index("ix_sec_docs_ticker", "sec_docs", ["ticker"])
    op.create_index("ix_sec_docs_cik", "sec_docs", ["cik"])


def downgrade() -> None:
    op.drop_table("sec_docs")
    op.drop_table("rate_limit_buckets")
    op.drop_table("cache")
    op.drop_table("reports")
    op.drop_table("runs")
