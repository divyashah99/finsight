"""add thread_id to runs and reports

Revision ID: 0002_thread_id
Revises: 0001_initial
Create Date: 2026-08-06
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_thread_id"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("runs", sa.Column("thread_id", sa.String(64), nullable=True))
    op.create_index("ix_runs_thread_id", "runs", ["thread_id"])

    op.add_column("reports", sa.Column("thread_id", sa.String(64), nullable=True))
    op.create_index("ix_reports_thread_id", "reports", ["thread_id"])


def downgrade() -> None:
    op.drop_index("ix_reports_thread_id", table_name="reports")
    op.drop_column("reports", "thread_id")

    op.drop_index("ix_runs_thread_id", table_name="runs")
    op.drop_column("runs", "thread_id")
