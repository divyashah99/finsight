"""Postgres-backed KV cache with TTL.

Keeps Alpha Vantage / OpenAI calls cheap and replayable. We deliberately don't
use Redis — Postgres is already in the stack, and the call volume is tiny.

Usage:
    val = await cache.get("av:OVERVIEW:AAPL")
    if val is None:
        val = await fetch_overview("AAPL")
        await cache.set("av:OVERVIEW:AAPL", val, ttl_seconds=86400)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from finsight.db.client import session_scope
from finsight.db.models import CacheEntry
from finsight.logging_setup import get_logger

log = get_logger(__name__)


async def get(key: str) -> Any | None:
    async with session_scope() as s:
        row = (
            await s.execute(select(CacheEntry).where(CacheEntry.key == key))
        ).scalar_one_or_none()
        if row is None:
            return None
        if row.expires_at <= datetime.now(timezone.utc):
            await s.execute(delete(CacheEntry).where(CacheEntry.key == key))
            return None
        log.debug("cache.hit key=%s", key)
        return row.value


async def set(key: str, value: Any, ttl_seconds: int) -> None:
    expires = datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
    serializable = json.loads(json.dumps(value, default=str))  # coerce datetimes
    stmt = (
        pg_insert(CacheEntry)
        .values(
            key=key,
            value=serializable,
            expires_at=expires,
            created_at=datetime.now(timezone.utc),
        )
        .on_conflict_do_update(
            index_elements=["key"],
            set_={"value": serializable, "expires_at": expires},
        )
    )
    async with session_scope() as s:
        await s.execute(stmt)


async def purge_expired() -> int:
    async with session_scope() as s:
        res = await s.execute(
            delete(CacheEntry).where(CacheEntry.expires_at <= datetime.now(timezone.utc))
        )
        return res.rowcount or 0
