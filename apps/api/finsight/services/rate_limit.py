"""Persisted token-bucket rate limiter.

State lives in Postgres so limits survive container restarts (Alpha Vantage's
daily quota would otherwise reset every deploy). We acquire one token per call;
if no token is available, the call blocks for up to `max_wait_seconds` waiting
for refill — past that we raise `RateLimitExceeded`.

Two buckets per provider:
    (provider, "min")  → short-window burst control
    (provider, "day")  → daily quota
A call must acquire from BOTH buckets, atomically per bucket.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from finsight.db.client import session_scope
from finsight.db.models import RateLimitBucket
from finsight.logging_setup import get_logger

log = get_logger(__name__)


class RateLimitExceeded(RuntimeError):
    pass


async def _refill_and_take(
    provider: str,
    window: str,
    capacity: float,
    refill_per_sec: float,
) -> tuple[bool, float]:
    """Atomically: refill bucket, attempt to consume 1 token.

    Returns (acquired, seconds_to_next_token).
    """
    now = datetime.now(timezone.utc)

    async with session_scope() as s:
        row = (
            await s.execute(
                select(RateLimitBucket)
                .where(RateLimitBucket.provider == provider, RateLimitBucket.window == window)
                .with_for_update()
            )
        ).scalar_one_or_none()

        if row is None:
            # First seed — start full so cold starts don't get blocked.
            stmt = (
                pg_insert(RateLimitBucket)
                .values(
                    provider=provider,
                    window=window,
                    tokens=capacity - 1,
                    capacity=capacity,
                    refill_per_sec=refill_per_sec,
                    updated_at=now,
                )
                .on_conflict_do_nothing()
            )
            await s.execute(stmt)
            return True, 0.0

        elapsed = (now - row.updated_at).total_seconds()
        new_tokens = min(row.capacity, row.tokens + elapsed * row.refill_per_sec)

        if new_tokens >= 1.0:
            row.tokens = new_tokens - 1
            row.updated_at = now
            return True, 0.0

        wait = (1.0 - new_tokens) / row.refill_per_sec if row.refill_per_sec > 0 else 999.0
        row.tokens = new_tokens
        row.updated_at = now
        return False, wait


async def acquire(
    provider: str,
    per_minute: int,
    per_day: int,
    max_wait_seconds: float = 60.0,
) -> None:
    """Acquire one token from both the minute and day buckets.

    Backs off and retries while wait < `max_wait_seconds`. Raises
    RateLimitExceeded if the daily quota is exhausted.
    """
    buckets = [
        ("min", float(per_minute), per_minute / 60.0),
        ("day", float(per_day), per_day / 86400.0),
    ]

    waited = 0.0
    while True:
        worst_wait = 0.0
        all_ok = True
        for window, capacity, refill in buckets:
            ok, wait = await _refill_and_take(provider, window, capacity, refill)
            if not ok:
                all_ok = False
                worst_wait = max(worst_wait, wait)

        if all_ok:
            return

        if waited + worst_wait > max_wait_seconds:
            raise RateLimitExceeded(
                f"{provider}: needed {worst_wait:.1f}s wait (cap {max_wait_seconds:.1f}s)"
            )

        log.info("rate_limit.wait provider=%s seconds=%.2f", provider, worst_wait)
        await asyncio.sleep(min(worst_wait, 5.0))
        waited += min(worst_wait, 5.0)
