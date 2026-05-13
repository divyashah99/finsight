"""APScheduler — periodic background jobs.

For now: nightly re-ingest for any ticker we've seen in a recent `Run`. Cheap
because `sec_ingest.ingest_ticker` is idempotent (skips filings already in DB).

The scheduler is bound to FastAPI's lifespan in `main.py` so it starts/stops
with the app.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select

from finsight.db.client import session_scope
from finsight.db.models import Run
from finsight.logging_setup import get_logger
from finsight.services import sec_ingest

log = get_logger(__name__)

_scheduler: AsyncIOScheduler | None = None


async def refresh_recent_tickers() -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    async with session_scope() as s:
        rows = (
            await s.execute(select(Run.ticker).where(Run.started_at >= cutoff).distinct())
        ).scalars().all()
    tickers = list(set(rows))
    log.info("scheduler.refresh tickers=%d", len(tickers))
    for t in tickers:
        try:
            await sec_ingest.ingest_ticker(t)
        except Exception as e:  # noqa: BLE001
            log.warning("scheduler.refresh_failed ticker=%s error=%s", t, e)


def start() -> None:
    global _scheduler
    if _scheduler is not None:
        return
    _scheduler = AsyncIOScheduler(timezone="UTC")
    # Daily at 04:30 UTC — well before US market open
    _scheduler.add_job(refresh_recent_tickers, CronTrigger(hour=4, minute=30))
    _scheduler.start()
    log.info("scheduler.started")


def shutdown() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        log.info("scheduler.stopped")
