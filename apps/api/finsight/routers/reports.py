"""GET /reports/{id} — fetch a stored memo."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from finsight.db.client import session_scope
from finsight.db.models import Report

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/{report_id}")
async def get_report(report_id: str) -> dict:
    try:
        rid = uuid.UUID(report_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid report id")

    async with session_scope() as s:
        rpt = await s.get(Report, rid)
        if not rpt:
            raise HTTPException(status_code=404, detail="not found")
        return {
            "id": str(rpt.id),
            "ticker": rpt.ticker,
            "memo": rpt.memo,
            "created_at": rpt.created_at.isoformat(),
        }
