"""GET/POST /research/{ticker} — runs the agentic graph and streams SSE.

The supervisor decides the sequence at runtime, so the event stream is dynamic
(not a fixed set of agents):

    event: start                data: {"ticker": "AAPL"}
    event: supervisor_decision  data: {"next": "filings", "reason": "..."}
    event: specialist_done      data: {"agent": "filings", "summary": "...", "citations": 4}
    event: final                data: {"report_id": "...", "thread_id": "...", "memo": {...}}
    event: error                data: {"error": "..."}

The frontend builds the timeline from `supervisor_decision` / `specialist_done`.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from finsight.agents.graph import stream_run
from finsight.agents.specialists import SPECIALISTS
from finsight.db.client import session_scope
from finsight.db.models import Report, Run
from finsight.logging_setup import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/research", tags=["research"])


@router.get("/{ticker}")
@router.post("/{ticker}")
async def research(ticker: str) -> EventSourceResponse:
    ticker = ticker.upper().strip()
    if not ticker or not ticker.isalnum() or len(ticker) > 8:
        raise HTTPException(status_code=400, detail="invalid ticker")

    run_id = uuid.uuid4()
    thread_id = str(run_id)  # follow-up chat thread for this memo
    started = datetime.now(timezone.utc)

    async with session_scope() as s:
        s.add(Run(id=run_id, ticker=ticker, thread_id=thread_id, status="running", started_at=started))

    async def event_stream() -> AsyncIterator[dict[str, Any]]:
        final_memo: dict[str, Any] = {}
        all_errors: list[Any] = []

        yield {"event": "start", "data": json.dumps({"ticker": ticker})}

        try:
            async for node, partial in stream_run(ticker):
                if node == "supervisor":
                    yield {
                        "event": "supervisor_decision",
                        "data": json.dumps(
                            {"next": partial.get("next_agent"), "reason": partial.get("supervisor_reason", "")}
                        ),
                    }
                elif node in SPECIALISTS:
                    findings = partial.get("findings") or []
                    summary = findings[-1].summary if findings else ""
                    yield {
                        "event": "specialist_done",
                        "data": json.dumps(
                            {
                                "agent": node,
                                "summary": summary[:400],
                                "citations": len(partial.get("citations") or []),
                            }
                        ),
                    }
                    all_errors.extend(partial.get("errors") or [])
                elif node == "synthesizer":
                    final_memo = partial.get("final_memo") or {}
        except Exception as e:  # noqa: BLE001
            log.exception("graph.error ticker=%s", ticker)
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
            return

        finished = datetime.now(timezone.utc)
        report_id = uuid.uuid4()
        try:
            async with session_scope() as s:
                s.add(
                    Report(
                        id=report_id,
                        run_id=run_id,
                        ticker=ticker,
                        thread_id=thread_id,
                        memo=final_memo,
                        created_at=finished,
                    )
                )
                run = await s.get(Run, run_id)
                if run:
                    run.status = "done"
                    run.finished_at = finished
                    run.duration_ms = int((finished - started).total_seconds() * 1000)
                    run.state_snapshot = {
                        "errors": [e.model_dump() if hasattr(e, "model_dump") else e for e in all_errors],
                    }
        except Exception as e:  # noqa: BLE001
            log.exception("persist.error")
            yield {"event": "error", "data": json.dumps({"error": f"persist: {e}"})}
            return

        yield {
            "event": "final",
            "data": json.dumps(
                {"report_id": str(report_id), "thread_id": thread_id, "memo": final_memo},
                default=str,
            ),
        }

    return EventSourceResponse(event_stream())
