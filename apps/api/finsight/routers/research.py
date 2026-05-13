"""POST /research/{ticker} — runs the agent graph and streams events as SSE.

Event schema (one event per LangGraph node update + per token):

    event: agent_start   data: {"agent": "market"}
    event: agent_done    data: {"agent": "market", "summary": "..."}
    event: token         data: {"delta": "..."}
    event: final         data: {"report_id": "...", "memo": {...}}
    event: error         data: {"error": "..."}

The frontend renders the timeline from `agent_*` events and pipes `token`
deltas into the memo viewer.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from finsight.agents.graph import stream_run
from finsight.db.client import session_scope
from finsight.db.models import Report, Run
from finsight.logging_setup import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/research", tags=["research"])


def _agent_summary(node: str, partial: dict[str, Any]) -> str:
    if node == "market":
        m = partial.get("market")
        if m and getattr(m, "name", None):
            return f"{m.name} · {m.sector or '?'}"
        return "fundamentals fetched"
    if node == "quant":
        q = partial.get("quant")
        return q.summary if q and q.summary else "signals computed"
    if node == "writer":
        return "memo drafted"
    if node == "news":
        n = partial.get("news")
        if n and getattr(n, "aggregate_label", None):
            return f"{n.aggregate_label} sentiment ({len(n.items)} articles)"
        return "news scanned"
    if node == "sec":
        s = partial.get("sec")
        if s:
            return f"{len(s.citations)} SEC citations retrieved"
        return "SEC filings searched"
    if node == "critic":
        return "critique complete"
    return node


@router.get("/{ticker}")
@router.post("/{ticker}")
async def research(ticker: str) -> EventSourceResponse:
    ticker = ticker.upper().strip()
    if not ticker or not ticker.isalnum() or len(ticker) > 8:
        raise HTTPException(status_code=400, detail="invalid ticker")

    run_id = uuid.uuid4()
    started = datetime.now(timezone.utc)

    async with session_scope() as s:
        s.add(Run(id=run_id, ticker=ticker, status="running", started_at=started))

    async def event_stream() -> AsyncIterator[dict[str, Any]]:
        # token deltas flow through this queue from the writer node into the SSE loop
        token_queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)

        async def emit_token(delta: str) -> None:
            await token_queue.put(delta)

        final_state: dict[str, Any] = {}

        async def run_graph() -> None:
            try:
                async for node, partial in stream_run(ticker, emit_token=emit_token):
                    final_state.update(partial)
                    yield_event = {
                        "event": "agent_done",
                        "data": json.dumps(
                            {"agent": node, "summary": _agent_summary(node, partial)},
                            default=str,
                        ),
                    }
                    await sse_queue.put(yield_event)
                    await sse_queue.put(
                        {"event": "agent_start_next", "data": json.dumps({"after": node})}
                    )
            except Exception as e:  # noqa: BLE001
                log.exception("graph.error ticker=%s", ticker)
                await sse_queue.put({"event": "error", "data": json.dumps({"error": str(e)})})
            finally:
                await token_queue.put(None)  # signal token consumer
                await sse_queue.put(None)  # signal main loop

        sse_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=512)

        # Drain the token queue concurrently — each token becomes an SSE event.
        async def drain_tokens() -> None:
            while True:
                delta = await token_queue.get()
                if delta is None:
                    return
                await sse_queue.put(
                    {"event": "token", "data": json.dumps({"delta": delta})}
                )

        # Kick off both producers
        yield {"event": "agent_start", "data": json.dumps({"agent": "market"})}
        graph_task = asyncio.create_task(run_graph())
        token_task = asyncio.create_task(drain_tokens())

        while True:
            evt = await sse_queue.get()
            if evt is None:
                break
            yield evt

        await asyncio.gather(graph_task, token_task, return_exceptions=True)

        # Persist final memo
        memo = final_state.get("final_memo") or {}
        finished = datetime.now(timezone.utc)
        report_id = uuid.uuid4()

        try:
            async with session_scope() as s:
                s.add(
                    Report(
                        id=report_id,
                        run_id=run_id,
                        ticker=ticker,
                        memo=memo,
                        created_at=finished,
                    )
                )
                run = await s.get(Run, run_id)
                if run:
                    run.status = "done"
                    run.finished_at = finished
                    run.duration_ms = int((finished - started).total_seconds() * 1000)
                    errors = final_state.get("errors") or []
                    run.state_snapshot = {
                        "errors": [e.model_dump() if hasattr(e, "model_dump") else e for e in errors],
                    }
        except Exception as e:  # noqa: BLE001
            log.exception("persist.error")
            yield {"event": "error", "data": json.dumps({"error": f"persist: {e}"})}
            return

        yield {
            "event": "final",
            "data": json.dumps(
                {"report_id": str(report_id), "memo": memo},
                default=str,
            ),
        }

    return EventSourceResponse(event_stream())
