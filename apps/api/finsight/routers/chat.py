"""POST /research/{ticker}/chat — follow-up Q&A on a generated memo.

Streams the analyst agent's answer as SSE. The agent (see `agents/analyst.py`)
decides which tools to call and remembers the conversation via the LangGraph
Postgres checkpointer, keyed by `thread_id`.

Event schema:
    event: tool    data: {"name": "yf_income_statement"}   # a tool was invoked
    event: token   data: {"delta": "..."}                   # answer token
    event: done    data: {}
    event: error   data: {"error": "..."}
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from finsight.agents.analyst import build_analyst
from finsight.db.client import session_scope
from finsight.db.models import Report
from finsight.logging_setup import get_logger
from finsight.tools.mcp_client import mcp_session

log = get_logger(__name__)
router = APIRouter(prefix="/research", tags=["chat"])


class ChatRequest(BaseModel):
    thread_id: str
    message: str


async def _load_memo(thread_id: str) -> dict[str, Any] | None:
    async with session_scope() as s:
        rpt = (
            await s.execute(
                select(Report)
                .where(Report.thread_id == thread_id)
                .order_by(Report.created_at.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        return rpt.memo if rpt else None


def _chunk_text(content: Any) -> str:
    """AIMessageChunk.content is usually a str; some providers use content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


@router.post("/{ticker}/chat")
async def chat(ticker: str, body: ChatRequest, request: Request) -> EventSourceResponse:
    ticker = ticker.upper().strip()
    if not ticker.isalnum() or len(ticker) > 8:
        raise HTTPException(status_code=400, detail="invalid ticker")
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="empty message")

    checkpointer = getattr(request.app.state, "checkpointer", None)
    if checkpointer is None:
        raise HTTPException(status_code=503, detail="chat unavailable: checkpointer not initialized")

    memo = await _load_memo(body.thread_id)
    config = {"configurable": {"thread_id": body.thread_id}}

    async def event_stream() -> AsyncIterator[dict[str, Any]]:
        try:
            async with mcp_session("yfinance") as mcp:
                from langchain_mcp_adapters.tools import load_mcp_tools

                mcp_tools = await load_mcp_tools(mcp.session)
                agent = build_analyst(ticker, mcp_tools, checkpointer, memo)

                async for event in agent.astream_events(
                    {"messages": [{"role": "user", "content": body.message}]},
                    config=config,
                    version="v2",
                ):
                    kind = event.get("event")
                    if kind == "on_tool_start":
                        yield {"event": "tool", "data": json.dumps({"name": event.get("name", "")})}
                    elif kind == "on_chat_model_stream":
                        text = _chunk_text(event["data"]["chunk"].content)
                        if text:
                            yield {"event": "token", "data": json.dumps({"delta": text})}
            yield {"event": "done", "data": "{}"}
        except Exception as e:  # noqa: BLE001
            log.exception("chat.error ticker=%s", ticker)
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_stream())
