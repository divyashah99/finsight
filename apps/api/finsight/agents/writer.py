"""Writer agent — two-call pipeline.

Call 1: stream a markdown narrative so the UI can render tokens live.
Call 2: convert the markdown into a strict-JSON `Memo` for storage/critique.

A second invocation triggered by the critic re-uses the same writer but injects
critic feedback into the prompt.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from finsight.agents.memo_schema import Memo
from finsight.agents.state import ResearchState
from finsight.logging_setup import get_logger
from finsight.prompts.writer import (
    STRUCTURER_SYSTEM,
    WRITER_SYSTEM,
    build_structurer_user_message,
    build_writer_user_message,
)
from finsight.services import llm

log = get_logger(__name__)


async def run(state: ResearchState, emit_token: Any | None = None) -> dict[str, Any]:
    ticker = state["ticker"]
    revision = state.get("revision_count", 0)
    critique_suggestion = None
    if revision > 0 and state.get("critique"):
        critique_suggestion = state["critique"].get("suggestion")

    log.info("writer.start ticker=%s revision=%d", ticker, revision)

    # ─── Pass 1: streaming markdown ─────────────────────────────────────────
    messages = [
        {"role": "system", "content": WRITER_SYSTEM},
        {
            "role": "user",
            "content": build_writer_user_message(dict(state), critique_suggestion),
        },
    ]

    if emit_token is None:
        markdown = await llm.chat(messages, temperature=0.3, max_tokens=1200)
    else:
        chunks: list[str] = []
        async for delta in llm.stream_chat(messages, temperature=0.3, max_tokens=1200):
            chunks.append(delta)
            await emit_token(delta)
        markdown = "".join(chunks)

    # ─── Pass 2: structured memo ────────────────────────────────────────────
    structurer_messages = [
        {"role": "system", "content": STRUCTURER_SYSTEM},
        {"role": "user", "content": build_structurer_user_message(markdown, dict(state))},
    ]
    try:
        memo = await llm.chat_structured(structurer_messages, schema=Memo, temperature=0.0)
        # Force fields the model can get wrong
        memo_dict = memo.model_dump()
        memo_dict["ticker"] = ticker
        memo_dict["as_of"] = date.today().isoformat()
        memo_dict["markdown"] = markdown  # keep the human-readable version too
    except Exception as e:  # noqa: BLE001
        log.warning("writer.structurer_failed error=%s", e)
        memo_dict = {
            "ticker": ticker,
            "as_of": date.today().isoformat(),
            "markdown": markdown,
            "structurer_error": str(e),
        }

    log.info("writer.done ticker=%s revision=%d", ticker, revision)
    return {
        "draft_memo": memo_dict,
        "final_memo": memo_dict,
    }
