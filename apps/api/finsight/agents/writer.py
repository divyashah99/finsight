"""Writer agent — single structured-output pass.

Calls `chat_structured` once to produce a strict-JSON `Memo` directly from
the evidence, skipping the previous markdown-then-structure two-pass.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from finsight.agents.memo_schema import Memo
from finsight.agents.state import ResearchState
from finsight.logging_setup import get_logger
from finsight.prompts.writer import WRITER_SYSTEM, build_writer_user_message
from finsight.services import llm

log = get_logger(__name__)


async def run(state: ResearchState, emit_token: Any | None = None) -> dict[str, Any]:
    ticker = state["ticker"]
    log.info("writer.start ticker=%s", ticker)

    messages = [
        {"role": "system", "content": WRITER_SYSTEM},
        {"role": "user", "content": build_writer_user_message(dict(state), None)},
    ]

    try:
        memo = await llm.chat_structured(messages, schema=Memo, temperature=0.2)
        memo_dict = memo.model_dump()
        memo_dict["ticker"] = ticker
        memo_dict["as_of"] = date.today().isoformat()
    except Exception as e:  # noqa: BLE001
        log.warning("writer.failed error=%s", e)
        memo_dict = {
            "ticker": ticker,
            "as_of": date.today().isoformat(),
            "structurer_error": str(e),
        }

    log.info("writer.done ticker=%s", ticker)
    return {
        "draft_memo": memo_dict,
        "final_memo": memo_dict,
    }
