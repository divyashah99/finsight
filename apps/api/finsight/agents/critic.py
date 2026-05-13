"""Critic agent.

Reviews the structured draft memo against the original evidence and decides
whether a revision is warranted. Output is a strict-JSON `Critique`.

The graph hard-caps revisions at 1 (see `should_revise` in graph.py) to keep
total cost bounded — without that, two LLMs could ping-pong forever.
"""

from __future__ import annotations

from typing import Any

from finsight.agents.memo_schema import Critique
from finsight.agents.state import ResearchState
from finsight.logging_setup import get_logger
from finsight.prompts.writer import CRITIC_SYSTEM, build_critic_user_message
from finsight.services import llm

log = get_logger(__name__)


async def run(state: ResearchState) -> dict[str, Any]:
    ticker = state["ticker"]
    memo = state.get("draft_memo") or {}

    log.info("critic.start ticker=%s", ticker)

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": build_critic_user_message(memo, dict(state))},
    ]
    try:
        critique = await llm.chat_structured(messages, schema=Critique, temperature=0.0)
        critique_dict = critique.model_dump()
    except Exception as e:  # noqa: BLE001
        log.warning("critic.failed error=%s", e)
        critique_dict = {
            "needs_revision": False,
            "issues": [],
            "confidence": 3,
            "suggestion": None,
            "error": str(e),
        }

    log.info(
        "critic.done ticker=%s needs_revision=%s confidence=%s",
        ticker,
        critique_dict.get("needs_revision"),
        critique_dict.get("confidence"),
    )
    return {
        "critique": critique_dict,
        "revision_count": state.get("revision_count", 0) + 1,
    }
