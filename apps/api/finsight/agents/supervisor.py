"""Supervisor — the agentic router.

Given the research request and the findings gathered so far, an LLM decides the
next action: dispatch one of the specialist sub-agents (with a focused task) or
stop and synthesize the memo. This is the model-driven control flow that makes
the system a real agent rather than a fixed pipeline. It can react to what it
learns (e.g. news surfaces a lawsuit → dispatch the filings analyst for legal
proceedings).

The dispatch *cap* is enforced by the graph, not here — an LLM that can decide
its own retry/loop budget is a liability.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from finsight.agents.state import ResearchState
from finsight.logging_setup import get_logger
from finsight.services import llm

log = get_logger(__name__)

NextAction = Literal["fundamentals", "technicals", "news", "filings", "synthesize"]


class RoutingDecision(BaseModel):
    next: NextAction = Field(description="Which specialist to dispatch next, or 'synthesize' to finish")
    task: str = Field(description="Concrete instruction for the chosen specialist (ignored if synthesize)")
    reason: str = Field(description="One short sentence: why this is the right next step")


_SYSTEM = """You are the lead of an equity-research team. Your job is to decide the
next step in researching a stock, then eventually to stop and write the memo.

Specialists you can dispatch (one at a time):
- fundamentals: valuation, profitability, growth, income statement
- technicals: price momentum, RSI/MACD, volatility, trend
- news: recent news sentiment and material headlines
- filings: searches the SEC 10-K/10-Q (risk factors, MD&A, legal, segments)

Guidance:
- Gather a well-rounded evidence base before synthesizing — normally you want at
  least fundamentals, news, and filings; add technicals when momentum matters.
- REACT to findings: if news or fundamentals surface something material (a
  lawsuit, a guidance cut, a segment risk), dispatch `filings` with a focused
  task to dig into it, or re-dispatch a specialist with a sharper question.
- Don't repeat a specialist with the same task; only re-dispatch to go deeper.
- Choose `synthesize` once you have enough to write a balanced bull/bear memo.
Return the decision as structured output."""


def _findings_digest(state: ResearchState) -> str:
    findings = state.get("findings") or []
    if not findings:
        return "(no findings yet)"
    return "\n\n".join(f"[{f.agent}] task: {f.task}\n{f.summary}" for f in findings)


async def route(state: ResearchState) -> RoutingDecision:
    ticker = state["ticker"]
    ran = sorted({f.agent for f in (state.get("findings") or [])})
    user = (
        f"Ticker under research: {ticker}\n"
        f"Specialists already run: {', '.join(ran) or 'none'}\n\n"
        f"Findings so far:\n{_findings_digest(state)}\n\n"
        "Decide the next action."
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]
    decision = await llm.chat_structured(messages, schema=RoutingDecision, temperature=0.1)
    log.info("supervisor.route next=%s reason=%s", decision.next, decision.reason)
    return decision
