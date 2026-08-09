"""Specialist sub-agents.

Each specialist is a LangChain `create_agent` tool-calling agent given a focused
system prompt and a subset of the research tools. The supervisor dispatches one
at a time with a task; the specialist decides which of its tools to call, then
returns a concise finding. The FilingsAnalyst additionally harvests the SEC
citations its `search_filings` calls surfaced (via the ContextVar sink).

No checkpointer here — specialists are stateless per dispatch; memory lives at
the run level (findings accumulate in graph state) and the follow-up chat
(`agents/analyst.py`) keeps its own thread memory.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from finsight.agents.state import SECCitation
from finsight.logging_setup import get_logger
from finsight.settings import settings
from finsight.tools.research_tools import reset_citation_sink, tools_for

log = get_logger(__name__)

SPECIALISTS = ("fundamentals", "technicals", "news", "filings")

_PROMPTS: dict[str, str] = {
    "fundamentals": (
        "You are a fundamentals analyst covering {ticker}. Use your tools to pull "
        "the company's fundamentals and, when useful, its income statement. "
        "Report a concise, numbers-first finding (valuation, profitability, growth). "
        "Do not speculate beyond the data; if something is unavailable, say so."
    ),
    "technicals": (
        "You are a technical analyst covering {ticker}. Use your tools to inspect "
        "recent price action and compute technical indicators (RSI, MACD, "
        "volatility, returns, moving-average trend). Report a concise finding on "
        "momentum and trend. State plainly if there's insufficient history."
    ),
    "news": (
        "You are a news analyst covering {ticker}. Use your tool to gather recent "
        "news sentiment and headlines. Report the aggregate mood and the few most "
        "material items (catalysts, controversies). Be concise and factual."
    ),
    "filings": (
        "You are a filings analyst covering {ticker}. Use `search_filings` to find "
        "relevant passages in the company's SEC 10-K/10-Q. Run focused searches for "
        "whatever the task asks (e.g. risk factors, MD&A, legal proceedings, a "
        "specific topic). Report a concise finding and, when you cite something, "
        "name the form and section."
    ),
}


def _model() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        temperature=0.1,
    )


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


async def run_specialist(name: str, ticker: str, task: str) -> tuple[str, list[SECCitation]]:
    """Run one specialist for a task; return (finding_text, citations)."""
    tools = tools_for(ticker)[name]
    sink: list[SECCitation] = reset_citation_sink() if name == "filings" else []

    agent = create_agent(
        model=_model(),
        tools=tools,
        system_prompt=_PROMPTS[name].format(ticker=ticker),
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
    summary = _text(result["messages"][-1].content).strip()
    citations = list(sink) if name == "filings" else []
    log.info("specialist.done name=%s citations=%d", name, len(citations))
    return summary, citations
