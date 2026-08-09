"""Conversational analyst agent — real LLM-driven tool use + memory.

The initial memo is produced by a deterministic LangGraph DAG (see `graph.py`).
Follow-up questions, however, are open-ended — you can't hardcode a graph for
"what did their latest income statement show?" vs "summarize the supply-chain
risks". So follow-ups are handled by a genuine tool-calling agent:

- **Tools** (the LLM decides which to call):
    * the Alpha Vantage MCP server's tools, loaded via `langchain-mcp-adapters`
      (this is what finally exercises `av_income_statement`, which the DAG never
      calls), and
    * `search_filings` — hybrid (dense + BM25) retrieval + LLM rerank over the
      company's ingested SEC filings.
- **Memory**: an `AsyncPostgresSaver` checkpointer keyed by `thread_id`, so the
  conversation persists across turns and process restarts.

Built with LangChain 1.0's `create_agent` (the `create_react_agent` prebuilt is
deprecated).
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from finsight.logging_setup import get_logger
from finsight.services import llm, reranker, vectorstore
from finsight.settings import settings

log = get_logger(__name__)

_SYSTEM = (
    "You are FinSight's equity-research analyst assistant. You are answering "
    "follow-up questions about the investment memo already generated for {ticker}.\n\n"
    "Guidelines:\n"
    "- Be concise, factual, and specific. Prefer concrete numbers.\n"
    "- Use the available tools to fetch fresh market data (fundamentals, price "
    "history, income statement, news sentiment) or to search {ticker}'s SEC "
    "filings (`search_filings`) whenever the question needs evidence you don't "
    "already have.\n"
    "- When you cite something from a filing, mention the form and section.\n"
    "- If the data isn't available, say so plainly rather than guessing.\n\n"
    "Memo under discussion:\n{memo_summary}"
)


def make_search_filings_tool(ticker: str):
    """A `search_filings` tool bound to one ticker (hybrid retrieve + rerank)."""

    @tool
    async def search_filings(query: str) -> str:
        """Search this company's SEC filings (10-K/10-Q) for passages relevant to
        the query. Use for questions about risk factors, MD&A, business segments,
        legal proceedings, or anything stated in the filings. Returns the most
        relevant excerpts."""
        dense = (await llm.embed([query]))[0]
        sparse = (await vectorstore.sparse_embed([query], query=True))[0]
        pool = await vectorstore.hybrid_search(dense, sparse, ticker, limit=12)
        hits = await reranker.rerank(query, pool, top_n=5)
        if not hits:
            return "No relevant passages found in the ingested filings."
        blocks = []
        for i, h in enumerate(hits, start=1):
            p = h.payload
            excerpt = (p.get("text") or "")[:600].replace("\n", " ")
            blocks.append(
                f"[{i}] {p.get('form_type', '?')} · {p.get('section', '?')} · "
                f"filed {p.get('filed_date', '?')}\n{excerpt}"
            )
        return "\n\n".join(blocks)

    return search_filings


def summarize_memo(memo: dict[str, Any] | None) -> str:
    """Compact the stored memo into a system-prompt seed."""
    if not memo:
        return "(No memo found for this session.)"
    parts = [
        f"Ticker: {memo.get('ticker', '?')}",
        f"Recommendation: {memo.get('recommendation', '?')} "
        f"(conviction {memo.get('conviction', '?')}/5)",
        f"Headline: {memo.get('headline', '')}",
    ]
    bull = memo.get("thesis_bull") or []
    bear = memo.get("thesis_bear") or []
    if bull:
        parts.append("Bull points: " + "; ".join(a.get("claim", "") for a in bull[:3]))
    if bear:
        parts.append("Bear points: " + "; ".join(a.get("claim", "") for a in bear[:3]))
    risks = memo.get("risks") or []
    if risks:
        parts.append("Top risks: " + "; ".join(r.get("title", "") for r in risks[:3]))
    return "\n".join(parts)


def build_analyst(ticker: str, tools: list, checkpointer, memo: dict[str, Any] | None):
    """Compile a tool-calling analyst agent for one ticker.

    `tools` should already include the loaded MCP tools; we append the
    ticker-bound `search_filings` tool here.
    """
    model = ChatOpenAI(
        model=settings.openai_chat_model,
        api_key=settings.openai_api_key,
        temperature=0.2,
    )
    all_tools = [*tools, make_search_filings_tool(ticker)]
    system_prompt = _SYSTEM.format(ticker=ticker, memo_summary=summarize_memo(memo))
    return create_agent(
        model=model,
        tools=all_tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )
