"""LangGraph orchestrator.

Topology:

         ┌──────► market ──► quant ──┐
    START┤                            ├──► writer ──► END
         ├──────► news ──────────────┤
         └──────► sec  ──────────────┘

Three parallel fan-out branches → writer (barrier join). Critic loop removed
to reduce latency for the demo.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable

from langgraph.graph import END, START, StateGraph

from finsight.agents import market, news, quant, sec, writer
from finsight.agents.state import ResearchState
from finsight.logging_setup import get_logger

log = get_logger(__name__)


TokenEmitter = Callable[[str], Awaitable[None]]


def build_graph(emit_token: TokenEmitter | None = None):
    g: StateGraph = StateGraph(ResearchState)

    async def _market_node(state: ResearchState) -> dict[str, Any]:
        return await market.run(state)

    async def _quant_node(state: ResearchState) -> dict[str, Any]:
        return await quant.run(state)

    async def _news_node(state: ResearchState) -> dict[str, Any]:
        return await news.run(state)

    async def _sec_node(state: ResearchState) -> dict[str, Any]:
        return await sec.run(state)

    async def _writer_node(state: ResearchState) -> dict[str, Any]:
        return await writer.run(state, emit_token=emit_token)

    for name, fn in [
        ("market", _market_node),
        ("quant", _quant_node),
        ("news", _news_node),
        ("sec", _sec_node),
        ("writer", _writer_node),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "market")
    g.add_edge(START, "news")
    g.add_edge(START, "sec")

    g.add_edge("market", "quant")

    g.add_edge("quant", "writer")
    g.add_edge("news", "writer")
    g.add_edge("sec", "writer")

    g.add_edge("writer", END)

    return g.compile()


async def stream_run(ticker: str, emit_token: TokenEmitter | None) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    graph = build_graph(emit_token=emit_token)
    initial: ResearchState = {"ticker": ticker, "revision_count": 0, "errors": []}
    async for event in graph.astream(initial, stream_mode="updates"):
        for node_name, partial in event.items():
            log.info("graph.update node=%s", node_name)
            yield node_name, partial
