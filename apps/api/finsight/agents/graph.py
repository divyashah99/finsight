"""LangGraph orchestrator.

Topology:

         ┌──────► market ──► quant ──┐
    START┤                            ├──► writer ──► critic ──┐
         ├──────► news ──────────────┤                  │      │
         └──────► sec  ──────────────┘                  │  approved
                                                        │      ▼
                                                    needs_     END
                                                   revision
                                                        │
                                                        └──► writer (cap: 1)

Three parallel fan-out branches → writer (barrier join). The critic can demand
exactly one revision before we accept the memo. The revision cap lives in the
conditional-edge function so it's impossible to bypass.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable, Literal

from langgraph.graph import END, START, StateGraph

from finsight.agents import critic, market, news, quant, sec, writer
from finsight.agents.state import ResearchState
from finsight.logging_setup import get_logger

log = get_logger(__name__)


TokenEmitter = Callable[[str], Awaitable[None]]

MAX_REVISIONS = 1


def _route_after_critic(state: ResearchState) -> Literal["writer", "__end__"]:
    """Decide whether to loop back for a revision or stop.

    The cap is enforced HERE — never inside the critic — so the critic can
    keep flagging issues without being able to spin the loop.
    """
    critique = state.get("critique") or {}
    revisions_done = state.get("revision_count", 0)
    needs = bool(critique.get("needs_revision"))
    if needs and revisions_done <= MAX_REVISIONS:
        log.info("critic.route -> writer (revision=%d)", revisions_done)
        return "writer"
    log.info("critic.route -> END (needs=%s revisions=%d)", needs, revisions_done)
    return "__end__"


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

    async def _critic_node(state: ResearchState) -> dict[str, Any]:
        return await critic.run(state)

    for name, fn in [
        ("market", _market_node),
        ("quant", _quant_node),
        ("news", _news_node),
        ("sec", _sec_node),
        ("writer", _writer_node),
        ("critic", _critic_node),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "market")
    g.add_edge(START, "news")
    g.add_edge(START, "sec")

    g.add_edge("market", "quant")

    g.add_edge("quant", "writer")
    g.add_edge("news", "writer")
    g.add_edge("sec", "writer")

    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", _route_after_critic, {"writer": "writer", "__end__": END})

    return g.compile()


async def stream_run(ticker: str, emit_token: TokenEmitter | None) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    graph = build_graph(emit_token=emit_token)
    initial: ResearchState = {"ticker": ticker, "revision_count": 0, "errors": []}
    async for event in graph.astream(initial, stream_mode="updates"):
        for node_name, partial in event.items():
            log.info("graph.update node=%s", node_name)
            yield node_name, partial
