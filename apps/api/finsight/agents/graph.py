"""LangGraph orchestrator — supervisor + specialist sub-agents (agentic).

Topology:

    START → supervisor ─(routes)─▶ fundamentals ─┐
                    ▲               technicals ───┤
                    │               news ─────────┤─▶ back to supervisor
                    │               filings ──────┘
                    └────────────────────────────────
              supervisor ─(synthesize)─▶ synthesizer → END

The supervisor's LLM decides which specialist runs next and when to stop; the
loop is bounded by MAX_DISPATCHES in the graph (not left to the model). Each
specialist is a tool-calling sub-agent (`agents/specialists.py`).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langgraph.graph import END, START, StateGraph

from finsight.agents import specialists, supervisor, synthesizer
from finsight.agents.state import AgentError, Finding, ResearchState
from finsight.logging_setup import get_logger

log = get_logger(__name__)

MAX_DISPATCHES = 6  # hard cap on specialist dispatches per run


async def _supervisor_node(state: ResearchState) -> dict[str, Any]:
    count = state.get("dispatch_count", 0)
    if count >= MAX_DISPATCHES:
        return {"next_agent": "synthesize", "supervisor_reason": f"reached {MAX_DISPATCHES}-dispatch cap"}
    try:
        decision = await supervisor.route(state)
    except Exception as e:  # noqa: BLE001
        log.warning("supervisor.route_failed error=%s", e)
        return {"next_agent": "synthesize", "supervisor_reason": f"router error: {e}"}

    if decision.next == "synthesize":
        return {"next_agent": "synthesize", "supervisor_reason": decision.reason}
    return {
        "next_agent": decision.next,
        "next_task": decision.task,
        "supervisor_reason": decision.reason,
        "dispatch_count": count + 1,
    }


def _make_specialist_node(name: str):
    async def _node(state: ResearchState) -> dict[str, Any]:
        ticker = state["ticker"]
        task = state.get("next_task") or f"Analyze {ticker}."
        try:
            summary, citations = await specialists.run_specialist(name, ticker, task)
        except Exception as e:  # noqa: BLE001
            log.warning("specialist.failed name=%s error=%s", name, e)
            return {
                "findings": [Finding(agent=name, task=task, summary=f"(failed: {e})")],
                "errors": [AgentError(agent=name, error=str(e))],
            }
        out: dict[str, Any] = {"findings": [Finding(agent=name, task=task, summary=summary)]}
        if citations:
            out["citations"] = citations
        return out

    return _node


def _route_from_supervisor(state: ResearchState) -> str:
    nxt = state.get("next_agent")
    if nxt in specialists.SPECIALISTS:
        return nxt
    return "synthesize"


def build_graph():
    g: StateGraph = StateGraph(ResearchState)

    g.add_node("supervisor", _supervisor_node)
    for name in specialists.SPECIALISTS:
        g.add_node(name, _make_specialist_node(name))
    g.add_node("synthesizer", synthesizer.run)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {**{name: name for name in specialists.SPECIALISTS}, "synthesize": "synthesizer"},
    )
    for name in specialists.SPECIALISTS:
        g.add_edge(name, "supervisor")  # loop back for the next decision
    g.add_edge("synthesizer", END)

    return g.compile()


async def stream_run(ticker: str) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    graph = build_graph()
    initial: ResearchState = {
        "ticker": ticker,
        "request": f"Research {ticker} and produce an investment memo.",
        "dispatch_count": 0,
        "findings": [],
        "citations": [],
        "errors": [],
    }
    async for event in graph.astream(initial, stream_mode="updates"):
        for node_name, partial in event.items():
            log.info("graph.update node=%s", node_name)
            yield node_name, partial
