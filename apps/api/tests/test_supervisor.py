"""Supervisor routing tests — the agentic control flow.

Stub the LLM so these run with no network/API key.
"""

from __future__ import annotations

import pytest

from finsight.agents import graph, supervisor
from finsight.agents.state import Finding
from finsight.agents.supervisor import RoutingDecision


@pytest.mark.asyncio
async def test_route_returns_decision(monkeypatch):
    async def fake_chat_structured(messages, schema, **kwargs):
        assert schema is RoutingDecision
        return RoutingDecision(next="filings", task="risk factors", reason="need risks")

    monkeypatch.setattr(supervisor.llm, "chat_structured", fake_chat_structured)

    decision = await supervisor.route({"ticker": "AAPL", "findings": []})
    assert decision.next == "filings"
    assert decision.task == "risk factors"


@pytest.mark.asyncio
async def test_supervisor_node_dispatches_and_increments(monkeypatch):
    async def fake_route(state):
        return RoutingDecision(next="fundamentals", task="valuation", reason="start with fundamentals")

    monkeypatch.setattr(graph.supervisor, "route", fake_route)

    out = await graph._supervisor_node({"ticker": "AAPL", "dispatch_count": 0, "findings": []})
    assert out["next_agent"] == "fundamentals"
    assert out["next_task"] == "valuation"
    assert out["dispatch_count"] == 1


@pytest.mark.asyncio
async def test_supervisor_node_forces_synthesize_at_cap(monkeypatch):
    called = False

    async def fake_route(state):  # should NOT be called once capped
        nonlocal called
        called = True
        return RoutingDecision(next="news", task="x", reason="y")

    monkeypatch.setattr(graph.supervisor, "route", fake_route)

    out = await graph._supervisor_node(
        {"ticker": "AAPL", "dispatch_count": graph.MAX_DISPATCHES, "findings": []}
    )
    assert out["next_agent"] == "synthesize"
    assert called is False  # cap short-circuits before routing


@pytest.mark.asyncio
async def test_supervisor_node_passes_through_synthesize(monkeypatch):
    async def fake_route(state):
        return RoutingDecision(next="synthesize", task="", reason="have enough evidence")

    monkeypatch.setattr(graph.supervisor, "route", fake_route)

    out = await graph._supervisor_node(
        {"ticker": "AAPL", "dispatch_count": 2, "findings": [Finding(agent="news", task="t", summary="s")]}
    )
    assert out["next_agent"] == "synthesize"
    # no dispatch increment when synthesizing
    assert "dispatch_count" not in out
