"""Run the FinSight evaluation suite against LangSmith.

Platform-first: creates/updates LangSmith datasets from evals/datasets/*.json and
runs `aevaluate()` for three targets — memo, RAG retrieval, analyst tool-use —
attaching the evaluators in evals/evaluators.py.

    python -m evals.run_evals [memo|rag|analyst|all]

Requires LANGSMITH_API_KEY (+ OPENAI_API_KEY, and a running Qdrant with indexed
filings for the rag/memo targets, plus MCP/Yahoo Finance for analyst/memo).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from evals import evaluators as ev

_DATA = Path(__file__).parent / "datasets"


# ─── Targets ───────────────────────────────────────────────────────────────


async def memo_target(inputs: dict[str, Any]) -> dict[str, Any]:
    from finsight.agents.graph import build_graph

    ticker = inputs["ticker"]
    graph = build_graph()
    initial = {
        "ticker": ticker,
        "request": f"Research {ticker} and produce an investment memo.",
        "dispatch_count": 0,
        "findings": [],
        "citations": [],
        "errors": [],
    }
    res = await graph.ainvoke(initial)
    findings = res.get("findings") or []
    citations = res.get("citations") or []
    return {
        "memo": res.get("final_memo") or {},
        "citations": [c.model_dump() if hasattr(c, "model_dump") else c for c in citations],
        "findings": [f.model_dump() if hasattr(f, "model_dump") else f for f in findings],
        "trajectory": [getattr(f, "agent", None) for f in findings],
        "terminated": bool(res.get("final_memo")),
    }


async def rag_target(inputs: dict[str, Any]) -> dict[str, Any]:
    from finsight.services import llm, vectorstore

    q, ticker = inputs["query"], inputs["ticker"]
    dense = (await llm.embed([q]))[0]
    sparse = (await vectorstore.sparse_embed([q], query=True))[0]
    hybrid = await vectorstore.hybrid_search(dense, sparse, ticker, limit=8)
    dense_hits = await vectorstore.search(dense, ticker, limit=8)

    def _p(hs):
        return [{"section": h.payload.get("section"), "text": h.payload.get("text")} for h in hs]

    return {"hybrid": _p(hybrid), "dense": _p(dense_hits)}


async def analyst_target(inputs: dict[str, Any]) -> dict[str, Any]:
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langgraph.checkpoint.memory import InMemorySaver

    from finsight.agents.analyst import build_analyst
    from finsight.tools.mcp_client import mcp_session

    ticker, question = inputs["ticker"], inputs["question"]
    async with mcp_session("yfinance") as mcp:
        tools = await load_mcp_tools(mcp.session)
        agent = build_analyst(ticker, tools, InMemorySaver(), memo=None)
        res = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": f"eval-{ticker}"}},
        )

    msgs = res.get("messages") or []
    tools_called: list[str] = []
    tool_outputs: list[str] = []
    for m in msgs:
        for tc in getattr(m, "tool_calls", None) or []:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                tools_called.append(name)
        if m.__class__.__name__ == "ToolMessage":
            tool_outputs.append(str(getattr(m, "content", "")))
    answer = ""
    if msgs:
        c = msgs[-1].content
        answer = c if isinstance(c, str) else str(c)
    return {"answer": answer, "tools_called": tools_called, "tool_outputs": "\n".join(tool_outputs)[:2000]}


# ─── Dataset + suite wiring ────────────────────────────────────────────────

_SUITES = {
    "memo": {
        "dataset": "finsight-memos",
        "file": "tickers.json",
        "target": memo_target,
        "evaluators": [ev.memo_schema_valid, ev.citations_in_range, ev.trajectory_coverage, ev.memo_faithfulness],
    },
    "rag": {
        "dataset": "finsight-rag",
        "file": "rag_gold.json",
        "target": rag_target,
        "evaluators": [ev.retrieval_precision_at_k, ev.hybrid_uplift, ev.rag_groundedness],
    },
    "analyst": {
        "dataset": "finsight-analyst",
        "file": "tool_selection.json",
        "target": analyst_target,
        "evaluators": [ev.tool_selection_match, ev.answer_groundedness],
    },
}


def _ensure_dataset(client, name: str, file: str) -> None:
    examples = json.loads((_DATA / file).read_text())
    try:
        client.read_dataset(dataset_name=name)
        return  # already exists — leave as-is
    except Exception:  # noqa: BLE001
        pass
    ds = client.create_dataset(dataset_name=name)
    client.create_examples(
        dataset_id=ds.id,
        inputs=[e["inputs"] for e in examples],
        outputs=[e.get("outputs") for e in examples],
    )
    print(f"created dataset {name} ({len(examples)} examples)")


async def _run_suite(name: str) -> None:
    from langsmith import Client, aevaluate

    suite = _SUITES[name]
    client = Client()
    _ensure_dataset(client, suite["dataset"], suite["file"])
    print(f"=== evaluating: {name} ===")
    await aevaluate(
        suite["target"],
        data=suite["dataset"],
        evaluators=suite["evaluators"],
        experiment_prefix=f"finsight-{name}",
        client=client,
        max_concurrency=1,  # bound cost + Yahoo Finance request rate
    )


async def main() -> None:
    if not os.getenv("LANGSMITH_API_KEY"):
        sys.exit("LANGSMITH_API_KEY not set — required for platform-first evaluation.")
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(_SUITES) if which == "all" else [which]
    for n in names:
        if n not in _SUITES:
            sys.exit(f"unknown suite {n!r}; choose from {list(_SUITES)} or 'all'")
        await _run_suite(n)


if __name__ == "__main__":
    asyncio.run(main())
