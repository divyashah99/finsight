"""LangSmith evaluators for the agentic pipeline.

Classic `(run, example) -> dict` signature: `run.outputs` is what the target
returned, `example.outputs` is the reference. Code evaluators are deterministic;
the LLM-judge evaluators are async and use the existing `llm.chat_structured`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from evals.validators import validate_citations, validate_memo
from finsight.services import llm

_CORE_SPECIALISTS = {"fundamentals", "technicals", "news", "filings"}


def _outputs(run: Any) -> dict[str, Any]:
    return getattr(run, "outputs", None) or {}


def _reference(example: Any) -> dict[str, Any]:
    return getattr(example, "outputs", None) or {}


# ─── Code evaluators ───────────────────────────────────────────────────────


def memo_schema_valid(run, example) -> dict:
    memo = _outputs(run).get("memo") or {}
    errs = validate_memo(memo)
    return {"key": "memo_schema_valid", "score": 1.0 if not errs else 0.0, "comment": "; ".join(errs)}


def citations_in_range(run, example) -> dict:
    out = _outputs(run)
    errs = validate_citations(out.get("memo") or {}, out.get("citations") or [])
    return {"key": "citations_in_range", "score": 1.0 if not errs else 0.0, "comment": "; ".join(errs)}


def trajectory_coverage(run, example) -> dict:
    out = _outputs(run)
    dispatched = set(out.get("trajectory") or [])
    terminated = bool(out.get("terminated"))
    if not terminated:
        return {"key": "trajectory_coverage", "score": 0.0, "comment": "run did not terminate in a memo"}
    covered = len(dispatched & _CORE_SPECIALISTS) / len(_CORE_SPECIALISTS)
    return {
        "key": "trajectory_coverage",
        "score": covered,
        "comment": f"dispatched {sorted(dispatched)}",
    }


def _relevant_count(hits: list[dict], expected_section: str, terms: list[str]) -> int:
    terms = [t.lower() for t in terms]
    n = 0
    for h in hits:
        text = (h.get("text") or "").lower()
        section_ok = (h.get("section") == expected_section) if expected_section else True
        if section_ok and any(t in text for t in terms):
            n += 1
    return n


def retrieval_precision_at_k(run, example) -> dict:
    out, ref = _outputs(run), _reference(example)
    hits = out.get("hybrid") or []
    rel = _relevant_count(hits, ref.get("expected_section", ""), ref.get("expected_terms", []))
    score = rel / len(hits) if hits else 0.0
    return {"key": "retrieval_precision_at_k", "score": score, "comment": f"{rel}/{len(hits)} relevant"}


def hybrid_uplift(run, example) -> dict:
    out, ref = _outputs(run), _reference(example)
    sec, terms = ref.get("expected_section", ""), ref.get("expected_terms", [])
    h = _relevant_count(out.get("hybrid") or [], sec, terms)
    d = _relevant_count(out.get("dense") or [], sec, terms)
    return {
        "key": "hybrid_uplift",
        "score": 1.0 if h >= d else 0.0,
        "comment": f"hybrid relevant={h} vs dense={d}",
    }


def tool_selection_match(run, example) -> dict:
    out, ref = _outputs(run), _reference(example)
    expected = ref.get("expected_tool")
    called = out.get("tools_called") or []
    return {
        "key": "tool_selection_match",
        "score": 1.0 if expected in called else 0.0,
        "comment": f"expected {expected}, called {called}",
    }


# ─── LLM-judge evaluators (async) ──────────────────────────────────────────


class _Judgement(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="0=fails, 1=fully satisfies")
    reason: str


async def _judge(system: str, user: str, key: str) -> dict:
    try:
        j = await llm.chat_structured(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            schema=_Judgement,
            temperature=0.0,
        )
        return {"key": key, "score": float(j.score), "comment": j.reason}
    except Exception as e:  # noqa: BLE001
        return {"key": key, "score": 0.0, "comment": f"judge error: {e}"}


async def memo_faithfulness(run, example) -> dict:
    out = _outputs(run)
    memo = out.get("memo") or {}
    findings = out.get("findings") or []
    system = (
        "You are a rigorous equity-research reviewer. Score 0..1 whether the memo is "
        "faithful to the research findings: no fabricated numbers, no claims that "
        "contradict the findings, and a balanced bull/bear case. 1.0 = fully grounded "
        "and balanced; 0 = fabricated or contradictory."
    )
    user = f"# Findings\n{findings}\n\n# Memo\n{memo}"
    return await _judge(system, user, "memo_faithfulness")


async def rag_groundedness(run, example) -> dict:
    out, ref = _outputs(run), _reference(example)
    hits = out.get("hybrid") or []
    top = "\n\n".join((h.get("text") or "")[:500] for h in hits[:3])
    query = (getattr(example, "inputs", None) or {}).get("query", "")
    system = (
        "Score 0..1 whether the retrieved passages are relevant to and could help "
        "answer the query. 1.0 = directly on-topic; 0 = unrelated."
    )
    user = f"# Query\n{query}\n\n# Retrieved passages\n{top}"
    return await _judge(system, user, "rag_groundedness")


async def answer_groundedness(run, example) -> dict:
    out = _outputs(run)
    answer = out.get("answer", "")
    tool_outputs = out.get("tool_outputs", "")
    question = (getattr(example, "inputs", None) or {}).get("question", "")
    system = (
        "Score 0..1 whether the assistant's answer is grounded in the tool outputs it "
        "retrieved and actually addresses the question (or plainly says data is "
        "unavailable rather than inventing it). 1.0 = grounded and responsive."
    )
    user = f"# Question\n{question}\n\n# Tool outputs\n{tool_outputs}\n\n# Answer\n{answer}"
    return await _judge(system, user, "answer_groundedness")
