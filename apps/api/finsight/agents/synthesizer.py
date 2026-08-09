"""Synthesizer — turns the supervisor's gathered findings into the memo.

Once the supervisor decides it has enough evidence, this node makes a single
structured pass producing a strict-JSON `Memo`, grounded in the specialist
findings and the SEC citations collected along the way. Keeping the `Memo`
contract means the existing UI (memo viewer) and any future eval work unchanged.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from langsmith import traceable

from finsight.agents.memo_schema import Memo
from finsight.agents.state import ResearchState, SECEvidence
from finsight.logging_setup import get_logger
from finsight.prompts.writer import _format_citations
from finsight.services import llm

log = get_logger(__name__)

SYNTH_SYSTEM = """You are a senior equity research analyst. Using ONLY the research
findings and SEC citations provided, produce a structured investment memo.

Rules:
- Be specific and numbers-first; never fabricate metrics not present in the findings.
- Balance the bull and bear cases (>= 2 substantive points each).
- `citation_ids` are 1-based indices into the SEC citation list; attach them to
  any claim drawn from a filing, and leave empty for claims from fundamentals/
  technicals/news.
- `recommendation` is one of buy, hold, sell, no_opinion; `conviction` is 1-5.
- If evidence is thin or missing, reflect that in a lower conviction rather than
  inventing detail."""


def _dedupe_citations(citations: list) -> list:
    seen: set[tuple[str, str]] = set()
    out = []
    for c in citations:
        key = (c.accession, (c.excerpt or "")[:100])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _build_user_message(state: ResearchState, sec_evidence: SECEvidence) -> str:
    findings = state.get("findings") or []
    digest = "\n\n".join(f"## {f.agent}\n{f.summary}" for f in findings) or "(no findings)"
    return f"""Write an investment memo for **{state['ticker']}**.

# Research findings
{digest}

# SEC citations (reference as [1], [2], ...)
{_format_citations(sec_evidence)}

Produce the structured memo now."""


@traceable(run_type="chain", name="synthesizer")
async def run(state: ResearchState) -> dict[str, Any]:
    ticker = state["ticker"]
    log.info("synthesizer.start ticker=%s findings=%d", ticker, len(state.get("findings") or []))

    sec_evidence = SECEvidence(citations=_dedupe_citations(state.get("citations") or []))
    messages = [
        {"role": "system", "content": SYNTH_SYSTEM},
        {"role": "user", "content": _build_user_message(state, sec_evidence)},
    ]
    try:
        memo = await llm.chat_structured(messages, schema=Memo, temperature=0.2)
        memo_dict = memo.model_dump()
        memo_dict["ticker"] = ticker
        memo_dict["as_of"] = date.today().isoformat()
    except Exception as e:  # noqa: BLE001
        log.warning("synthesizer.failed error=%s", e)
        memo_dict = {"ticker": ticker, "as_of": date.today().isoformat(), "structurer_error": str(e)}

    log.info("synthesizer.done ticker=%s", ticker)
    return {"final_memo": memo_dict}
