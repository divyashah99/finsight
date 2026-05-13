"""Writer-agent prompt templates.

We stream a markdown narrative first (for UX), then a second model call produces
the strict-JSON `Memo` for storage and downstream critique. Both calls share
the same evidence context, so the structured version doesn't drift from what
the user saw.
"""

from __future__ import annotations

import json
from typing import Any

WRITER_SYSTEM = """You are a senior equity research analyst.
You produce concise, evidence-grounded investment memos for a portfolio manager.

Rules:
- Be specific. Cite numbers, not adjectives.
- Acknowledge uncertainty and missing data; never fabricate metrics.
- When you use a fact from an SEC filing, reference its citation number in square brackets, e.g. [2].
- Structure: Headline, Thesis, Bull Case, Bear Case, Key Metrics, Risks, Verdict.
- Output GitHub-flavored markdown."""

STRUCTURER_SYSTEM = """You convert investment-memo markdown into a strict JSON object.

Rules:
- Use ONLY facts that appear in the provided markdown or evidence — never invent.
- `citation_ids` are 1-based indices into the SEC evidence list. Empty list if a
  claim is not from a filing.
- `recommendation` must be one of: buy, hold, sell, no_opinion.
- `conviction` is an integer 1-5 reflecting how strong the case is."""


def _dump(model: Any) -> str:
    if model is None:
        return "null"
    if hasattr(model, "model_dump"):
        return json.dumps(model.model_dump(), indent=2, default=str)
    return json.dumps(model, indent=2, default=str)


def _format_citations(sec_evidence: Any) -> str:
    if sec_evidence is None or not getattr(sec_evidence, "citations", None):
        return "(no SEC citations available)"
    lines = []
    for i, c in enumerate(sec_evidence.citations, start=1):
        lines.append(
            f"[{i}] {c.form_type} · {c.section or '?'} · filed {c.filed_date or '?'}\n"
            f"    {c.excerpt[:300]}"
        )
    return "\n\n".join(lines)


def build_writer_user_message(state: dict[str, Any], critique: str | None = None) -> str:
    ticker = state.get("ticker", "?")
    market = state.get("market")
    quant = state.get("quant")
    news = state.get("news")
    sec = state.get("sec")

    revision_note = ""
    if critique:
        revision_note = (
            "\n\n# Critic feedback (incorporate this revision)\n"
            f"{critique}\n"
            "Rewrite the memo addressing the feedback above.\n"
        )

    return f"""Write an investment memo for **{ticker}**.

# Fundamentals
{_dump(market)}

# Technical signals
{_dump(quant)}

# News sentiment
{_dump(news)}

# SEC citations (reference as [1], [2], etc.)
{_format_citations(sec)}
{revision_note}
Produce the memo now. ~400-600 words."""


def build_structurer_user_message(markdown: str, state: dict[str, Any]) -> str:
    sec = state.get("sec")
    return f"""Convert the following investment-memo markdown into strict JSON
matching the provided schema. Preserve all numeric claims.

# Memo (markdown)
{markdown}

# SEC citation list (for resolving [N] references)
{_format_citations(sec)}
"""


CRITIC_SYSTEM = """You are a skeptical senior PM reviewing a junior analyst's memo.

You produce a Critique JSON. Set `needs_revision=true` ONLY if at least one of:
- A material fact contradicts the provided fundamentals/news/SEC evidence
- The memo claims a specific number that isn't in the evidence
- Bull and bear cases are imbalanced (e.g. fewer than 2 substantive points on a side)
- A high-severity risk from the SEC evidence is missing entirely

Otherwise set `needs_revision=false`. Don't nitpick wording or style.
Your `suggestion` (if revising) should be 1-3 sentences telling the writer
exactly what to fix."""


def build_critic_user_message(memo: Any, state: dict[str, Any]) -> str:
    market = state.get("market")
    quant = state.get("quant")
    news = state.get("news")
    sec = state.get("sec")
    return f"""Review this memo against the evidence below.

# Memo (structured)
{_dump(memo)}

# Fundamentals
{_dump(market)}

# Technicals
{_dump(quant)}

# News
{_dump(news)}

# SEC citations
{_format_citations(sec)}
"""
