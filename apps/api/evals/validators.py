"""Deterministic memo validators.

Pure functions returning a list of human-readable problems (empty = valid). They
encode constraints the `Memo` schema does NOT enforce (e.g. headline length, and
citation-id range against the actual evidence). Reused by both the pytest suite
and the LangSmith code-evaluators, so the two never drift.
"""

from __future__ import annotations

from datetime import date
from typing import Any

_RECOMMENDATIONS = {"buy", "hold", "sell", "no_opinion"}
_SEVERITIES = {"low", "medium", "high"}
_CITED_SECTIONS = ("thesis_bull", "thesis_bear", "risks")


def validate_memo(memo: dict[str, Any]) -> list[str]:
    """Structural constraints on a synthesized memo dict."""
    errs: list[str] = []

    if "structurer_error" in memo:
        return [f"synthesis failed: {memo['structurer_error']}"]

    headline = memo.get("headline") or ""
    if len(headline) > 120:
        errs.append(f"headline is {len(headline)} chars (max 120)")

    conv = memo.get("conviction")
    if not (isinstance(conv, int) and 1 <= conv <= 5):
        errs.append(f"conviction {conv!r} not an int in 1..5")

    rec = memo.get("recommendation")
    if rec not in _RECOMMENDATIONS:
        errs.append(f"recommendation {rec!r} not in {sorted(_RECOMMENDATIONS)}")

    for side in ("thesis_bull", "thesis_bear"):
        n = len(memo.get(side) or [])
        if not (2 <= n <= 5):
            errs.append(f"{side} has {n} items (expected 2..5)")

    risks = memo.get("risks") or []
    if not (1 <= len(risks) <= 6):
        errs.append(f"risks has {len(risks)} items (expected 1..6)")
    for r in risks:
        if r.get("severity") not in _SEVERITIES:
            errs.append(f"risk severity {r.get('severity')!r} invalid")

    as_of = memo.get("as_of")
    try:
        date.fromisoformat(str(as_of))
    except (TypeError, ValueError):
        errs.append(f"as_of {as_of!r} is not an ISO date")

    return errs


def validate_citations(memo: dict[str, Any], citations: list[Any]) -> list[str]:
    """Every citation_id must be a 1-based index into `citations` (no zeros,
    no out-of-range, no duplicates within a single claim)."""
    n = len(citations)
    errs: list[str] = []
    for section in _CITED_SECTIONS:
        for i, item in enumerate(memo.get(section) or []):
            ids = item.get("citation_ids") or []
            if len(ids) != len(set(ids)):
                errs.append(f"{section}[{i}] has duplicate citation_ids {ids}")
            for cid in ids:
                if not isinstance(cid, int) or cid < 1 or cid > n:
                    errs.append(f"{section}[{i}] citation_id {cid!r} out of range 1..{n}")
    return errs
