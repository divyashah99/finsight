"""Tests for the deterministic memo validators (no network)."""

from __future__ import annotations

from evals.validators import validate_citations, validate_memo


def _good_memo() -> dict:
    return {
        "ticker": "AAPL",
        "as_of": "2026-08-08",
        "recommendation": "hold",
        "conviction": 3,
        "headline": "Apple: solid fundamentals, regulatory overhang",
        "thesis_bull": [
            {"claim": "Strong cash position", "evidence": "…", "citation_ids": [1]},
            {"claim": "Services growth", "evidence": "…", "citation_ids": []},
        ],
        "thesis_bear": [
            {"claim": "Regulatory risk", "evidence": "…", "citation_ids": [2]},
            {"claim": "Competition", "evidence": "…", "citation_ids": []},
        ],
        "key_metrics": [{"name": "P/E", "value": "27"}],
        "catalysts": ["earnings"],
        "risks": [{"title": "Antitrust", "detail": "…", "severity": "high", "citation_ids": [2]}],
    }


def test_good_memo_passes():
    memo = _good_memo()
    assert validate_memo(memo) == []
    assert validate_citations(memo, citations=[{}, {}]) == []  # 2 citations → ids 1,2 valid


def test_headline_too_long_flagged():
    memo = _good_memo()
    memo["headline"] = "x" * 121
    assert any("headline" in e for e in validate_memo(memo))


def test_bad_conviction_and_recommendation():
    memo = _good_memo()
    memo["conviction"] = 9
    memo["recommendation"] = "strong_buy"
    errs = validate_memo(memo)
    assert any("conviction" in e for e in errs)
    assert any("recommendation" in e for e in errs)


def test_thesis_and_risk_cardinality():
    memo = _good_memo()
    memo["thesis_bull"] = [memo["thesis_bull"][0]]  # only 1 (min 2)
    memo["risks"] = []  # min 1
    errs = validate_memo(memo)
    assert any("thesis_bull" in e for e in errs)
    assert any("risks" in e for e in errs)


def test_structurer_error_is_hard_fail():
    assert validate_memo({"structurer_error": "boom"}) == ["synthesis failed: boom"]


def test_citation_out_of_range_flagged():
    memo = _good_memo()
    memo["thesis_bull"][0]["citation_ids"] = [5]  # only 2 citations exist
    errs = validate_citations(memo, citations=[{}, {}])
    assert any("out of range" in e for e in errs)


def test_duplicate_citation_ids_flagged():
    memo = _good_memo()
    memo["risks"][0]["citation_ids"] = [1, 1]
    errs = validate_citations(memo, citations=[{}, {}])
    assert any("duplicate" in e for e in errs)
