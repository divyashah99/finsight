"""Golden-value / sanity tests for the quant indicators (pure, no network)."""

from __future__ import annotations

import math
from datetime import date, timedelta

from finsight.agents.quant import compute_signals
from finsight.agents.state import PriceBar


def _bars(closes: list[float]) -> list[PriceBar]:
    start = date(2024, 1, 1)
    bars = []
    for i, c in enumerate(closes):
        bars.append(
            PriceBar(
                date=(start + timedelta(days=i)).isoformat(),  # monotonic dates
                open=c,
                high=c + 1,
                low=c - 1,
                close=c,
                volume=1_000_000,
            )
        )
    return bars


def test_insufficient_history_short_circuits():
    q = compute_signals(_bars([100.0] * 10))
    assert q.summary == "insufficient history"


def test_indicator_sanity_bounds():
    # 80-day noisy uptrend so gains AND losses exist (real RSI, not NaN).
    closes = [100 + i * 0.5 + (2 if i % 3 == 0 else -1) for i in range(80)]
    q = compute_signals(_bars(closes))

    assert q.rsi_14 is None or (0.0 <= q.rsi_14 <= 100.0)
    for v in (q.volatility_30d, q.volatility_90d):
        assert v is None or v >= 0.0
    for r in (q.return_1m, q.return_3m):
        assert r is None or r > -1.0  # can't lose more than 100%
    assert q.last_close == closes[-1]
    # rising series → positive 1m return
    assert q.return_1m is not None and q.return_1m > 0
    assert isinstance(q.summary, str) and q.summary


def test_above_sma_flag_consistency():
    closes = [50.0 + i for i in range(60)]  # strictly rising
    q = compute_signals(_bars(closes))
    if q.sma_50 is not None and q.last_close is not None:
        # last_close is the max of a strictly rising series → above SMA-50
        assert q.last_close > q.sma_50
    # SMA-200 needs 200 bars; with 60 it must be None
    assert q.sma_200 is None
    assert q.above_sma_200 is None


def test_no_nan_leaks_into_floats():
    q = compute_signals(_bars([100 + (i % 5) for i in range(40)]))
    for field in (q.rsi_14, q.macd, q.macd_signal, q.volatility_30d):
        assert field is None or math.isfinite(field)
