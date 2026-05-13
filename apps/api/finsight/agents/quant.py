"""Quant agent.

Pure-Python technical indicators on the daily series. No external calls — runs
in the same process as the orchestrator. We deliberately avoid `ta-lib` (C
dependency, painful to deploy) and reimplement RSI/MACD/vol in pandas.

Indicators produced:
- 1m / 3m / 12m total return
- 30d / 90d annualized realized volatility
- RSI(14)
- MACD(12,26,9) — value + signal
- SMA(50), SMA(200), and trend flag
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from finsight.agents.state import AgentError, QuantSignals, ResearchState
from finsight.logging_setup import get_logger

log = get_logger(__name__)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _annualized_vol(close: pd.Series, window: int) -> float | None:
    if len(close) < window + 1:
        return None
    returns = close.pct_change().dropna()
    if len(returns) < window:
        return None
    return float(returns.tail(window).std() * np.sqrt(252))


def _return_over(close: pd.Series, bars: int) -> float | None:
    if len(close) < bars + 1:
        return None
    return float(close.iloc[-1] / close.iloc[-bars - 1] - 1)


def _last(series: pd.Series) -> float | None:
    if series.empty or pd.isna(series.iloc[-1]):
        return None
    return float(series.iloc[-1])


def _summarize(q: QuantSignals) -> str:
    parts: list[str] = []
    if q.rsi_14 is not None:
        if q.rsi_14 > 70:
            parts.append(f"RSI {q.rsi_14:.0f} (overbought)")
        elif q.rsi_14 < 30:
            parts.append(f"RSI {q.rsi_14:.0f} (oversold)")
        else:
            parts.append(f"RSI {q.rsi_14:.0f} (neutral)")
    if q.macd is not None and q.macd_signal is not None:
        cross = "bullish" if q.macd > q.macd_signal else "bearish"
        parts.append(f"MACD {cross}")
    if q.above_sma_200 is True:
        parts.append("above 200d SMA")
    elif q.above_sma_200 is False:
        parts.append("below 200d SMA")
    if q.return_3m is not None:
        parts.append(f"3m return {q.return_3m * 100:+.1f}%")
    return " · ".join(parts) if parts else "insufficient history"


async def run(state: ResearchState) -> dict[str, Any]:
    ticker = state["ticker"]
    bars = state.get("price_bars") or []
    if len(bars) < 20:
        log.info("quant.skip ticker=%s bars=%d", ticker, len(bars))
        return {
            "quant": QuantSignals(summary="insufficient history"),
            "errors": [AgentError(agent="quant", error=f"only {len(bars)} bars available")],
        }

    df = pd.DataFrame([b.model_dump() for b in bars])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    close = df["close"]

    rsi = _rsi(close, 14)
    macd_line, macd_signal = _macd(close)
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()

    last_close = _last(close)
    last_sma_200 = _last(sma_200)

    q = QuantSignals(
        last_close=last_close,
        return_1m=_return_over(close, 21),
        return_3m=_return_over(close, 63),
        return_12m=_return_over(close, 252),
        volatility_30d=_annualized_vol(close, 30),
        volatility_90d=_annualized_vol(close, 90),
        rsi_14=_last(rsi),
        macd=_last(macd_line),
        macd_signal=_last(macd_signal),
        sma_50=_last(sma_50),
        sma_200=last_sma_200,
        above_sma_200=(last_close > last_sma_200) if (last_close and last_sma_200) else None,
    )
    q.summary = _summarize(q)
    log.info("quant.done ticker=%s summary=%s", ticker, q.summary)
    return {"quant": q}
