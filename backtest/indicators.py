"""Indicator implementations matching MetaTrader 5 built-ins.

- EMA: standard exponential moving average, alpha = 2 / (period + 1) (iMA MODE_EMA)
- RSI: Wilder-smoothed relative strength index (iRSI)
- ATR: Wilder-smoothed average true range (iATR)
"""

import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    # RSI = 100 * AG / (AG + AL); avoids a separate divide-by-zero branch
    return 100.0 * avg_gain / (avg_gain + avg_loss)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()
