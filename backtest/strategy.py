"""Strategy parameters and signal computation.

The rules here must stay in sync with mql5/Experts/XAUUSD_TrendEA.mq5:

Long signal on closed bar i:
    ema_fast[i] > ema_slow[i]
    close[i-1] <= ema_fast[i-1]  and  close[i] > ema_fast[i]
    rsi[i] >= rsi_long_min
Short signal is the mirror image. Orders are executed by the engine at
the open of bar i+1, subject to the session/ATR/daily-loss filters.
"""

from dataclasses import dataclass

import pandas as pd

from .indicators import atr, ema, rsi


@dataclass
class StrategyParams:
    ema_fast_period: int = 50
    ema_slow_period: int = 200
    rsi_period: int = 14
    rsi_long_min: float = 52.0
    rsi_short_max: float = 48.0
    atr_period: int = 14
    atr_min: float = 0.0
    allow_longs: bool = True
    allow_shorts: bool = True
    close_on_opposite: bool = True

    # risk & exits
    risk_percent: float = 1.0
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0
    trail_atr_mult: float = 0.0
    daily_loss_limit_pct: float = 0.0

    # filters
    use_session_filter: bool = True
    session_start_hour: int = 7
    session_end_hour: int = 20

    # execution model
    spread: float = 0.30           # bid/ask spread in price units (~30 points on gold)
    initial_balance: float = 10_000.0
    contract_size: float = 100.0   # ounces per 1.0 lot of XAUUSD
    lot_step: float = 0.01
    min_lot: float = 0.01
    max_lot: float = 100.0


def compute_signals(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    """Return df plus indicator and signal columns (signals refer to that bar's close)."""
    out = df.copy()
    out["ema_fast"] = ema(out["close"], p.ema_fast_period)
    out["ema_slow"] = ema(out["close"], p.ema_slow_period)
    out["rsi"] = rsi(out["close"], p.rsi_period)
    out["atr"] = atr(out["high"], out["low"], out["close"], p.atr_period)

    prev_close = out["close"].shift(1)
    prev_ema_fast = out["ema_fast"].shift(1)

    trend_up = out["ema_fast"] > out["ema_slow"]
    trend_down = out["ema_fast"] < out["ema_slow"]
    cross_up = (prev_close <= prev_ema_fast) & (out["close"] > out["ema_fast"])
    cross_down = (prev_close >= prev_ema_fast) & (out["close"] < out["ema_fast"])

    out["long_signal"] = (
        p.allow_longs & trend_up & cross_up & (out["rsi"] >= p.rsi_long_min)
    )
    out["short_signal"] = (
        p.allow_shorts & trend_down & cross_down & (out["rsi"] <= p.rsi_short_max)
    )

    # do not signal while indicators are still warming up
    warmup = max(p.ema_slow_period, p.atr_period, p.rsi_period) + 1
    out.iloc[:warmup, out.columns.get_indexer(["long_signal", "short_signal"])] = False
    return out


def session_allows(hour: int, p: StrategyParams) -> bool:
    if not p.use_session_filter:
        return True
    if p.session_start_hour <= p.session_end_hour:
        return p.session_start_hour <= hour < p.session_end_hour
    return hour >= p.session_start_hour or hour < p.session_end_hour
