"""Data loading and synthetic data generation.

Real data: export bars from MetaTrader 5 via
    View -> Symbols -> Bars -> request XAUUSD H1 range -> Export Bars
which produces a tab-separated file with <DATE>, <TIME>, <OPEN>, <HIGH>,
<LOW>, <CLOSE>, ... columns. `load_csv` handles that format as well as a
generic ``time,open,high,low,close`` CSV.

Synthetic data: a seeded regime-switching random walk with gold-like
volatility, used to validate the pipeline when no real data is available.
Synthetic results say nothing about live performance.
"""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close"]


def load_csv(path: str) -> pd.DataFrame:
    """Load MT5-exported or generic OHLC CSV into a UTC-naive indexed frame."""
    with open(path, "r", encoding="utf-8-sig") as fh:
        first_line = fh.readline()
    sep = "\t" if "\t" in first_line else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().strip("<>").lower() for c in df.columns]

    if "date" in df.columns and "time" in df.columns:
        idx = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str), format="mixed"
        )
    elif "time" in df.columns:
        idx = pd.to_datetime(df["time"], format="mixed")
    elif "date" in df.columns:
        idx = pd.to_datetime(df["date"], format="mixed")
    else:
        raise ValueError(f"{path}: no <DATE>/<TIME> or time column found")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    out = df[REQUIRED_COLUMNS].copy()
    out.index = idx
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out.astype(float)


def generate_synthetic(
    start: str = "2023-01-02",
    end: str = "2025-12-31",
    start_price: float = 1840.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate seeded synthetic XAUUSD-like H1 bars (weekends skipped)."""
    rng = np.random.default_rng(seed)
    stamps = pd.date_range(start, end, freq="1h")
    stamps = stamps[stamps.dayofweek < 5]
    n = len(stamps)

    # hourly sigma for ~16% annualized volatility over ~24*252 trading hours
    hourly_sigma = 0.16 / np.sqrt(252 * 24)

    # regime-switching drift: bull / bear / chop blocks of 200-800 hours
    drift = np.empty(n)
    i = 0
    while i < n:
        block = int(rng.integers(200, 800))
        regime = rng.choice([1.0, -0.7, 0.0], p=[0.45, 0.25, 0.30])
        # regime drift expressed as a fraction of hourly sigma
        drift[i : i + block] = regime * 0.09 * hourly_sigma
        i += block

    log_returns = drift + hourly_sigma * rng.standard_normal(n)
    closes = start_price * np.exp(np.cumsum(log_returns))
    opens = np.empty(n)
    opens[0] = start_price
    opens[1:] = closes[:-1]

    # intra-bar range on top of the open/close body
    wick = np.abs(rng.standard_normal(n)) * hourly_sigma * closes * 0.6
    highs = np.maximum(opens, closes) + wick
    lows = np.minimum(opens, closes) - np.abs(rng.standard_normal(n)) * hourly_sigma * closes * 0.6

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes}, index=stamps
    )
    return df.round(2)
