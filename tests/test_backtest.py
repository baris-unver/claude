"""Sanity tests for the backtest harness (run with: python -m pytest tests/)."""

import numpy as np
import pandas as pd
import pytest

from backtest.data import generate_synthetic, load_csv
from backtest.engine import run_backtest
from backtest.indicators import atr, ema, rsi
from backtest.metrics import compute_metrics
from backtest.strategy import StrategyParams, compute_signals, session_allows


@pytest.fixture(scope="module")
def synth():
    return generate_synthetic(start="2023-01-02", end="2023-12-29", seed=7)


def test_synthetic_data_shape(synth):
    assert len(synth) > 5000
    assert (synth["high"] >= synth[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (synth["low"] <= synth[["open", "close"]].min(axis=1) + 1e-9).all()
    assert synth.index.is_monotonic_increasing
    # weekends skipped
    assert (synth.index.dayofweek < 5).all()


def test_synthetic_data_is_deterministic():
    a = generate_synthetic(start="2023-01-02", end="2023-02-01", seed=3)
    b = generate_synthetic(start="2023-01-02", end="2023-02-01", seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_ema_converges_on_constant_series():
    s = pd.Series([5.0] * 100)
    assert ema(s, 10).iloc[-1] == pytest.approx(5.0)


def test_rsi_bounds_and_direction():
    up = pd.Series(np.linspace(100, 200, 50))
    down = pd.Series(np.linspace(200, 100, 50))
    assert rsi(up, 14).iloc[-1] > 99.0
    assert rsi(down, 14).iloc[-1] < 1.0


def test_atr_on_constant_range():
    n = 50
    high = pd.Series([102.0] * n)
    low = pd.Series([100.0] * n)
    close = pd.Series([101.0] * n)
    assert atr(high, low, close, 14).iloc[-1] == pytest.approx(2.0, rel=1e-3)


def test_session_filter():
    p = StrategyParams(session_start_hour=7, session_end_hour=20)
    assert session_allows(7, p) and session_allows(19, p)
    assert not session_allows(6, p) and not session_allows(20, p)
    overnight = StrategyParams(session_start_hour=22, session_end_hour=6)
    assert session_allows(23, overnight) and session_allows(2, overnight)
    assert not session_allows(12, overnight)


def test_signals_respect_warmup(synth):
    p = StrategyParams()
    sig = compute_signals(synth, p)
    warmup = max(p.ema_slow_period, p.atr_period, p.rsi_period) + 1
    assert not sig["long_signal"].iloc[:warmup].any()
    assert not sig["short_signal"].iloc[:warmup].any()


def test_backtest_runs_and_accounts_correctly(synth):
    p = StrategyParams()
    result = run_backtest(synth, p)
    assert len(result.equity) == len(synth) - 1
    if not result.trades.empty:
        # balance accounting: final equity equals initial balance plus total P/L
        assert result.equity.iloc[-1] == pytest.approx(
            p.initial_balance + result.trades["pnl"].sum(), abs=1e-6
        )
        # every closed trade must have exit info
        assert result.trades["exit_price"].notna().all()
        assert result.trades["pnl"].notna().all()


def test_stop_loss_is_pessimistic():
    """A bar that spans both SL and TP must be counted as a loss."""
    idx = pd.date_range("2024-01-01 08:00", periods=6, freq="1h")
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0, 150.0, 100.0],
            "low": [99.0, 99.0, 99.0, 99.0, 50.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=idx,
    )
    p = StrategyParams(
        ema_fast_period=2, ema_slow_period=3, use_session_filter=False, spread=0.0
    )
    sig = compute_signals(df, p)
    sig.loc[sig.index[3], "long_signal"] = True

    # drive the engine directly with a forced long entry on bar 4
    from backtest import engine as eng

    forced = df.copy()
    result = None

    original = eng.compute_signals
    try:
        eng.compute_signals = lambda d, params: sig
        result = eng.run_backtest(forced, p)
    finally:
        eng.compute_signals = original

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["reason"] == "sl"
    assert result.trades.iloc[0]["pnl"] < 0


def test_metrics_no_trades(synth):
    p = StrategyParams(allow_longs=False, allow_shorts=False)
    result = run_backtest(synth, p)
    m = compute_metrics(result)
    assert m["trades"] == 0


def test_load_mt5_csv(tmp_path):
    path = tmp_path / "mt5.csv"
    path.write_text(
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
        "2024.01.02\t00:00:00\t2064.0\t2066.0\t2063.0\t2065.5\t100\t0\t30\n"
        "2024.01.02\t01:00:00\t2065.5\t2067.0\t2064.0\t2066.0\t120\t0\t30\n"
    )
    df = load_csv(str(path))
    assert list(df.columns) == ["open", "high", "low", "close"]
    assert len(df) == 2
    assert df.iloc[0]["close"] == 2065.5
