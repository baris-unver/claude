# XAUUSD Trend EA + Backtest

A trend-following trading strategy for XAUUSD (gold), implemented twice from one
shared rule set:

- **`mql5/Experts/XAUUSD_TrendEA.mq5`** — a MetaTrader 5 Expert Advisor for live
  trading and the built-in Strategy Tester.
- **`backtest/`** — a Python harness that mirrors the EA's rules bar-for-bar, so
  the strategy can be backtested, cross-checked, and iterated on outside MetaTrader.

## Strategy rules

All signals are evaluated on **closed** bars (default timeframe H1) and executed
at the open of the next bar.

| Component | Rule (defaults) |
|---|---|
| Trend filter | EMA(50) above/below EMA(200) |
| Entry trigger | Close crosses back over EMA(50) in the trend direction |
| Confirmation | RSI(14) ≥ 52 for longs, ≤ 48 for shorts |
| Stop loss | 2.0 × ATR(14) from entry |
| Take profit | 3.0 × ATR(14) from entry |
| Trailing stop | Optional, k × ATR (off by default) |
| Exit override | Close on opposite signal (on by default) |
| Position sizing | Risk 1% of balance per trade; skip trade if below min lot |
| Session filter | Only open trades 07:00–20:00 server time |
| Other filters | Max spread, minimum ATR, daily loss limit (all optional) |

One position at a time, scoped by magic number.

## Backtesting in MetaTrader 5 (recommended for final validation)

1. Copy `mql5/Experts/XAUUSD_TrendEA.mq5` into your terminal's
   `MQL5/Experts/` folder and compile it in MetaEditor (F7).
2. Open the Strategy Tester (Ctrl+R), select the EA, symbol **XAUUSD**,
   timeframe **H1**, modeling mode *Every tick based on real ticks* for the most
   realistic run.
3. Load `mql5/Presets/XAUUSD_TrendEA_H1.set` under *Inputs → Load*. The preset
   also carries sensible optimization ranges for the key parameters.

The EA compiles against the standard library only (`Trade/Trade.mqh`).

## Backtesting in Python

```bash
pip install -r requirements.txt

# demo on seeded synthetic data (validates the pipeline only)
python run_backtest.py --synthetic --plot results/equity.png

# real data exported from MT5
python run_backtest.py --csv data/XAUUSD_H1.csv --plot results/equity.png --trades-csv results/trades.csv

# parameter overrides, e.g. tighter stop + ATR trailing
python run_backtest.py --csv data/XAUUSD_H1.csv --sl-mult 1.5 --trail-mult 2.0
```

**Getting real data:** in MT5 open *View → Symbols → Bars*, request the XAUUSD
H1 range you want, then *Export Bars*. The loader accepts that tab-separated
format (`<DATE> <TIME> <OPEN> …`) as well as a generic
`time,open,high,low,close` CSV. (This repository ships no market data — the
build environment had no access to market-data providers, so the committed demo
run uses the synthetic generator.)

`python run_backtest.py --help` lists every strategy parameter; each maps 1:1
to an EA input.

### Execution model

The Python engine is deliberately conservative:

- Quotes are bid; longs pay the spread on entry, shorts on exit (default $0.30).
- SL/TP are checked intrabar against high/low; if both fall inside one bar the
  **stop-loss is assumed to be hit first**.
- The trailing stop updates once per bar close (the EA trails every tick, which
  can only be tighter).
- Position sizing floors to the lot step and refuses trades that would exceed
  the risk budget at the minimum lot, matching the EA.

### Sanity check

On seeded synthetic random-walk data the strategy comes out near breakeven
(profit factor ≈ 1.0) — which is exactly what a leak-free engine should show on
data with no exploitable structure. A backtest engine that "profits" on random
data has look-ahead bias.

```
Trades              : 125 (79 long / 46 short)
Net profit          : +249.40 (+2.49%)
Profit factor       : 1.03
Max drawdown        : 11.12%
```

Reproduce with `python run_backtest.py --synthetic` and run the test suite with
`python -m pytest tests/` (11 tests: indicator correctness, session logic,
accounting consistency, pessimistic SL handling, MT5 CSV parsing).

## Repository layout

```
mql5/Experts/XAUUSD_TrendEA.mq5    the Expert Advisor
mql5/Presets/XAUUSD_TrendEA_H1.set Strategy Tester preset + optimization ranges
backtest/indicators.py             EMA / Wilder RSI / Wilder ATR (MT5-matching)
backtest/strategy.py               parameters + signal rules (single source of truth)
backtest/engine.py                 bar-by-bar simulator (spread, SL/TP, sizing)
backtest/metrics.py                performance statistics
backtest/data.py                   MT5/generic CSV loader + synthetic generator
run_backtest.py                    CLI entry point
tests/                             sanity test suite
```

## Disclaimer

This code is for research and education. Past performance — simulated or
otherwise — does not guarantee future results. Test extensively on a demo
account before risking real capital.
