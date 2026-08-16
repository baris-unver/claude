#!/usr/bin/env python3
"""Run the XAUUSD_TrendEA strategy backtest.

Examples:
    # demo run on seeded synthetic data (pipeline validation only)
    python run_backtest.py --synthetic

    # real data exported from MT5 (View -> Symbols -> Bars -> Export Bars)
    python run_backtest.py --csv data/XAUUSD_H1.csv

    # parameter overrides
    python run_backtest.py --csv data/XAUUSD_H1.csv --ema-fast 40 --sl-mult 1.5 --trail-mult 2.0
"""

import argparse
import os

from backtest.data import generate_synthetic, load_csv
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics, format_report
from backtest.strategy import StrategyParams


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="OHLC CSV (MT5 export or time,open,high,low,close)")
    src.add_argument("--synthetic", action="store_true", help="use seeded synthetic data")
    ap.add_argument("--seed", type=int, default=42, help="synthetic data seed")

    ap.add_argument("--ema-fast", type=int, default=50)
    ap.add_argument("--ema-slow", type=int, default=200)
    ap.add_argument("--rsi-period", type=int, default=14)
    ap.add_argument("--rsi-long-min", type=float, default=52.0)
    ap.add_argument("--rsi-short-max", type=float, default=48.0)
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--atr-min", type=float, default=0.0)
    ap.add_argument("--no-longs", action="store_true")
    ap.add_argument("--no-shorts", action="store_true")
    ap.add_argument("--no-close-on-opposite", action="store_true")

    ap.add_argument("--risk-pct", type=float, default=1.0)
    ap.add_argument("--sl-mult", type=float, default=2.0)
    ap.add_argument("--tp-mult", type=float, default=3.0)
    ap.add_argument("--trail-mult", type=float, default=0.0)
    ap.add_argument("--daily-loss-limit", type=float, default=0.0)

    ap.add_argument("--no-session-filter", action="store_true")
    ap.add_argument("--session-start", type=int, default=7)
    ap.add_argument("--session-end", type=int, default=20)

    ap.add_argument("--spread", type=float, default=0.30, help="spread in price units")
    ap.add_argument("--balance", type=float, default=10_000.0)

    ap.add_argument("--trades-csv", help="write closed trades to this CSV")
    ap.add_argument("--plot", help="write equity curve PNG to this path (needs matplotlib)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    params = StrategyParams(
        ema_fast_period=args.ema_fast,
        ema_slow_period=args.ema_slow,
        rsi_period=args.rsi_period,
        rsi_long_min=args.rsi_long_min,
        rsi_short_max=args.rsi_short_max,
        atr_period=args.atr_period,
        atr_min=args.atr_min,
        allow_longs=not args.no_longs,
        allow_shorts=not args.no_shorts,
        close_on_opposite=not args.no_close_on_opposite,
        risk_percent=args.risk_pct,
        sl_atr_mult=args.sl_mult,
        tp_atr_mult=args.tp_mult,
        trail_atr_mult=args.trail_mult,
        daily_loss_limit_pct=args.daily_loss_limit,
        use_session_filter=not args.no_session_filter,
        session_start_hour=args.session_start,
        session_end_hour=args.session_end,
        spread=args.spread,
        initial_balance=args.balance,
    )

    if args.synthetic:
        print("Data: SYNTHETIC (seeded random walk — validates the pipeline, "
              "says nothing about live performance)")
        df = generate_synthetic(seed=args.seed)
    else:
        print(f"Data: {args.csv}")
        df = load_csv(args.csv)

    print(f"Bars: {len(df)}  ({df.index[0]} .. {df.index[-1]})\n")

    result = run_backtest(df, params)
    metrics = compute_metrics(result)
    print(format_report(metrics))

    if args.trades_csv and not result.trades.empty:
        os.makedirs(os.path.dirname(args.trades_csv) or ".", exist_ok=True)
        result.trades.to_csv(args.trades_csv, index=False)
        print(f"\nTrades written to {args.trades_csv}")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(11, 7), sharex=True, height_ratios=[3, 1]
        )
        result.equity.plot(ax=ax1, color="#1a7f37", linewidth=1.0)
        ax1.set_title("XAUUSD_TrendEA backtest — equity (mark-to-market)")
        ax1.set_ylabel("Equity")
        ax1.grid(alpha=0.3)

        peak = result.equity.cummax()
        dd = (result.equity - peak) / peak * 100.0
        dd.plot(ax=ax2, color="#b91c1c", linewidth=0.8)
        ax2.fill_between(dd.index, dd, 0, color="#b91c1c", alpha=0.15)
        ax2.set_ylabel("Drawdown %")
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        fig.savefig(args.plot, dpi=120)
        print(f"Equity plot written to {args.plot}")


if __name__ == "__main__":
    main()
