#!/usr/bin/env python3
"""Quick backtester for the sequential XAUUSD short/long strategy.

Replicates the exact rule set of mql5/XAUUSD_Sequential_EA.mq5 so the
strategy can be smoke-tested without MetaTrader:

  * --csv FILE     run on real M1 history exported from MT5
                   (View -> Symbols -> Bars -> Request -> Export Bars,
                   or any CSV with time/open/high/low/close columns)
  * --synthetic    run a Monte Carlo on GBM minute paths calibrated to
                   gold-like volatility (default: 100 paths x 30 days)

Prices in the simulator are Bid; a fixed spread (USD) is charged on the
sides that trade at Ask (long entry, short exits). PnL is reported in
USD using the standard XAUUSD contract of 100 oz per 1.00 lot.

All stage-4 ratio checks happen once per minute (on the minute close),
matching the EA's "wait till the next minute value". Stage-2 band
updates and all stop-loss triggers are evaluated on every price step;
in CSV mode each M1 bar is expanded to open/high/low/close sub-steps
(O-L-H-C for up bars, O-H-L-C for down bars) to approximate ticks.
"""

import argparse
import csv
import math
import sys
from collections import Counter, defaultdict

import numpy as np

CONTRACT = 100.0  # oz per 1.00 lot -> 1 USD move on 0.01 lot = 1 USD

LOT1, LOT2, LOT3 = 0.01, 0.03, 0.02

# stage-2 bands: (lower bound as fraction of x, SL as fraction of x),
# evaluated top-down for cp < 0.999x; rule 2-j closes below 0.9775x
BANDS = [
    (0.9985, 0.99925),  # 2
    (0.998, 0.99875),   # 2-a
    (0.9975, 0.99825),  # 2-b
    (0.997, 0.99775),   # 2-c
    (0.9955, 0.99725),  # 2-d
    (0.994, 0.99575),   # 2-e
    (0.9925, 0.99425),  # 2-f
    (0.99, 0.993),      # 2-g
    (0.985, 0.9905),    # 2-h
    (0.9775, 0.9855),   # 2-i
]


class Sim:
    """One strategy cycle machine over a price stream."""

    S_OPEN, S_SHORT, S_PAIR, S_THREE = range(4)

    def __init__(self, spread, long_sl_offset=1.0):
        self.spread = spread
        self.off = long_sl_offset
        self.cycles = []  # (outcome, pnl_usd, minutes)
        self._reset()
        self.state = self.S_OPEN
        self.equity = 0.0

    def _reset(self):
        self.state = self.S_OPEN
        self.x = self.y = self.z = 0.0
        self.short1_sl = None
        self.long_sl = None
        self.t_open = 0

    def _finish(self, outcome, pnl, t):
        self.equity += pnl
        self.cycles.append((outcome, pnl, t - self.t_open))
        self._reset()

    # pnl helpers: bid prices in, spread charged on ask-side fills
    def _pnl_short(self, entry_bid, exit_bid, lots):
        return (entry_bid - (exit_bid + self.spread)) * lots * CONTRACT

    def _pnl_long(self, exit_bid, lots):
        return (exit_bid - (self.y + self.spread)) * lots * CONTRACT

    def step(self, t, cp, minute_close):
        """Advance the machine one price step.

        t: minute index, cp: bid price, minute_close: True when this step
        is the minute's closing value (stage-4 checks run only then).
        """
        if self.state == self.S_OPEN:
            self.x = cp
            self.t_open = t
            self.state = self.S_SHORT
            return

        if self.state == self.S_SHORT:
            x = self.x
            if self.short1_sl is not None and cp >= self.short1_sl:
                self._finish("stage2_sl", self._pnl_short(x, self.short1_sl, LOT1), t)
                return
            if cp <= 0.9775 * x:
                self._finish("rule_2j", self._pnl_short(x, cp, LOT1), t)
                return
            if cp >= 1.001 * x:
                self.y = cp  # long fills at ask = cp + spread (charged in pnl)
                self.long_sl = None
                self.state = self.S_PAIR
                return
            if cp < 0.999 * x:
                for lo, slf in BANDS:
                    if cp > lo * x:
                        self.short1_sl = slf * x
                        break
            return

        if self.state == self.S_PAIR:
            x, y = self.x, self.y
            if self.long_sl is not None and cp <= self.long_sl:
                # 4-j: long SL fires, close the short too
                pnl = self._pnl_long(self.long_sl, LOT2) + self._pnl_short(x, cp, LOT1)
                self._finish("rule_4j", pnl, t)
                return
            if cp < x:  # 5
                self.z = cp
                self.state = self.S_THREE
                return
            if not minute_close:
                return
            r = (cp - x) / (y - x)
            if r >= 12.0:  # 4-i
                pnl = self._pnl_long(cp, LOT2) + self._pnl_short(x, cp, LOT1)
                self._finish("rule_4i", pnl, t)
                return
            if r >= 2.25:  # 4 .. 4-h
                self.long_sl = cp - self.off
            return

        if self.state == self.S_THREE:
            x, z = self.x, self.z
            threshold = 0.998 * x if z >= 0.9995 * x else 0.9975 * x
            if cp < threshold:
                pnl = (self._pnl_short(x, cp, LOT1)
                       + self._pnl_long(cp, LOT2)
                       + self._pnl_short(z, cp, LOT3))
                self._finish("rule_6", pnl, t)
            return

    def open_pnl(self, cp):
        """Mark-to-market of whatever is still open at the end of data."""
        if self.state in (self.S_OPEN,):
            return 0.0
        pnl = self._pnl_short(self.x, cp, LOT1)
        if self.state in (self.S_PAIR, self.S_THREE):
            pnl += self._pnl_long(cp, LOT2)
        if self.state == self.S_THREE:
            pnl += self._pnl_short(self.z, cp, LOT3)
        return pnl


def run_closes(sim, closes):
    for t, cp in enumerate(closes):
        sim.step(t, cp, minute_close=True)
    return sim


def run_bars(sim, bars):
    """bars: iterable of (o, h, l, c). OHLC expanded to sub-steps."""
    for t, (o, h, l, c) in enumerate(bars):
        path = (o, l, h, c) if c >= o else (o, h, l, c)
        for i, cp in enumerate(path):
            sim.step(t, cp, minute_close=(i == 3))
    return sim


def load_csv(path):
    bars = []
    with open(path, newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delim = "\t" if "\t" in sample.splitlines()[0] else ","
        reader = csv.reader(f, delimiter=delim)
        header = next(reader)
        cols = [h.strip("<>").strip().lower() for h in header]

        def col(name):
            return cols.index(name) if name in cols else None

        io_, ih, il, ic = col("open"), col("high"), col("low"), col("close")
        if None in (io_, ih, il, ic):
            raise SystemExit(f"CSV must have open/high/low/close columns, got: {cols}")
        for row in reader:
            if not row or not row[io_]:
                continue
            bars.append((float(row[io_]), float(row[ih]),
                         float(row[il]), float(row[ic])))
    return bars


def gbm_minutes(s0, vol_annual, days, rng, t_dof=0):
    n = days * 1440
    dt = 1.0 / (252 * 1440)
    if t_dof > 2:
        raw = rng.standard_t(t_dof, n)
        raw /= math.sqrt(t_dof / (t_dof - 2.0))  # unit variance
    else:
        raw = rng.standard_normal(n)
    steps = vol_annual * math.sqrt(dt) * raw - 0.5 * vol_annual**2 * dt
    return s0 * np.exp(np.cumsum(steps))


def summarize(all_cycles, open_pnls, label):
    pnls = [p for _, p, _ in all_cycles]
    total = sum(pnls) + sum(open_pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    print(f"\n=== {label} ===")
    print(f"cycles completed : {len(pnls)}")
    print(f"closed PnL       : {sum(pnls):+.2f} USD")
    print(f"open PnL at end  : {sum(open_pnls):+.2f} USD")
    print(f"TOTAL PnL        : {total:+.2f} USD")
    if pnls:
        print(f"win rate         : {len(wins) / len(pnls) * 100:.1f}%  "
              f"(avg win {np.mean(wins) if wins else 0:+.2f}, "
              f"avg loss {np.mean(losses) if losses else 0:+.2f})")
    by = defaultdict(list)
    for outcome, p, dur in all_cycles:
        by[outcome].append((p, dur))
    print(f"{'outcome':>10} {'count':>6} {'sum USD':>10} {'avg USD':>9} {'avg min':>8}")
    for outcome, rows in sorted(by.items()):
        ps = [r[0] for r in rows]
        ds = [r[1] for r in rows]
        print(f"{outcome:>10} {len(rows):>6} {sum(ps):>10.2f} "
              f"{np.mean(ps):>9.2f} {np.mean(ds):>8.0f}")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="M1 OHLC CSV exported from MT5")
    ap.add_argument("--synthetic", action="store_true",
                    help="Monte Carlo on synthetic GBM minute paths")
    ap.add_argument("--paths", type=int, default=100)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--s0", type=float, default=3350.0, help="start price")
    ap.add_argument("--vol", type=float, default=0.16, help="annualized volatility")
    ap.add_argument("--t-dof", type=int, default=0,
                    help="Student-t dof for fat-tailed returns (0 = normal)")
    ap.add_argument("--spread", type=float, default=0.30, help="spread in USD")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.csv and not args.synthetic:
        ap.error("pick a mode: --csv FILE or --synthetic")

    if args.csv:
        bars = load_csv(args.csv)
        print(f"loaded {len(bars)} bars from {args.csv}")
        sim = run_bars(Sim(args.spread), bars)
        summarize(sim.cycles, [sim.open_pnl(bars[-1][3])], f"CSV {args.csv}")
        return

    rng = np.random.default_rng(args.seed)
    all_cycles, open_pnls, totals = [], [], []
    outcome_last = Counter()
    for _ in range(args.paths):
        closes = gbm_minutes(args.s0, args.vol, args.days, rng, args.t_dof)
        sim = run_closes(Sim(args.spread), closes)
        op = sim.open_pnl(closes[-1])
        all_cycles.extend(sim.cycles)
        open_pnls.append(op)
        totals.append(sum(p for _, p, _ in sim.cycles) + op)
        if sim.cycles:
            outcome_last[sim.cycles[-1][0]] += 1

    dist = ("normal" if args.t_dof <= 2 else f"student-t(dof={args.t_dof})")
    summarize(all_cycles, open_pnls,
              f"Monte Carlo: {args.paths} paths x {args.days} days, "
              f"S0={args.s0}, vol={args.vol:.0%} ({dist}), spread={args.spread}")
    totals = np.array(totals)
    print(f"\nper-path total PnL: mean {totals.mean():+.2f}, median "
          f"{np.median(totals):+.2f}, min {totals.min():+.2f}, max {totals.max():+.2f}")
    print(f"profitable paths  : {(totals > 0).sum()}/{len(totals)}")


if __name__ == "__main__":
    sys.exit(main())
