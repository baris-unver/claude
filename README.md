# XAUUSD Sequential Short/Long Strategy — EA + Quick Backtest

Two ways to test the strategy:

| | What | Where | Fidelity |
|---|---|---|---|
| 1 | `mql5/XAUUSD_Sequential_EA.mq5` | MetaTrader 5 Strategy Tester | Definitive — real broker ticks, spread, swap |
| 2 | `backtest/backtest.py` | Any machine with Python | Fast smoke test — same rules, M1 CSV or synthetic Monte Carlo |

## 1. MetaTrader 5 Strategy Tester (the real test)

1. Copy `mql5/XAUUSD_Sequential_EA.mq5` into your terminal's data folder under
   `MQL5/Experts/` (MetaTrader: *File → Open Data Folder*).
2. Open it in MetaEditor and compile (**F7**). It uses only the standard
   `<Trade\Trade.mqh>` library — no dependencies.
3. Open the Strategy Tester (**Ctrl+R**) and set:
   - **Expert**: `XAUUSD_Sequential_EA`
   - **Symbol**: `XAUUSD`, **Timeframe**: M1
   - **Modeling**: *Every tick based on real ticks* (stage-2 stop trailing is
     tick-sensitive; "1 minute OHLC" is an acceptable faster approximation)
   - **Date range**: start with 3–6 months; history downloads automatically
   - **Deposit/leverage**: anything realistic; the EA trades fixed 0.01/0.03/0.02 lots
4. Run and read the **Backtest** report tab (net profit, drawdown, trade list).

**Requirements/notes**
- Needs a **hedging** account type (the strategy holds a short and a long on
  XAUUSD at the same time). Netting accounts would merge the positions.
- `InpRestartAfterCycle=true` starts a new cycle (step 1) each time all
  positions close, so a long test produces a statistically useful trade count.
  Set it to `false` to test exactly one cycle.
- CP ("current price") is interpreted as **Bid** throughout.

## 2. Quick test with historical data (no MetaTrader needed)

`backtest/backtest.py` replicates the EA's state machine exactly.

### On real M1 history exported from MT5

In MetaTrader 5: *View → Symbols → XAUUSD → Bars tab* → pick M1 and a date
range → **Request** → **Export Bars**. Then:

```bash
python3 backtest/backtest.py --csv XAUUSD_M1.csv --spread 0.30
```

Any CSV/TSV with `open,high,low,close` columns works (MT5's `<OPEN>`-style
headers are handled). Each bar is expanded to O/H/L/C sub-steps to
approximate ticks; stage-4 ratio checks run once per minute close, matching
the rules.

### Synthetic Monte Carlo (runs anywhere, instantly)

```bash
pip install numpy
python3 backtest/backtest.py --synthetic --paths 100 --days 30 \
    --s0 3350 --vol 0.16 --spread 0.30          # gold-like volatility
python3 backtest/backtest.py --synthetic --t-dof 4 ...   # fat-tailed returns
```

## Results of the quick test (synthetic, 100 paths × 30 days each)

| Scenario | Mean PnL / path | Profitable paths |
|---|---|---|
| 16% vol, $0.30 spread | **−$15.65** | 35/100 |
| 16% vol, no spread | +$42.50 | 70/100 |
| 25% vol, $0.30 spread | +$1.60 | 45/100 |
| 16% vol, fat tails (t, dof 4), $0.30 spread | **−$23.59** | 36/100 |

Outcome mix is stable across scenarios: ~51% of cycles end on the stage-2
trailing stop (avg ≈ +$4), ~23% on rule 4-j long trailing stop (avg ≈ +$7),
~26% on the stage-5/6 path (avg ≈ **−$14**). Win rate ≈ 74%, but the losing
outcome is ~3× the size of the average win.

### Structural properties worth knowing (independent of any backtest)

1. **The stage-5/6 path is a guaranteed loss, locked at entry.** After the
   third position opens you hold 0.01 + 0.02 short vs 0.03 long — a full
   hedge. Total PnL from that moment is frozen at
   `100·(x − 3y + 2z) ≤ −0.003x − 2(x − z)` USD, i.e. at least ≈ **−$10**
   at x ≈ $3350, regardless of where price goes afterwards. The stage-6
   "exit when CP < 0.998x" only decides *when* the loss is realized, not
   how large it is.
2. **Every other exit locks a profit** (all stage-2 stops sit below x; the
   stage-4 stop sits above breakeven of the pair). So the strategy is
   many-small-wins vs. one structurally-certain larger loss, and the edge
   comes down to path probabilities — which is why spread and fat tails
   flip it from marginally positive to negative.
3. **Stage 6 has no upside exit.** If price rises after the third position
   opens, the (hedged) trio can stay open indefinitely waiting for
   CP < 0.998x — tying up margin. Because of point 1 this doesn't change
   PnL, but you may want a time- or upside-based exit anyway.

### Interpretation assumptions (flagged, not silently decided)

- Rules 4…4-h all prescribe the same action (SL = CP − $1), so they are
  implemented as one band `2.25 ≤ r < 12`, re-set on every new minute —
  including downward if price retreats, as literally written.
- If the stage-2 stop fires while waiting (price bounced), the cycle ends
  with a small locked profit and a new cycle starts.
- Rule 4-j's "take profit sell" is interpreted as the long's trailing SL
  firing (the strategy never sets an actual TP order).
