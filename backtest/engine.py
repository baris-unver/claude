"""Bar-by-bar backtest engine.

Execution model (kept deliberately conservative):
- Signals from closed bar i are executed at the open of bar i+1.
- Quotes are bid prices. Longs are filled at ask = bid + spread and exited
  at bid; shorts are filled at bid and exited at ask.
- SL/TP are evaluated intrabar against high/low; if both could have been
  hit within one bar, the stop-loss is assumed to have been hit first.
- The trailing stop is updated once per bar at the close (the EA trails
  tick-by-tick, so live trailing can only be tighter than modeled here).
"""

from dataclasses import dataclass, field

import math

import pandas as pd

from .strategy import StrategyParams, compute_signals, session_allows


@dataclass
class Trade:
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    lots: float
    sl: float
    tp: float
    exit_time: pd.Timestamp = None
    exit_price: float = None
    pnl: float = None
    reason: str = None


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    params: StrategyParams = field(repr=False, default=None)


def _size_position(balance: float, sl_distance: float, p: StrategyParams) -> float:
    if sl_distance <= 0:
        return 0.0
    risk_money = balance * p.risk_percent / 100.0
    loss_per_lot = sl_distance * p.contract_size
    lots = risk_money / loss_per_lot
    lots = math.floor(lots / p.lot_step) * p.lot_step
    if lots < p.min_lot:
        return 0.0  # mirror the EA: refuse oversized risk instead of bumping to min lot
    return min(round(lots, 2), p.max_lot)


def run_backtest(df: pd.DataFrame, p: StrategyParams) -> BacktestResult:
    data = compute_signals(df, p)

    opens = data["open"].to_numpy()
    highs = data["high"].to_numpy()
    lows = data["low"].to_numpy()
    closes = data["close"].to_numpy()
    atrs = data["atr"].to_numpy()
    long_sig = data["long_signal"].to_numpy()
    short_sig = data["short_signal"].to_numpy()
    times = data.index

    balance = p.initial_balance
    day_start_balance = balance
    current_day = None
    position: Trade | None = None
    trades: list[Trade] = []
    equity = []

    def close_position(i: int, price: float, reason: str):
        nonlocal balance, position
        if position.direction == "long":
            pnl = (price - position.entry_price) * position.lots * p.contract_size
        else:
            pnl = (position.entry_price - price) * position.lots * p.contract_size
        position.exit_time = times[i]
        position.exit_price = price
        position.pnl = pnl
        position.reason = reason
        balance += pnl
        trades.append(position)
        position = None

    for i in range(1, len(data)):
        bar_day = times[i].date()
        if bar_day != current_day:
            current_day = bar_day
            day_start_balance = balance

        prev_long = bool(long_sig[i - 1])
        prev_short = bool(short_sig[i - 1])
        signal_atr = atrs[i - 1]

        # --- act at the open of bar i on signals from closed bar i-1
        if position is not None and p.close_on_opposite:
            if position.direction == "long" and prev_short:
                close_position(i, opens[i], "opposite")
            elif position.direction == "short" and prev_long:
                close_position(i, opens[i] + p.spread, "opposite")

        if position is None and (prev_long or prev_short) and not math.isnan(signal_atr):
            allowed = True
            if p.atr_min > 0 and signal_atr < p.atr_min:
                allowed = False
            if allowed and not session_allows(times[i].hour, p):
                allowed = False
            if allowed and p.daily_loss_limit_pct > 0 and day_start_balance > 0:
                loss_pct = (day_start_balance - balance) / day_start_balance * 100.0
                if loss_pct >= p.daily_loss_limit_pct:
                    allowed = False

            if allowed:
                sl_distance = p.sl_atr_mult * signal_atr
                lots = _size_position(balance, sl_distance, p)
                if lots > 0:
                    if prev_long:
                        fill = opens[i] + p.spread  # buy at ask
                        position = Trade(
                            direction="long",
                            entry_time=times[i],
                            entry_price=fill,
                            lots=lots,
                            sl=fill - sl_distance,
                            tp=fill + p.tp_atr_mult * signal_atr if p.tp_atr_mult > 0 else 0.0,
                        )
                    elif prev_short:
                        fill = opens[i]  # sell at bid
                        position = Trade(
                            direction="short",
                            entry_time=times[i],
                            entry_price=fill,
                            lots=lots,
                            sl=fill + sl_distance,
                            tp=fill - p.tp_atr_mult * signal_atr if p.tp_atr_mult > 0 else 0.0,
                        )

        # --- intrabar SL/TP on bar i (pessimistic: SL before TP)
        if position is not None:
            if position.direction == "long":
                if lows[i] <= position.sl:
                    close_position(i, position.sl, "sl")
                elif position.tp > 0 and highs[i] >= position.tp:
                    close_position(i, position.tp, "tp")
            else:
                if highs[i] + p.spread >= position.sl:
                    close_position(i, position.sl, "sl")
                elif position.tp > 0 and lows[i] + p.spread <= position.tp:
                    close_position(i, position.tp, "tp")

        # --- trail at bar close (only ever tightens)
        if position is not None and p.trail_atr_mult > 0 and not math.isnan(atrs[i]):
            trail_distance = p.trail_atr_mult * atrs[i]
            if position.direction == "long":
                new_sl = closes[i] - trail_distance
                if new_sl > position.sl:
                    position.sl = new_sl
            else:
                new_sl = closes[i] + p.spread + trail_distance
                if new_sl < position.sl:
                    position.sl = new_sl

        # --- mark-to-market equity at bar close
        if position is None:
            equity.append(balance)
        elif position.direction == "long":
            equity.append(
                balance + (closes[i] - position.entry_price) * position.lots * p.contract_size
            )
        else:
            equity.append(
                balance
                + (position.entry_price - (closes[i] + p.spread)) * position.lots * p.contract_size
            )

    # close any open position at the final close
    if position is not None:
        last = len(data) - 1
        price = closes[last] if position.direction == "long" else closes[last] + p.spread
        close_position(last, price, "end_of_data")
        equity[-1] = balance

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    equity_series = pd.Series(equity, index=times[1:], name="equity")
    return BacktestResult(trades=trades_df, equity=equity_series, params=p)
