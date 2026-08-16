"""Performance metrics for a backtest run."""

import numpy as np
import pandas as pd

from .engine import BacktestResult


def compute_metrics(result: BacktestResult) -> dict:
    trades = result.trades
    equity = result.equity
    initial = result.params.initial_balance

    m: dict = {"initial_balance": initial}

    if trades.empty:
        m.update({"trades": 0, "net_profit": 0.0, "final_balance": initial})
        return m

    pnl = trades["pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    m["trades"] = len(trades)
    m["net_profit"] = float(pnl.sum())
    m["final_balance"] = initial + m["net_profit"]
    m["return_pct"] = m["net_profit"] / initial * 100.0
    m["win_rate_pct"] = len(wins) / len(trades) * 100.0
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    m["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    m["avg_win"] = float(wins.mean()) if len(wins) else 0.0
    m["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
    m["expectancy"] = float(pnl.mean())

    # max drawdown on the mark-to-market equity curve
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    m["max_drawdown_pct"] = float(-drawdown.min() * 100.0)

    # annualized Sharpe from daily equity returns
    daily = equity.resample("1D").last().dropna()
    daily_ret = daily.pct_change().dropna()
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        m["sharpe"] = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
    else:
        m["sharpe"] = 0.0

    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years > 0 and m["final_balance"] > 0:
        m["cagr_pct"] = ((m["final_balance"] / initial) ** (1 / years) - 1) * 100.0
    else:
        m["cagr_pct"] = 0.0

    # longest losing streak
    streak = longest = 0
    for x in pnl:
        streak = streak + 1 if x <= 0 else 0
        longest = max(longest, streak)
    m["longest_loss_streak"] = longest

    m["exit_reasons"] = trades["reason"].value_counts().to_dict()
    m["long_trades"] = int((trades["direction"] == "long").sum())
    m["short_trades"] = int((trades["direction"] == "short").sum())
    return m


def format_report(m: dict) -> str:
    if m.get("trades", 0) == 0:
        return "No trades were taken."
    lines = [
        f"Trades              : {m['trades']} ({m['long_trades']} long / {m['short_trades']} short)",
        f"Net profit          : {m['net_profit']:+,.2f} ({m['return_pct']:+.2f}%)",
        f"Final balance       : {m['final_balance']:,.2f} (from {m['initial_balance']:,.2f})",
        f"CAGR                : {m['cagr_pct']:+.2f}%",
        f"Win rate            : {m['win_rate_pct']:.1f}%",
        f"Profit factor       : {m['profit_factor']:.2f}",
        f"Avg win / avg loss  : {m['avg_win']:+,.2f} / {m['avg_loss']:+,.2f}",
        f"Expectancy per trade: {m['expectancy']:+,.2f}",
        f"Max drawdown        : {m['max_drawdown_pct']:.2f}%",
        f"Sharpe (daily, ann.): {m['sharpe']:.2f}",
        f"Longest loss streak : {m['longest_loss_streak']}",
        f"Exit reasons        : {m['exit_reasons']}",
    ]
    return "\n".join(lines)
