#!/usr/bin/env python3
"""
Backtest the MomentumSurgeStrategy on recent momentum stocks.

Fetches 5-min bars from yfinance for recent movers and tests both
the pullback and surge strategies to compare signal generation.
"""

import sys
sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("yfinance").setLevel(logging.WARNING)

from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from loguru import logger

# Suppress debug noise from strategies during backtest
logger.remove()
logger.add(sys.stderr, level="INFO")

from src.bot.signals.momentum_pullback import MomentumPullbackStrategy
from src.bot.signals.momentum_surge import MomentumSurgeStrategy
from src.data.indicators import macd, rsi, atr


def fetch_bars(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    """Fetch 5-min bars from yfinance."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        return df
    df.columns = [c.lower() for c in df.columns]
    if "vwap" not in df.columns:
        df["vwap"] = 0.0
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def simulate_trade(bars: pd.DataFrame, entry_idx: int, entry_price: float,
                   stop_price: float, strategy) -> dict:
    """Simulate a trade from entry to exit."""
    max_price = entry_price
    exit_price = None
    exit_reason = None
    exit_idx = None

    for i in range(entry_idx + 1, len(bars)):
        cur_price = float(bars["close"].iloc[i])
        cur_high = float(bars["high"].iloc[i])
        cur_low = float(bars["low"].iloc[i])
        max_price = max(max_price, cur_high)

        # Check stop loss
        if cur_low <= stop_price:
            exit_price = stop_price
            exit_reason = "stop_loss"
            exit_idx = i
            break

        # Check strategy exit (need at least 40 bars of history)
        if i >= 40:
            sub_bars = bars.iloc[max(0, i - 99):i + 1]
            if len(sub_bars) >= 40:
                try:
                    from src.bot.signals.base import SignalDirection
                    should_exit, reason = strategy.should_exit(
                        symbol="TEST", bars=sub_bars,
                        entry_price=entry_price,
                        direction=SignalDirection.LONG,
                        current_price=cur_price,
                    )
                    if should_exit:
                        exit_price = cur_price
                        exit_reason = reason
                        exit_idx = i
                        break
                except Exception:
                    pass

    # If no exit, close at last bar
    if exit_price is None:
        exit_price = float(bars["close"].iloc[-1])
        exit_reason = "end_of_data"
        exit_idx = len(bars) - 1

    pnl_pct = (exit_price - entry_price) / entry_price * 100
    r_multiple = (exit_price - entry_price) / (entry_price - stop_price) if entry_price != stop_price else 0

    return {
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_price": stop_price,
        "pnl_pct": pnl_pct,
        "r_multiple": r_multiple,
        "max_price": max_price,
        "max_r": (max_price - entry_price) / (entry_price - stop_price) if entry_price != stop_price else 0,
        "exit_reason": exit_reason,
        "bars_held": exit_idx - entry_idx,
    }


def backtest_symbol(symbol: str, bars: pd.DataFrame, pullback: MomentumPullbackStrategy,
                    surge: MomentumSurgeStrategy) -> list[dict]:
    """Run both strategies on a symbol's bars, scanning bar-by-bar."""
    trades = []

    # Slide a window across the bars, checking for signals
    for i in range(40, len(bars) - 5):  # Need 40 bars history, 5 bars forward
        window = bars.iloc[max(0, i - 99):i + 1]
        cur_price = float(bars["close"].iloc[i])

        # Try pullback first
        signal = pullback.generate(symbol, window, cur_price)
        strategy_used = pullback
        strategy_name = "pullback"

        # Then surge
        if signal is None:
            signal = surge.generate(symbol, window, cur_price)
            strategy_used = surge
            strategy_name = "surge"

        if signal is None:
            continue

        # Simulate trade
        result = simulate_trade(bars, i, signal.entry_price, signal.stop_price, strategy_used)
        result["symbol"] = symbol
        result["strategy"] = strategy_name
        result["signal_strength"] = signal.strength
        result["entry_time"] = bars.index[i]
        trades.append(result)

        # Skip ahead past this trade
        # (don't generate overlapping signals)
        # Jump past exit
        pass  # We'll deduplicate below

    # Deduplicate: only keep first signal per trading day
    seen_days = set()
    unique_trades = []
    for t in trades:
        day = t["entry_time"].date() if hasattr(t["entry_time"], "date") else str(t["entry_time"])[:10]
        key = (t["symbol"], day, t["strategy"])
        if key not in seen_days:
            seen_days.add(key)
            unique_trades.append(t)

    return unique_trades


def main():
    # Test symbols: recent momentum movers
    symbols = [
        "LRMR", "VIR",      # Today's movers
        "ALUR", "VNDA",      # Yesterday's movers
        "ACLX", "EHAB",      # Yesterday's movers
        "GNPX", "SPAI",      # Recent scanners
        "SMCI", "NVDA",      # Popular momentum stocks
        "PLTR", "MARA",      # Volatile tech/crypto
    ]

    pullback = MomentumPullbackStrategy(
        macd_fast=8, macd_slow=21, macd_signal=5,
        atr_period=14, atr_stop_multiplier=1.0,
        pullback_min_candles=2, pullback_max_candles=8,
        pullback_max_retracement=0.50,
        risk_reward_target=10.0, min_signal_strength=0.5,
    )

    surge = MomentumSurgeStrategy(
        macd_fast=8, macd_slow=21, macd_signal=5,
        atr_period=14, atr_stop_multiplier=2.0,
        volume_multiplier=3.0, roc_min=0.03,
        risk_reward_target=10.0, min_signal_strength=0.5,
    )

    print("=" * 80)
    print("  MOMENTUM SURGE vs PULLBACK BACKTEST")
    print(f"  Period: Last 5 trading days | Timeframe: 5-min bars")
    print("=" * 80)

    all_trades = []

    for symbol in symbols:
        print(f"\n  Fetching {symbol}...", end=" ", flush=True)
        bars = fetch_bars(symbol)
        if bars.empty or len(bars) < 50:
            print(f"insufficient data ({len(bars)} bars)")
            continue
        print(f"{len(bars)} bars", end=" ", flush=True)

        trades = backtest_symbol(symbol, bars, pullback, surge)
        all_trades.extend(trades)
        print(f"→ {len(trades)} signals")

    if not all_trades:
        print("\n  No signals generated. Try different symbols or parameters.")
        return

    # ── Results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(all_trades)

    print("\n" + "=" * 80)
    print("  TRADE LOG")
    print("=" * 80)
    print(f"\n  {'Symbol':<8} {'Strategy':<10} {'Entry':>8} {'Exit':>8} {'P&L':>8} "
          f"{'R':>6} {'MaxR':>6} {'Bars':>5} {'Exit Reason':<20}")
    print(f"  {'─'*8} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*5} {'─'*20}")

    for _, t in df.iterrows():
        pnl_str = f"{t['pnl_pct']:+.1f}%"
        print(f"  {t['symbol']:<8} {t['strategy']:<10} ${t['entry_price']:>7.2f} "
              f"${t['exit_price']:>7.2f} {pnl_str:>8} "
              f"{t['r_multiple']:>+5.1f}R {t['max_r']:>5.1f}R {t['bars_held']:>5} "
              f"{t['exit_reason']:<20}")

    # ── Summary by strategy ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SUMMARY BY STRATEGY")
    print("=" * 80)

    for strat_name in ["pullback", "surge"]:
        strat_trades = df[df["strategy"] == strat_name]
        if strat_trades.empty:
            print(f"\n  {strat_name.upper()}: 0 trades")
            continue

        n = len(strat_trades)
        wins = len(strat_trades[strat_trades["pnl_pct"] > 0])
        avg_pnl = strat_trades["pnl_pct"].mean()
        avg_r = strat_trades["r_multiple"].mean()
        total_r = strat_trades["r_multiple"].sum()
        avg_max_r = strat_trades["max_r"].mean()
        win_rate = wins / n * 100

        print(f"\n  {strat_name.upper()}:")
        print(f"    Trades: {n} | Wins: {wins} | Win Rate: {win_rate:.0f}%")
        print(f"    Avg P&L: {avg_pnl:+.2f}% | Avg R: {avg_r:+.2f}R | Total R: {total_r:+.1f}R")
        print(f"    Avg Max R (unrealized): {avg_max_r:.2f}R")

    # ── Overall ──────────────────────────────────────────────────────────
    print(f"\n  OVERALL: {len(df)} trades | "
          f"Win Rate: {len(df[df['pnl_pct'] > 0]) / len(df) * 100:.0f}% | "
          f"Total R: {df['r_multiple'].sum():+.1f}R")
    print("=" * 80)


if __name__ == "__main__":
    main()
