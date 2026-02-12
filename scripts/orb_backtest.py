#!/usr/bin/env python3
"""
ORB (Opening Range Breakout) Backtest

Pure ORB strategy:
1. Define opening range as first 30 minutes (9:30-10:00 AM ET)
2. Long when price breaks above OR high
3. Short when price breaks below OR low
4. Track P&L to end of day (or current time)
"""

import os
import requests
from datetime import datetime

API_KEY = os.environ.get("ALPACA_API_KEY")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
BASE = "https://data.alpaca.markets/v2"
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET_KEY}


def run_orb_backtest(watchlist: list[str], date: str, end_hour_utc: int = 21):
    """
    Run ORB backtest on given watchlist for given date.

    Args:
        watchlist: List of stock symbols
        date: Date in YYYY-MM-DD format
        end_hour_utc: End hour in UTC (21 = 4 PM ET, market close)
    """
    # ORB: First 30 minutes (9:30-10:00 ET = 14:30-15:00 UTC)
    or_start = f"{date}T14:30:00Z"
    or_end = f"{date}T15:00:00Z"

    # End of analysis period
    day_end = f"{date}T{end_hour_utc:02d}:00:00Z"

    print("=" * 90)
    print(f"ORB BACKTEST - {date}")
    print("Opening Range: 9:30 - 10:00 AM ET (first 30 minutes)")
    print("=" * 90)
    print(f"{'Symbol':<8} {'OR High':>10} {'OR Low':>10} {'Range':>8} {'Current':>10} {'Long?':>6} {'Short?':>7} {'Long P&L':>10} {'Short P&L':>10}")
    print("-" * 90)

    total_long_pnl = []
    total_short_pnl = []

    for symbol in watchlist:
        try:
            # Get opening range bars (5-min bars from 9:30-10:00)
            params = {
                "symbols": symbol,
                "timeframe": "5Min",
                "start": or_start,
                "end": or_end,
                "limit": 10,
                "feed": "iex"
            }
            resp = requests.get(f"{BASE}/stocks/bars", headers=HEADERS, params=params)
            data = resp.json()

            if "bars" not in data or symbol not in data["bars"] or not data["bars"][symbol]:
                print(f"{symbol:<8} {'NO DATA'}")
                continue

            or_bars = data["bars"][symbol]
            or_high = max(b["h"] for b in or_bars)
            or_low = min(b["l"] for b in or_bars)
            or_range = or_high - or_low

            # Get rest of day bars
            params2 = {
                "symbols": symbol,
                "timeframe": "5Min",
                "start": or_end,
                "end": day_end,
                "limit": 100,
                "feed": "iex"
            }
            resp2 = requests.get(f"{BASE}/stocks/bars", headers=HEADERS, params=params2)
            data2 = resp2.json()

            if "bars" not in data2 or symbol not in data2["bars"] or not data2["bars"][symbol]:
                print(f"{symbol:<8} {or_high:>10.2f} {or_low:>10.2f} {or_range:>8.2f} {'NO DATA':>10}")
                continue

            rest_bars = data2["bars"][symbol]
            current_price = rest_bars[-1]["c"]

            # Check for breakouts
            long_triggered = False
            short_triggered = False
            long_entry = None
            short_entry = None

            for bar in rest_bars:
                if not long_triggered and bar["h"] > or_high:
                    long_triggered = True
                    long_entry = or_high
                if not short_triggered and bar["l"] < or_low:
                    short_triggered = True
                    short_entry = or_low

            # Calculate P&L
            long_pnl_str = ""
            short_pnl_str = ""

            if long_triggered and long_entry:
                pnl = ((current_price - long_entry) / long_entry) * 100
                long_pnl_str = f"{pnl:+.2f}%"
                total_long_pnl.append(pnl)

            if short_triggered and short_entry:
                pnl = ((short_entry - current_price) / short_entry) * 100
                short_pnl_str = f"{pnl:+.2f}%"
                total_short_pnl.append(pnl)

            long_flag = "YES" if long_triggered else "NO"
            short_flag = "YES" if short_triggered else "NO"

            print(f"{symbol:<8} {or_high:>10.2f} {or_low:>10.2f} {or_range:>8.2f} {current_price:>10.2f} {long_flag:>6} {short_flag:>7} {long_pnl_str:>10} {short_pnl_str:>10}")

        except Exception as e:
            print(f"{symbol:<8} ERROR: {e}")

    print("-" * 90)

    if total_long_pnl:
        avg_long = sum(total_long_pnl) / len(total_long_pnl)
        winners = sum(1 for p in total_long_pnl if p > 0)
        print(f"LONG breakouts: {len(total_long_pnl)} | Winners: {winners} | Avg P&L: {avg_long:+.2f}%")
    else:
        print("LONG breakouts: 0")

    if total_short_pnl:
        avg_short = sum(total_short_pnl) / len(total_short_pnl)
        winners = sum(1 for p in total_short_pnl if p > 0)
        print(f"SHORT breakouts: {len(total_short_pnl)} | Winners: {winners} | Avg P&L: {avg_short:+.2f}%")
    else:
        print("SHORT breakouts: 0")


if __name__ == "__main__":
    import sys

    # Default watchlist - our core + screened
    default_watchlist = [
        "TSLA", "AAPL", "QQQ", "SPY",  # Core
        "NVDA", "MSFT", "INTC", "JOBY", "VALE", "LBRT"  # Screened
    ]

    # Get date from args or use today
    if len(sys.argv) > 1:
        date = sys.argv[1]
    else:
        date = datetime.now().strftime("%Y-%m-%d")

    # Get end hour (for intraday backtests before market close)
    end_hour = 21  # Default: 4 PM ET = 21:00 UTC
    if len(sys.argv) > 2:
        end_hour = int(sys.argv[2])

    run_orb_backtest(default_watchlist, date, end_hour)
