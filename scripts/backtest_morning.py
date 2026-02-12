#!/usr/bin/env python3
"""
Backtest this morning's trading window using the actual strategy components.

Simulates what the bot would have done between 7:00-10:00 AM ET today.
"""

import logging
import sys
from datetime import datetime

import pytz

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("alpaca").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.WARNING)

ET = pytz.timezone("America/New_York")


def main():
    from src.core.alpaca_client import AlpacaClient
    from src.bot.screener import StockScreener, MomentumScreener
    from src.bot.float_provider import FloatDataProvider
    from src.bot.signals.momentum_pullback import MomentumPullbackStrategy
    from src.data.indicators import macd as calc_macd, atr as calc_atr
    from config.settings import get_settings
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    settings = get_settings()
    client = AlpacaClient()
    float_provider = FloatDataProvider(fmp_api_key=settings.fmp_api_key)

    strategy = MomentumPullbackStrategy(
        macd_fast=8, macd_slow=21, macd_signal=5,
        atr_period=14, atr_stop_multiplier=1.5,
        pullback_min_candles=2, pullback_max_candles=15,
        pullback_max_retracement=0.65,
        risk_reward_target=2.0, min_signal_strength=0.5,
    )

    equity = client.get_equity()
    buying_power = client.get_buying_power()
    now_et = datetime.now(ET)

    print("=" * 72)
    print(f"  MOMENTUM DAY TRADING BACKTEST — {now_et.strftime('%A, %B %d, %Y %I:%M %p ET')}")
    print(f"  Trading Window: 7:00 - 10:00 AM ET")
    print(f"  Account: ${equity:.2f} equity, ${buying_power:.2f} buying power")
    print("=" * 72)

    # ── Step 1: Get raw top gainers ───────────────────────────────────
    print("\n" + "─" * 72)
    print("  STEP 1: TOP GAINERS (raw from Alpaca)")
    print("─" * 72)

    screener = StockScreener()
    raw_gainers = screener.get_top_gainers(top_n=20)

    print(f"\n  {'#':<4} {'Symbol':<10} {'Price':>8} {'Change':>10}")
    print(f"  {'─'*4} {'─'*10} {'─'*8} {'─'*10}")
    for i, r in enumerate(raw_gainers, 1):
        marker = ""
        if 1.0 <= r.price <= 10.0:
            marker = " ◄ IN RANGE"
        print(f"  {i:<4} {r.symbol:<10} ${r.price:>7.2f} {r.change_pct:>+9.1f}%{marker}")

    # ── Step 2: Price filter ($2-$10) ─────────────────────────────────
    in_range = [g for g in raw_gainers if 1.0 <= g.price <= 10.0]
    print(f"\n  Price filter ($1-$10, prefer $2+): {len(in_range)} of {len(raw_gainers)} pass")

    if not in_range:
        print("  No stocks in price range today. Exiting.")
        return

    # ── Step 3: Manual enrichment for each candidate ──────────────────
    print("\n" + "─" * 72)
    print("  STEP 2: ENRICHMENT (relative volume + float data)")
    print("─" * 72)

    enriched = []
    for g in in_range:
        symbol = g.symbol
        print(f"\n  ── {symbol} (${g.price:.2f}, {g.change_pct:+.1f}%) ──")

        # Skip warrants/rights (often have data issues)
        if ".WS" in symbol or symbol.endswith("W") or symbol.endswith("R"):
            print(f"     ⚠️  Skipping warrant/right: {symbol}")
            continue

        try:
            # Get 21-day daily bars
            bars_daily = client.get_bars(symbol, timeframe="1Day", limit=21)
            if bars_daily is None or len(bars_daily) < 5:
                print(f"     ❌ Insufficient daily bars ({len(bars_daily) if bars_daily is not None else 0})")
                continue

            today_vol = int(bars_daily["volume"].iloc[-1])
            hist_bars = bars_daily.iloc[:-1]
            avg_vol = hist_bars["volume"].mean()
            rel_vol = today_vol / avg_vol if avg_vol > 0 else 1.0

            today_high = float(bars_daily["high"].iloc[-1])
            today_low = float(bars_daily["low"].iloc[-1])
            prev_close = float(bars_daily["close"].iloc[-2]) if len(bars_daily) > 1 else g.price

            print(f"     Volume: {today_vol:,.0f} today / {avg_vol:,.0f} avg = {rel_vol:.1f}x relative volume")
            print(f"     Range: ${today_low:.2f} - ${today_high:.2f} (prev close: ${prev_close:.2f})")

            # Float data
            float_shares = None
            float_str = "N/A"
            try:
                float_data = float_provider.get_float(symbol)
                if float_data and float_data.float_shares:
                    float_shares = float_data.float_shares
                    float_str = f"{float_shares / 1e6:.1f}M"
                    print(f"     Float: {float_str} shares")
                else:
                    print(f"     Float: unavailable")
            except Exception as e:
                print(f"     Float: error ({e})")

            # Filter results
            passes_rv = rel_vol >= 5.0
            passes_float = float_shares is None or (float_shares / 1e6) <= 20.0

            status_parts = []
            if not passes_rv:
                status_parts.append(f"relVol {rel_vol:.1f}x < 5.0x")
            if float_shares and not passes_float:
                status_parts.append(f"float {float_shares / 1e6:.1f}M > 20M")

            if passes_rv and passes_float:
                print(f"     ✅ PASSES ALL FILTERS")
            elif passes_rv:
                print(f"     ✅ Passes volume filter (float unknown/ok)")
            else:
                print(f"     ❌ Fails: {', '.join(status_parts)}")

            enriched.append({
                "symbol": symbol,
                "price": g.price,
                "change_pct": g.change_pct,
                "rel_vol": rel_vol,
                "float_shares": float_shares,
                "float_str": float_str,
                "today_vol": today_vol,
                "avg_vol": avg_vol,
                "high_of_day": today_high,
                "passes": passes_rv,  # Lenient: pass if volume ok
            })

        except Exception as e:
            print(f"     ❌ Error: {e}")

    if not enriched:
        print("\n  No candidates survived enrichment. Exiting.")
        return

    # ── Step 4: Signal generation on enriched candidates ──────────────
    print("\n" + "─" * 72)
    print("  STEP 3: SIGNAL GENERATION (5-min bars, MACD + pullback)")
    print("─" * 72)

    # Sort by relative volume descending
    enriched.sort(key=lambda x: x["rel_vol"], reverse=True)

    signals_found = []
    for c in enriched:
        symbol = c["symbol"]
        print(f"\n  ── {symbol} (${c['price']:.2f}, {c['change_pct']:+.1f}%, {c['rel_vol']:.1f}x relVol) ──")

        try:
            bars = client.get_bars(symbol, timeframe="5Min", limit=100)
            if bars is None or len(bars) < 40:
                print(f"     ❌ Insufficient 5-min bars ({len(bars) if bars is not None else 0})")
                continue

            price = client.get_latest_price(symbol)

            # MACD (8/21/5 — faster for volatile low-float stocks)
            macd_line, signal_line, histogram = calc_macd(
                bars["close"], fast_period=8, slow_period=21, signal_period=5,
            )
            cur_macd = float(macd_line.iloc[-1])
            cur_signal = float(signal_line.iloc[-1])
            cur_hist = float(histogram.iloc[-1])

            # ATR
            atr_val = float(calc_atr(bars["high"], bars["low"], bars["close"], period=14).iloc[-1])

            print(f"     Price: ${price:.2f}")
            print(f"     MACD(8/21/5): {cur_macd:.4f} | Signal: {cur_signal:.4f} | Histogram: {cur_hist:.4f}")
            print(f"     ATR(14): ${atr_val:.4f}")

            # MACD check: only require MACD > 0 (trend filter)
            # MACD can be below signal during pullback — that's expected
            if cur_macd <= 0:
                print(f"     ❌ MACD below zero line — no bullish momentum")
                continue

            macd_vs_signal = "above" if cur_macd > cur_signal else "below"
            print(f"     ✅ MACD positive ({macd_vs_signal} signal line)")

            # Run the pullback strategy
            signal = strategy.generate(symbol, bars, price)

            if signal:
                signals_found.append((c, signal))
                rr = signal.risk_reward_ratio or 0
                meta = signal.metadata
                print(f"     🎯 SIGNAL: LONG @ ${signal.entry_price:.2f}")
                print(f"        Stop: ${signal.stop_price:.2f} (risk: ${signal.risk_amount:.2f}/sh)")
                print(f"        Target: ${signal.target_price:.2f} (R:R = {rr:.1f}x)")
                print(f"        Strength: {signal.strength:.2f} ({signal.strength_category.value})")
                print(f"        Pullback: {meta.get('pullback_candles', '?')} candles, "
                      f"{meta.get('retracement_pct', '?')}% retracement")
                print(f"        Entry vol ratio: {meta.get('volume_ratio', '?')}x vs pullback avg")
            else:
                print(f"     ❌ No valid pullback pattern (or failed volume/retracement check)")

        except Exception as e:
            print(f"     ❌ Error: {e}")

    # ── Step 5: Simulate the trade ────────────────────────────────────
    print("\n" + "─" * 72)
    print("  STEP 4: TRADE SIMULATION")
    print("─" * 72)

    if not signals_found:
        print("\n  No signals generated today.")
        print("  The scanner found candidates but none had a valid pullback")
        print("  pattern with MACD confirmation. This is the strategy being selective —")
        print("  better to skip than take a bad trade.\n")

        # Still show what a hypothetical trade would look like on the best candidate
        if enriched:
            best = enriched[0]
            print(f"  Best candidate: {best['symbol']} ({best['change_pct']:+.1f}%, {best['rel_vol']:.1f}x relVol)")
            print(f"  Even though it was a big mover, we need MACD > 0 + a clean pullback")
            print(f"  pattern before entering. No gambling!\n")
    else:
        cand, signal = signals_found[0]
        symbol = signal.symbol

        # Position sizing
        max_risk = equity * 0.02
        risk_per_sh = signal.risk_amount
        shares_by_risk = int(max_risk / risk_per_sh) if risk_per_sh > 0 else 0
        shares_by_bp = int(buying_power * 0.90 / signal.entry_price)
        shares = min(shares_by_risk, shares_by_bp)
        cost = shares * signal.entry_price
        total_risk = shares * risk_per_sh
        total_reward = shares * (signal.reward_amount or 0)

        print(f"\n  📈 TRADE: BUY {shares} shares of {symbol} @ ${signal.entry_price:.2f}")
        print(f"     Cost: ${cost:.2f} ({cost / equity * 100:.0f}% of equity)")
        print(f"     Risk: ${total_risk:.2f} ({total_risk / equity * 100:.1f}% of equity)")
        print(f"     Potential reward: ${total_reward:.2f}")
        print(f"     Stop: ${signal.stop_price:.2f} | Target: ${signal.target_price:.2f}")

        # Simulate outcome
        cur_price = client.get_latest_price(symbol)
        unrealized = (cur_price - signal.entry_price) * shares

        # Check if stop/target was hit in the bars
        bars = client.get_bars(symbol, timeframe="5Min", limit=100)
        hit_stop = hit_target = False
        exit_price = cur_price

        if bars is not None:
            for i in range(len(bars)):
                if float(bars["low"].iloc[i]) <= signal.stop_price:
                    hit_stop = True
                    exit_price = signal.stop_price
                    break
                if signal.target_price and float(bars["high"].iloc[i]) >= signal.target_price:
                    hit_target = True
                    exit_price = signal.target_price
                    break

        sim_pnl = (exit_price - signal.entry_price) * shares

        print(f"\n  📊 RESULT:")
        if hit_target:
            print(f"     🎉 TARGET HIT @ ${exit_price:.2f}")
            print(f"     P&L: ${sim_pnl:+.2f} ({sim_pnl / equity * 100:+.1f}%)")
        elif hit_stop:
            print(f"     🛑 STOP HIT @ ${exit_price:.2f}")
            print(f"     P&L: ${sim_pnl:+.2f} ({sim_pnl / equity * 100:+.1f}%)")
        else:
            print(f"     ⏳ Trade still open — current price: ${cur_price:.2f}")
            print(f"     Unrealized P&L: ${unrealized:+.2f} ({unrealized / equity * 100:+.1f}%)")

    print("\n" + "=" * 72)
    progress = (equity - 400) / 3600 * 100
    print(f"  Account: ${equity:.2f} → Goal: $4,000 ({progress:.1f}% progress)")
    print(f"  Strategy: Ross Cameron Momentum Pullback (5-min bars)")
    print("=" * 72)


if __name__ == "__main__":
    main()
