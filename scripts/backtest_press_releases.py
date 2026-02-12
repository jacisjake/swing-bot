#!/usr/bin/env python3
"""
Backtest the press release scanner from midnight to now.

Simulates what would have happened if the press release scanner had been
running from midnight ET through the current time. Cross-references catalyst
hits with today's actual top gainers and runs the momentum pullback strategy
to identify trades the bot would have taken.

Usage:
    python scripts/backtest_press_releases.py
    python scripts/backtest_press_releases.py --lookback 24   # 24-hour lookback
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone

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
UTC = timezone.utc


def main():
    parser = argparse.ArgumentParser(description="Backtest press release scanner")
    parser.add_argument(
        "--lookback", type=int, default=None,
        help="Hours to look back (default: from midnight ET to now)",
    )
    args = parser.parse_args()

    from src.core.alpaca_client import AlpacaClient
    from src.bot.screener import StockScreener, MomentumScreener
    from src.bot.float_provider import FloatDataProvider
    from src.bot.press_release_scanner import PressReleaseScanner
    from src.bot.signals.momentum_pullback import MomentumPullbackStrategy
    from src.data.indicators import macd as calc_macd, atr as calc_atr
    from config.settings import get_settings

    settings = get_settings()
    client = AlpacaClient()
    float_provider = FloatDataProvider(fmp_api_key=settings.fmp_api_key)

    strategy = MomentumPullbackStrategy(
        macd_fast=12, macd_slow=26, macd_signal=9,
        atr_period=14, atr_stop_multiplier=1.5,
        pullback_min_candles=2, pullback_max_candles=8,
        pullback_max_retracement=0.50,
        risk_reward_target=2.0, min_signal_strength=0.5,
    )

    now_et = datetime.now(ET)
    now_utc = datetime.now(UTC)

    # Calculate lookback
    if args.lookback:
        lookback_hours = args.lookback
    else:
        # From midnight ET to now
        midnight_et = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
        lookback_hours = max(1, int((now_et - midnight_et).total_seconds() / 3600) + 1)

    equity = client.get_equity()
    buying_power = client.get_buying_power()

    print("=" * 78)
    print(f"  PRESS RELEASE CATALYST BACKTEST")
    print(f"  {now_et.strftime('%A, %B %d, %Y %I:%M %p ET')}")
    print(f"  Lookback: {lookback_hours} hours (since midnight ET)")
    print(f"  Account: ${equity:.2f} equity, ${buying_power:.2f} buying power")
    print("=" * 78)

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE 1: SCAN PRESS RELEASES
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "═" * 78)
    print("  PHASE 1: PRESS RELEASE SCAN")
    print("  Scanning GlobeNewswire + PR Newswire + FMP for overnight catalysts")
    print("═" * 78)

    scanner = PressReleaseScanner(
        fmp_api_key=settings.fmp_api_key,
        lookback_hours=lookback_hours,
    )

    # Run the scan
    t0 = time.time()
    hits = scanner.scan()
    scan_time = time.time() - t0

    status = scanner.get_status()
    print(f"\n  Scan completed in {scan_time:.1f}s")
    print(f"  Total hits: {status['total_hits']}")
    print(f"  Positive:   {status['positive_hits']}")
    print(f"  Negative:   {status['negative_hits']}")
    print(f"  Neutral:    {status['neutral_hits']}")
    print(f"  Unique symbols: {status['unique_symbols']}")
    print(f"  FMP enabled: {status['fmp_enabled']}")

    if not hits:
        print("\n  ⚠️  No press releases with extractable tickers found.")
        print("  This could mean:")
        print("   - No newswire press releases in the lookback window")
        print("   - All press releases were for stocks without ticker symbols")
        print("   - RSS feeds may be temporarily unavailable")
        print("\n  Continuing with momentum scanner only...\n")

    # ── Show all hits ──
    if hits:
        print(f"\n  {'─' * 74}")
        print(f"  {'#':<4} {'Symbol':<8} {'Sentiment':<10} {'Source':<16} {'Published':<18} Headline")
        print(f"  {'─' * 74}")

        for i, hit in enumerate(hits, 1):
            sentiment_icon = {
                "positive": "↑ POS",
                "negative": "↓ NEG",
                "neutral": "● NEU",
            }.get(hit.sentiment, "?")

            pub_str = hit.published.strftime("%m/%d %I:%M %p") if hit.published else "unknown"
            headline_short = hit.headline[:50] + "..." if len(hit.headline) > 50 else hit.headline

            print(f"  {i:<4} {hit.symbol:<8} {sentiment_icon:<10} {hit.source:<16} {pub_str:<18} {headline_short}")

            if hit.matched_keywords:
                kw_str = ", ".join(hit.matched_keywords[:3])
                print(f"       Keywords: {kw_str}")

    # ── Positive catalyst symbols ──
    positive_symbols = scanner.get_catalyst_symbols(positive_only=True)
    all_symbols = scanner.get_catalyst_symbols(positive_only=False)

    if positive_symbols:
        print(f"\n  🎯 Positive catalyst symbols: {', '.join(positive_symbols)}")
    if all_symbols:
        print(f"  📋 All catalyst symbols: {', '.join(all_symbols)}")

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE 2: CROSS-REFERENCE WITH TODAY'S TOP GAINERS
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "═" * 78)
    print("  PHASE 2: CROSS-REFERENCE WITH TODAY'S MARKET DATA")
    print("  Checking if press release stocks actually moved today")
    print("═" * 78)

    screener = StockScreener()
    raw_gainers = screener.get_top_gainers(top_n=20)

    gainer_symbols = {g.symbol for g in raw_gainers}
    gainer_map = {g.symbol: g for g in raw_gainers}

    # Find overlaps
    pr_in_gainers = set(all_symbols) & gainer_symbols
    positive_in_gainers = set(positive_symbols) & gainer_symbols

    print(f"\n  Today's top 20 gainers: {', '.join(g.symbol for g in raw_gainers[:10])}...")
    print(f"  PR symbols in top gainers: {', '.join(pr_in_gainers) if pr_in_gainers else 'NONE'}")
    print(f"  Positive PR in gainers:    {', '.join(positive_in_gainers) if positive_in_gainers else 'NONE'}")

    # Show detail for overlapping symbols
    if pr_in_gainers:
        print(f"\n  {'─' * 74}")
        print(f"  MATCHES: Press release stocks that are also top gainers today")
        print(f"  {'─' * 74}")

        for sym in pr_in_gainers:
            gainer = gainer_map[sym]
            pr_hits = scanner.get_hits_for_symbol(sym)
            best_hit = pr_hits[0]

            marker = "🟢" if sym in positive_in_gainers else "🟡"
            print(f"\n  {marker} {sym}: ${gainer.price:.2f} ({gainer.change_pct:+.1f}%)")
            print(f"    PR: {best_hit.headline[:70]}")
            print(f"    Sentiment: {best_hit.sentiment} | Source: {best_hit.source}")

    # ── Check PR symbols for price action even if not top gainers ──
    pr_only_symbols = set(all_symbols) - gainer_symbols
    if pr_only_symbols:
        print(f"\n  {'─' * 74}")
        print(f"  PR-ONLY: Press release stocks NOT in top 20 gainers")
        print(f"  {'─' * 74}")

        for sym in list(pr_only_symbols)[:10]:  # Check up to 10
            try:
                price = client.get_latest_price(sym)
                bars_daily = client.get_bars(sym, timeframe="1Day", limit=2)
                if bars_daily is not None and len(bars_daily) >= 2:
                    prev_close = float(bars_daily["close"].iloc[-2])
                    change_pct = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0
                    pr_hits = scanner.get_hits_for_symbol(sym)
                    sentiment = pr_hits[0].sentiment if pr_hits else "?"

                    if abs(change_pct) >= 5:
                        icon = "📈" if change_pct > 0 else "📉"
                    else:
                        icon = "➖"

                    print(f"  {icon} {sym}: ${price:.2f} ({change_pct:+.1f}%) | PR sentiment: {sentiment}")
                    if pr_hits:
                        print(f"    PR: {pr_hits[0].headline[:65]}")
            except Exception as e:
                print(f"  ❌ {sym}: error getting price ({e})")

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE 3: MOMENTUM SCAN (Full pipeline)
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "═" * 78)
    print("  PHASE 3: FULL MOMENTUM SCAN + ENRICHMENT")
    print("  Running the same pipeline the bot uses at 7:00 AM")
    print("═" * 78)

    # Get in-range gainers
    in_range = [g for g in raw_gainers if 1.0 <= g.price <= 10.0]
    print(f"\n  Price filter ($1-$10): {len(in_range)} of {len(raw_gainers)} pass")

    if not in_range:
        print("  No stocks in price range. Only larger-cap stocks moving today.")
        _print_summary(scanner, raw_gainers, [], equity)
        return

    # Enrich each candidate
    enriched = []
    for g in in_range:
        symbol = g.symbol
        print(f"\n  ── {symbol} (${g.price:.2f}, {g.change_pct:+.1f}%) ──")

        if ".WS" in symbol or symbol.endswith("W") or symbol.endswith("R"):
            print(f"     ⚠️  Skipping warrant/right")
            continue

        try:
            bars_daily = client.get_bars(symbol, timeframe="1Day", limit=21)
            if bars_daily is None or len(bars_daily) < 5:
                print(f"     ❌ Insufficient daily bars")
                continue

            today_vol = int(bars_daily["volume"].iloc[-1])
            hist_bars = bars_daily.iloc[:-1]
            avg_vol = hist_bars["volume"].mean()
            rel_vol = today_vol / avg_vol if avg_vol > 0 else 1.0

            today_high = float(bars_daily["high"].iloc[-1])
            today_low = float(bars_daily["low"].iloc[-1])
            prev_close = float(bars_daily["close"].iloc[-2]) if len(bars_daily) > 1 else g.price

            print(f"     Volume: {today_vol:,.0f} / {avg_vol:,.0f} avg = {rel_vol:.1f}x")

            # Float data
            float_shares = None
            float_str = "N/A"
            try:
                float_data = float_provider.get_float(symbol)
                if float_data and float_data.float_shares:
                    float_shares = float_data.float_shares
                    float_str = f"{float_shares / 1e6:.1f}M"
                    print(f"     Float: {float_str}")
                else:
                    print(f"     Float: unavailable")
            except Exception:
                print(f"     Float: error")

            # Check for press release catalyst
            pr_hits = scanner.get_hits_for_symbol(symbol)
            if pr_hits:
                best = pr_hits[0]
                print(f"     🔥 PR CATALYST: [{best.sentiment.upper()}] {best.headline[:60]}")
            else:
                print(f"     No press release catalyst")

            # Filter results
            passes_rv = rel_vol >= 5.0
            passes_float = float_shares is None or (float_shares / 1e6) <= 20.0

            if passes_rv and passes_float:
                print(f"     ✅ PASSES ALL FILTERS (relVol={rel_vol:.1f}x, float={float_str})")
            elif passes_rv:
                print(f"     ✅ Passes volume (float unknown/ok)")
            else:
                fails = []
                if not passes_rv:
                    fails.append(f"relVol {rel_vol:.1f}x < 5.0x")
                if float_shares and not passes_float:
                    fails.append(f"float {float_shares / 1e6:.1f}M > 20M")
                print(f"     ❌ Fails: {', '.join(fails)}")

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
                "passes": passes_rv,
                "has_pr_catalyst": len(pr_hits) > 0,
                "pr_sentiment": pr_hits[0].sentiment if pr_hits else None,
                "pr_headline": pr_hits[0].headline if pr_hits else None,
            })

        except Exception as e:
            print(f"     ❌ Error: {e}")

    if not enriched:
        print("\n  No candidates survived enrichment.")
        _print_summary(scanner, raw_gainers, [], equity)
        return

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE 4: SIGNAL GENERATION
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "═" * 78)
    print("  PHASE 4: SIGNAL GENERATION (5-min bars, MACD + pullback)")
    print("═" * 78)

    enriched.sort(key=lambda x: x["rel_vol"], reverse=True)

    signals_found = []
    for c in enriched:
        symbol = c["symbol"]
        pr_tag = " 🔥PR" if c["has_pr_catalyst"] else ""
        print(f"\n  ── {symbol} (${c['price']:.2f}, {c['change_pct']:+.1f}%, "
              f"{c['rel_vol']:.1f}x relVol{pr_tag}) ──")

        try:
            bars = client.get_bars(symbol, timeframe="5Min", limit=100)
            if bars is None or len(bars) < 40:
                print(f"     ❌ Insufficient 5-min bars ({len(bars) if bars is not None else 0})")
                continue

            price = client.get_latest_price(symbol)

            # MACD
            macd_line, signal_line, histogram = calc_macd(
                bars["close"], fast_period=12, slow_period=26, signal_period=9,
            )
            cur_macd = float(macd_line.iloc[-1])
            cur_signal = float(signal_line.iloc[-1])
            cur_hist = float(histogram.iloc[-1])
            atr_val = float(calc_atr(bars["high"], bars["low"], bars["close"], period=14).iloc[-1])

            print(f"     Price: ${price:.2f} | MACD: {cur_macd:.4f} | Signal: {cur_signal:.4f} | Hist: {cur_hist:.4f}")
            print(f"     ATR(14): ${atr_val:.4f}")

            if cur_macd <= 0:
                print(f"     ❌ MACD below zero — no bullish momentum")
                continue
            if cur_macd <= cur_signal:
                print(f"     ❌ MACD below signal — bearish crossover")
                continue

            print(f"     ✅ MACD positive and above signal")

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
            else:
                print(f"     ❌ No valid pullback pattern")

        except Exception as e:
            print(f"     ❌ Error: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    #  PHASE 5: TRADE SIMULATION
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "═" * 78)
    print("  PHASE 5: TRADE SIMULATION")
    print("═" * 78)

    if not signals_found:
        print("\n  No signals generated.")
        if enriched:
            best = enriched[0]
            print(f"  Best candidate: {best['symbol']} ({best['change_pct']:+.1f}%, {best['rel_vol']:.1f}x)")
            if best["has_pr_catalyst"]:
                print(f"  Had PR catalyst: [{best['pr_sentiment']}] {best['pr_headline'][:60]}")
            print(f"  Strategy requires MACD > 0 + clean pullback to enter.")
    else:
        for i, (cand, signal) in enumerate(signals_found):
            symbol = signal.symbol
            pr_tag = " [PR CATALYST]" if cand["has_pr_catalyst"] else ""

            # Position sizing
            max_risk = equity * 0.02
            risk_per_sh = signal.risk_amount
            shares_by_risk = int(max_risk / risk_per_sh) if risk_per_sh > 0 else 0
            shares_by_bp = int(buying_power * 0.90 / signal.entry_price)
            shares = min(shares_by_risk, shares_by_bp)
            cost = shares * signal.entry_price
            total_risk = shares * risk_per_sh
            total_reward = shares * (signal.reward_amount or 0)

            print(f"\n  📈 TRADE #{i + 1}: BUY {shares} shares of {symbol} @ ${signal.entry_price:.2f}{pr_tag}")
            print(f"     Cost: ${cost:.2f} ({cost / equity * 100:.0f}% of equity)")
            print(f"     Risk: ${total_risk:.2f} ({total_risk / equity * 100:.1f}% of equity)")
            print(f"     Potential reward: ${total_reward:.2f}")
            print(f"     Stop: ${signal.stop_price:.2f} | Target: ${signal.target_price:.2f}")

            if cand["has_pr_catalyst"]:
                print(f"     Catalyst: [{cand['pr_sentiment']}] {cand['pr_headline'][:60]}")

            # Simulate outcome
            try:
                cur_price = client.get_latest_price(symbol)
                sim_bars = client.get_bars(symbol, timeframe="5Min", limit=100)
                hit_stop = hit_target = False
                exit_price = cur_price

                if sim_bars is not None:
                    for j in range(len(sim_bars)):
                        if float(sim_bars["low"].iloc[j]) <= signal.stop_price:
                            hit_stop = True
                            exit_price = signal.stop_price
                            break
                        if signal.target_price and float(sim_bars["high"].iloc[j]) >= signal.target_price:
                            hit_target = True
                            exit_price = signal.target_price
                            break

                sim_pnl = (exit_price - signal.entry_price) * shares

                print(f"\n     📊 RESULT:")
                if hit_target:
                    print(f"     🎉 TARGET HIT @ ${exit_price:.2f}")
                    print(f"     P&L: ${sim_pnl:+.2f} ({sim_pnl / equity * 100:+.1f}%)")
                elif hit_stop:
                    print(f"     🛑 STOP HIT @ ${exit_price:.2f}")
                    print(f"     P&L: ${sim_pnl:+.2f} ({sim_pnl / equity * 100:+.1f}%)")
                else:
                    unrealized = (cur_price - signal.entry_price) * shares
                    print(f"     ⏳ Still open — current: ${cur_price:.2f}")
                    print(f"     Unrealized P&L: ${unrealized:+.2f} ({unrealized / equity * 100:+.1f}%)")
            except Exception as e:
                print(f"     ❌ Could not simulate: {e}")

            # Only simulate first trade (1 trade/day limit)
            if i == 0:
                break

    _print_summary(scanner, raw_gainers, signals_found, equity)


def _print_summary(scanner, raw_gainers, signals_found, equity):
    """Print final summary."""

    print("\n" + "═" * 78)
    print("  SUMMARY")
    print("═" * 78)

    status = scanner.get_status()

    print(f"\n  Press Release Scanner:")
    print(f"    Feeds scanned:    {status['feeds_configured']} RSS + FMP")
    print(f"    Total PR hits:    {status['total_hits']}")
    print(f"    Positive:         {status['positive_hits']}")
    print(f"    Negative:         {status['negative_hits']}")
    print(f"    Neutral:          {status['neutral_hits']}")
    print(f"    Unique symbols:   {status['unique_symbols']}")
    if status['positive_symbols']:
        print(f"    Positive tickers: {', '.join(status['positive_symbols'])}")

    print(f"\n  Momentum Scanner:")
    print(f"    Top gainers:      {len(raw_gainers)}")
    in_range = [g for g in raw_gainers if 1.0 <= g.price <= 10.0]
    print(f"    In price range:   {len(in_range)}")

    print(f"\n  Strategy:")
    print(f"    Signals found:    {len(signals_found)}")
    if signals_found:
        cand, sig = signals_found[0]
        has_pr = "YES" if cand["has_pr_catalyst"] else "NO"
        print(f"    Best signal:      {sig.symbol} (strength={sig.strength:.2f}, R:R={sig.risk_reward_ratio:.1f})")
        print(f"    Had PR catalyst:  {has_pr}")

    print(f"\n  Account:")
    progress = (equity - 400) / 3600 * 100
    print(f"    Equity:           ${equity:.2f}")
    print(f"    Goal:             $4,000 ({progress:.1f}% progress)")

    print(f"\n  {'=' * 78}")
    print()


if __name__ == "__main__":
    main()
