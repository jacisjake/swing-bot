#!/usr/bin/env python3
"""
Trading bot CLI entry point.

Usage:
    python scripts/run_bot.py           # Run with default config
    python scripts/run_bot.py --help    # Show help
    python scripts/run_bot.py --status  # Show bot status
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bot.config import get_bot_config
from src.bot.main import TradingBot, run_bot


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Swing Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_bot.py                    # Start the bot
    python scripts/run_bot.py --dry-run          # Show what would be monitored
    python scripts/run_bot.py --status           # Show current status

Environment variables (or in .env):
    ALPACA_API_KEY       - Alpaca API key
    ALPACA_SECRET_KEY    - Alpaca secret key
    TRADING_MODE         - paper or live (default: paper)
    BOT_STOCK_WATCHLIST  - Comma-separated stock symbols
    BOT_CRYPTO_WATCHLIST - Comma-separated crypto symbols
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration and exit without running",
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current account and position status",
    )

    parser.add_argument(
        "--check-signals",
        action="store_true",
        help="Check for signals once and exit",
    )

    return parser.parse_args()


def show_config():
    """Display current configuration."""
    config = get_bot_config()

    print("=" * 60)
    print("TRADING BOT CONFIGURATION")
    print("=" * 60)
    print()
    print(f"Trading Mode: {config.trading_mode.value.upper()}")
    print()
    print("Risk Settings:")
    print(f"  Max Position Risk: {config.max_position_risk_pct:.1%}")
    print(f"  Max Drawdown: {config.max_drawdown_pct:.1%}")
    print(f"  Max Positions: {config.max_positions}")
    print()
    print("Scheduler:")
    print(f"  Stock Check Interval: {config.stock_check_interval_minutes} min")
    print(f"  Crypto Check Interval: {config.crypto_check_interval_minutes} min")
    print(f"  Position Monitor: {config.position_monitor_interval_minutes} min")
    print()
    print("Stock Strategy (MACD):")
    print(f"  Fast Period: {config.macd_fast_period} days")
    print(f"  Slow Period: {config.macd_slow_period} days")
    print(f"  Signal Period: {config.macd_signal_period}x")
    print(f"  ATR Stop Multiplier: {config.stock_atr_stop_multiplier}x")
    print()
    print("Crypto Strategy (Mean Reversion):")
    print(f"  RSI Period: {config.crypto_rsi_period}")
    print(f"  RSI Oversold: {config.crypto_rsi_oversold}")
    print(f"  RSI Exit: {config.crypto_rsi_exit}")
    print(f"  BB Period: {config.crypto_bb_period}")
    print()
    print("Watchlists:")
    print(f"  Stocks: {config.stock_symbols}")
    print(f"  Crypto: {config.crypto_symbols}")
    print()
    print("=" * 60)


async def show_status():
    """Display current account and position status."""
    from src.core.alpaca_client import AlpacaClient
    from src.core.position_manager import PositionManager

    client = AlpacaClient()
    position_manager = PositionManager()

    print("=" * 60)
    print("ACCOUNT STATUS")
    print("=" * 60)
    print()

    try:
        account = client.get_account()
        print(f"Equity:       ${float(account['equity']):,.2f}")
        print(f"Buying Power: ${float(account['buying_power']):,.2f}")
        print(f"Cash:         ${float(account['cash']):,.2f}")
        print(f"Status:       {account['status']}")
        print()

        positions = client.get_positions()
        print(f"Open Positions: {len(positions)}")
        print()

        if positions:
            print("Positions:")
            print("-" * 60)
            total_value = 0
            total_pnl = 0

            for p in positions:
                symbol = p["symbol"]
                qty = float(p["qty"])
                entry = float(p["avg_entry_price"])
                current = float(p["current_price"])
                value = float(p["market_value"])
                pnl = float(p["unrealized_pl"])
                pnl_pct = float(p["unrealized_plpc"]) * 100

                total_value += value
                total_pnl += pnl

                arrow = "▲" if pnl >= 0 else "▼"
                print(f"  {symbol:8} {qty:>8.4f} @ ${entry:>8.2f} → ${current:>8.2f}")
                print(f"           Value: ${value:>8.2f}  P&L: {arrow} ${pnl:>8.2f} ({pnl_pct:+.1f}%)")

            print("-" * 60)
            arrow = "▲" if total_pnl >= 0 else "▼"
            print(f"  Total Value: ${total_value:,.2f}  Total P&L: {arrow} ${total_pnl:,.2f}")

    except Exception as e:
        print(f"Error: {e}")

    print()
    print("=" * 60)


async def check_signals_once():
    """Check for signals once and display results."""
    from src.bot.config import get_bot_config
    from src.bot.signals.macd import MACDStrategy
    from src.bot.signals.mean_reversion import MeanReversionStrategy
    from src.core.alpaca_client import AlpacaClient

    config = get_bot_config()
    client = AlpacaClient()

    stock_strategy = MACDStrategy(
        fast_period=config.macd_fast_period,
        slow_period=config.macd_slow_period,
        signal_period=config.macd_signal_period,
        atr_stop_multiplier=config.stock_atr_stop_multiplier,
    )
    crypto_strategy = MeanReversionStrategy(
        rsi_period=config.crypto_rsi_period,
        rsi_oversold=config.crypto_rsi_oversold,
    )

    print("=" * 60)
    print("SIGNAL CHECK")
    print("=" * 60)
    print()

    print("Checking stocks...")
    for symbol in config.stock_symbols:
        try:
            bars = client.get_bars(symbol, timeframe=config.stock_timeframe, limit=50)
            if bars is not None and len(bars) >= 25:
                price = client.get_latest_price(symbol)
                signal = stock_strategy.generate(symbol, bars, price)
                if signal:
                    print(f"  ✓ {symbol}: {signal.direction.value.upper()} @ ${signal.entry_price:.2f}")
                    print(f"      Stop: ${signal.stop_price:.2f}, Target: ${signal.target_price:.2f}")
                    print(f"      Strength: {signal.strength:.2f}, R:R: {signal.risk_reward_ratio:.1f}")
                else:
                    print(f"  - {symbol}: No signal")
        except Exception as e:
            print(f"  ! {symbol}: Error - {e}")

    print()
    print("Checking crypto...")
    for symbol in config.crypto_symbols:
        try:
            bars = client.get_bars(symbol, timeframe="1Hour", limit=50)
            if bars is not None and len(bars) >= 25:
                price = client.get_latest_price(symbol)
                signal = crypto_strategy.generate(symbol, bars, price)
                if signal:
                    print(f"  ✓ {symbol}: {signal.direction.value.upper()} @ ${signal.entry_price:.2f}")
                    print(f"      Stop: ${signal.stop_price:.2f}, Target: ${signal.target_price:.2f}")
                    print(f"      Strength: {signal.strength:.2f}")
                    print(f"      RSI: {signal.metadata.get('rsi', 'N/A')}")
                else:
                    print(f"  - {symbol}: No signal")
        except Exception as e:
            print(f"  ! {symbol}: Error - {e}")

    print()
    print("=" * 60)


async def run_with_api():
    """Run bot with API server."""
    import uvicorn
    from src.bot.api import app, set_bot
    from src.bot.config import get_bot_config
    from src.bot.main import TradingBot, setup_signal_handlers

    config = get_bot_config()
    bot = TradingBot(config)
    setup_signal_handlers(bot)

    # Give API access to bot
    set_bot(bot)

    # Create API server config
    api_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="warning",
    )
    api_server = uvicorn.Server(api_config)

    # Run bot and API concurrently
    await asyncio.gather(
        bot.start(),
        api_server.serve(),
    )


def main():
    """Main entry point."""
    args = parse_args()

    if args.dry_run:
        show_config()
        return 0

    if args.status:
        asyncio.run(show_status())
        return 0

    if args.check_signals:
        asyncio.run(check_signals_once())
        return 0

    # Run the bot with API
    print("Starting trading bot...")
    print("Dashboard: http://localhost:8080")
    print("Press Ctrl+C to stop")
    print()

    try:
        asyncio.run(run_with_api())
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
