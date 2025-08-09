# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **screener-based multi-asset trading bot** for Alpaca Markets that trades with REAL MONEY. The system dynamically selects and trades up to 15 symbols (10 stocks + 5 cryptos) using market screeners and EMA-based technical analysis.

**Critical Context**: This bot evolved from a dual-asset system (hardcoded LTC/NVDA) to a dynamic screener-based multi-symbol system on August 4, 2025. All LTC-specific code has been removed.

## Commands

### Docker Deployment (Production)
```bash
# CRITICAL: Always run from the correct directory
cd "d:\projects\Aplaca Projects\simple_swinger"

# Stop existing container before rebuilding (IMPORTANT!)
docker-compose down

# Build and run with live trading
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker logs alpaca-swing-bot -f
```

### Local Development & Testing
```bash
# Check current positions (requires .env file with API keys)
cd simple_swinger
python check_positions.py

# Run bot locally (for testing - BE CAREFUL, this trades live!)
python swing_trader.py
```

### Container Management
```bash
# Force rebuild without cache (use when code changes aren't reflected)
docker-compose build --no-cache
docker-compose up

# Check container status
docker ps

# Access container shell
docker exec -it alpaca-swing-bot /bin/bash
```

## Architecture

### Core Trading System (`swing_trader.py`)
The monolithic trading engine (751 lines) implements:
- **Dynamic Symbol Selection**: `screen_top_stock_movers()` and `screen_top_crypto_movers()` select symbols based on market performance
- **Multi-Symbol Execution**: `run_multi_symbol_trading()` orchestrates trading across all selected symbols
- **Generic Trading Functions**: `trade_stock_symbol()` and `trade_crypto_symbol()` handle any symbol (replaced old LTC/NVDA-specific functions)
- **Position Management**: `manage_position()` implements stop-loss and take-profit logic with different thresholds for stocks (6%/3%) vs crypto (3%/1.5%)
- **Entry Signal Detection**: `check_entry_signal()` uses EMA 9/21 crossover with green candle confirmation

### Trading Strategy Flow
1. **Symbol Screening** (hourly): Updates `active_stock_symbols` and `active_crypto_symbols` lists
2. **Market Analysis** (5-min): Checks entry signals for all active symbols
3. **Position Entry**: Places market orders when EMA crossover + bullish candles detected
4. **Risk Management**: Monitors positions for stop-loss/take-profit thresholds
5. **Portfolio Allocation**: Limits total exposure to 10% (6% stocks, 4% crypto) with equal weighting

### Key Configuration Variables
Environment variables control behavior (set in `.env` or Docker environment):
- `ALPACA_LIVE_API_KEY` / `ALPACA_LIVE_API_SECRET`: **NEVER hardcode these**
- `MAX_STOCK_SYMBOLS=10`: Maximum concurrent stock positions
- `MAX_CRYPTO_SYMBOLS=5`: Maximum concurrent crypto positions
- `MAX_POSITION_PERCENT=0.20`: Maximum 20% of portfolio per position
- `MAX_CASH_PER_TRADE=0.10`: Maximum 10% of available cash per trade
- `MIN_PRICE=5.0` / `MAX_PRICE=500.0`: Price range filters
- `MIN_DAILY_VOLUME=1000000`: Minimum $1M daily volume
- `SCREENER_UPDATE_HOURS=1`: Symbol list refresh frequency

## Critical Development Rules

### API Integration
- The bot uses Alpaca's `ScreenerClient` with `MarketMoversRequest` for stock screening
- Crypto screening analyzes 24-hour performance of major pairs (BTC, ETH, LTC, etc.)
- All API calls should include error handling and logging
- API validation happens on startup via `validate_api_access()`

### Position Management
- **Stocks**: Trade only during market hours (9:30 AM - 4:00 PM ET)
- **Crypto**: Trade 24/7
- **Position Sizing**: 
  - Maximum 20% of portfolio value per position
  - Maximum 10% of available cash per trade
  - Uses the smaller of the two limits
  - **NO MARGIN**: Never borrows money - only trades with available cash
  - Blocks any trade that would exceed cash balance
- **Risk Parameters**: 
  - Stocks: 6% take profit / 3% initial stop-loss / 2% trailing stop
  - Crypto: 3% take profit / 1.5% initial stop-loss / 1% trailing stop
- **Trailing Stop-Loss**: Automatically adjusts stop-loss upward as price rises, locking in profits
- **Scaling Out Strategy** (Partial Profit Taking):
  - **Stocks**: Take 25% at +3%, 33% at +5%, 50% at +8%
  - **Crypto**: Take 33% at +1.5%, 50% at +2.5%, 50% at +3.5%
  - Remaining position continues with trailing stop

### Docker Deployment Issues & Solutions
- **Wrong Directory Error**: Always `cd` to `simple_swinger` folder before docker commands
- **Cached Code Problem**: Use `docker-compose down` before rebuilding
- **Container Using Old Code**: Rebuild with `--no-cache` flag
- **API Keys Not Loading**: Check Portainer environment variables or `.env` file mounting

## Known Issues & Gotchas

1. **No Duplicate Order Prevention**: Could place multiple orders for same signal
2. **No Stale Data Detection**: Could trade on old data if API delays occur
3. **Abrupt Error Handling**: Bot stops on many errors instead of recovering
4. **Monolithic Architecture**: All logic in single 751-line file makes testing difficult

## System Evolution History

### What Was Removed (Legacy Code)
- `run_ltc_scalping()`: Replaced with generic `trade_crypto_symbol()`
- `run_nvda_trading()`: Replaced with generic `trade_stock_symbol()`
- `LTCPosition` class: Replaced with generic `PositionTracker`
- All hardcoded symbol-specific logic

### Current Implementation
- Generic functions work with any symbol (stocks or crypto)
- Dynamic symbol selection via screeners (fallback symbols only on error)
- Unified position management with asset-type-specific parameters
- Hourly symbol list updates based on market performance

## Testing & Safety

### Before Making Changes
1. Review this file for system context
2. Check `check_positions.py` to understand current positions
3. Test with paper trading if possible (change `paper=False` to `paper=True` in trading client)
4. Ensure no hardcoded API keys in code

### Deployment Checklist
1. Stop existing container: `docker-compose down`
2. Rebuild without cache: `docker-compose build --no-cache`
3. Check logs after deployment: `docker logs alpaca-swing-bot -f`
4. Verify API access in logs: Look for "✅ Account access validated"
5. Monitor first trading cycle for errors

## Portfolio Status & Monitoring

- Current portfolio value: Check logs for "Portfolio Value: $XXX.XX"
- Active positions: Run `python check_positions.py` or check container logs
- Symbol updates: Look for "📊 Selected X stocks/cryptos" in logs
- Trade execution: Watch for "🚀 [SYMBOL] BUY/SELL" messages

## Future Improvements Priority

1. **Modularize** `swing_trader.py` into separate modules (screener, strategy, risk, orders)
2. **Add retry logic** for API calls with exponential backoff
3. **Implement trailing stops** to lock in profits
4. **Add daily loss limits** with automatic shutdown
5. **Create health check endpoint** for monitoring