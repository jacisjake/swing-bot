# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **screener-based multi-asset trading bot** for Alpaca Markets that trades with REAL MONEY. The system dynamically selects and trades up to 15 symbols (10 stocks + 5 cryptos) using market screeners and EMA-based technical analysis.

**Critical Context**: This bot evolved from a dual-asset system (hardcoded LTC/NVDA) to a dynamic screener-based multi-symbol system on August 4, 2025. All LTC-specific code has been removed.

## Commands

### Prerequisites
```bash
# Install Python dependencies (for local development)
cd simple_swinger
pip install -r requirements.txt

# Environment setup - copy and configure
cp .env.example .env  # Edit with your Alpaca API keys
```

### Docker Deployment (Production)
```bash
# CRITICAL: Always run from the correct directory
cd "D:\projects\Aplaca Projects\simple_swinger"

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

# Test with paper trading (modify swing_trader.py: paper=True in TradingClient)
# Look for: trading_client = TradingClient(API_KEY, SECRET_KEY, paper=False)
# Change to: trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
```

### Container Management
```bash
# Force rebuild without cache (use when code changes aren't reflected)
docker-compose build --no-cache
docker-compose up

# Check container status
docker ps

# Access container shell for debugging
docker exec -it alpaca-swing-bot /bin/bash

# View container logs in real-time
docker logs alpaca-swing-bot -f --tail=100
```

## Architecture

### Core Trading System (`swing_trader.py`)
The monolithic trading engine (~1050+ lines) contains all trading logic in a single file:

**Key Classes:**
- `PositionTracker`: Tracks individual position state (entry price, quantity, order ID)

**Core Functions (in execution order):**
1. **Initialization & Validation**:
   - `validate_api_access()`: Tests API connectivity on startup
   - `get_portfolio_value()`: Gets current account balance
   
2. **Market Screening** (Hourly):
   - `screen_top_stock_movers()`: Uses Alpaca's MarketMoversRequest for stock selection
   - `screen_top_crypto_movers()`: Custom crypto screening based on 24h performance
   - `update_symbol_lists()`: Updates global symbol lists with screener results
   
3. **Trading Orchestration** (5-minute cycles):
   - `run_multi_symbol_trading()`: Main trading loop coordinator
   - `trade_stock_symbol()` / `trade_crypto_symbol()`: Symbol-specific trading logic
   
4. **Technical Analysis**:
   - `calculate_ema()`, `calculate_macd()`, `calculate_rsi()`: Technical indicators
   - `check_entry_signal()`: 4 different entry strategies using MACD/RSI/EMA combinations
   - `check_exit_signal()`: Technical exit signals (RSI overbought, MACD bearish crossover, momentum loss)
   
5. **Position Management**:
   - `manage_position_with_indicators()`: Enhanced position management with technical exits
   - `manage_position()`: Traditional stop-loss/take-profit backup system
   - `calculate_position_size_per_symbol()`: Position sizing with cash/portfolio limits
   
6. **Order Execution**:
   - `place_order()`: Market order placement with error handling
   - `check_existing_positions()`: Position status verification

### Trading Strategy Flow
1. **Symbol Screening** (hourly): Updates `active_stock_symbols` and `active_crypto_symbols` lists
2. **Market Analysis** (5-min): Checks multiple entry signals for all active symbols using:
   - **Strategy 1**: MACD crossover + RSI (30-70) + Green candle
   - **Strategy 2**: EMA crossover + MACD bullish + Green candle
   - **Strategy 3**: RSI oversold bounce (from <30 to >35) + MACD momentum
   - **Strategy 4**: Strong momentum (all indicators bullish + RSI increasing)
3. **Position Entry**: Places market orders when any entry strategy triggers
4. **Technical Exit Monitoring**: Continuously checks for:
   - RSI overbought (>75) for profit taking
   - MACD bearish crossover for trend reversal
   - Momentum loss (declining MACD histogram)
   - EMA death cross (bearish signal)
5. **Risk Management**: Traditional stop-loss/take-profit as backup to technical exits
6. **Portfolio Allocation**: Limits total exposure to 10% (6% stocks, 4% crypto) with equal weighting

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

## Development & Testing Tools

### Position Monitoring (`check_positions.py`)
- **Purpose**: Standalone script to check current positions without running the full bot
- **Usage**: `python check_positions.py` (requires `.env` file)
- **Features**: Shows portfolio value, position details, P&L, and exit threshold analysis
- **Safety**: Read-only - no trading operations performed

### Configuration Files
- **`.env`**: Live API keys and trading parameters (never commit!)
- **`.env.example`**: Template with parameter descriptions
- **`.env.test`**: Test environment configuration
- **`docker-compose.yml`**: Production container configuration
- **`requirements.txt`**: Python dependencies (pandas, alpaca-py, schedule, etc.)

### Logging System
- **Container**: `/app/logs/trading.log` (mounted to `./logs/trading.log` on host)
- **Format**: Timestamped with INFO/ERROR levels
- **Key Messages**: Look for "✅ Account access validated", "📊 Selected X stocks/cryptos", "🚀 [SYMBOL] BUY/SELL"

## Known Issues & Gotchas

1. **No Duplicate Order Prevention**: Could place multiple orders for same signal during rapid cycles
2. **No Stale Data Detection**: Could trade on old data if Alpaca API experiences delays
3. **Abrupt Error Handling**: Bot stops completely on many errors instead of graceful recovery
4. **Monolithic Architecture**: All logic in single 1050+ line file makes testing and debugging difficult
5. **Memory Leaks**: Long-running container may accumulate data in global variables over time
6. **API Rate Limits**: No built-in rate limiting for Alpaca API calls

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
1. **Review System Context**: Read this file completely for current system understanding
2. **Check Current State**: Run `python check_positions.py` to see active positions
3. **Enable Paper Trading**: In `swing_trader.py:34`, change `paper=False` to `paper=True` in TradingClient
4. **Verify Environment**: Ensure `.env` file exists with valid API keys (never commit!)
5. **Test Locally First**: Run `python swing_trader.py` locally before containerizing

### Code Modification Guidelines
- **API Keys**: Always use environment variables, never hardcode
- **Error Handling**: Add try/catch blocks for all API calls
- **Logging**: Add meaningful log messages for debugging
- **Position Limits**: Respect MAX_POSITION_PERCENT and MAX_CASH_PER_TRADE limits
- **Market Hours**: Ensure stock trades only happen during market hours (9:30 AM - 4:00 PM ET)

### Deployment Checklist
1. **Stop Container**: `docker-compose down` (from simple_swinger directory)
2. **Rebuild Clean**: `docker-compose build --no-cache` 
3. **Deploy**: `docker-compose up -d --build`
4. **Monitor Startup**: `docker logs alpaca-swing-bot -f --tail=50`
5. **Verify API Access**: Look for "✅ Account access validated" in logs
6. **Check Symbol Selection**: Look for "📊 Selected X stocks/cryptos" messages
7. **Monitor First Cycle**: Watch for entry/exit signals and order execution
8. **Verify Positions**: Run `check_positions.py` to confirm expected state

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