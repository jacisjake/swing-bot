<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Screener-Based Multi-Asset Trading Bot - AI Assistant Memory & Instructions

**Last Updated**: August 4, 2025  
**System Version**: Screener-Based Multi-Symbol Trading System  
**Portfolio Value**: $155.11 (Live Trading Account)

## 🧠 AI Assistant Memory & Context

### System Evolution History
- **Original**: Dual-asset system (hardcoded LTC/NVDA trading)
- **Current**: Dynamic screener-based multi-symbol trading system
- **Transformation Date**: August 4, 2025
- **Key Change**: Removed all LTC-specific references and implemented generic multi-symbol approach

### Critical Context to Remember
- **Live Trading**: This bot trades with REAL MONEY 
- **User Request**: Wanted "screener features from chat" that were missing from original implementation
- **Problem Solved**: Container was running old dual-asset code instead of new screener system
- **Solution Applied**: Completely removed LTC-specific functions and references

## 🎯 Current System Architecture

### Core Components
- **swing_trader.py**: Screener-based multi-symbol trading engine
- **Dockerfile**: Production containerization with security hardening  
- **docker-compose.yml**: Orchestration with environment variable configuration
- **Requirements**: Alpaca API, pandas, schedule, pytz for professional trading operations

### Dynamic Symbol Selection (NEW)
- **Stock Screener**: Uses Alpaca's GAINERS screener with volume/price filters
- **Crypto Screener**: Performance-based selection from major crypto pairs
- **Update Frequency**: Every 1 hour (configurable via SCREENER_UPDATE_HOURS)
- **Criteria**: $5-$500 price range, $1M+ daily volume, top movers
- **Limits**: Max 10 stocks, Max 5 cryptos (configurable)

## 📈 Trading Strategies

### Multi-Symbol Stock Trading (Market Hours Only)
- **Timeframe**: 5-minute bars
- **Schedule**: Every 5 minutes during market hours (9:30 AM - 4:00 PM ET)
- **Entry**: EMA 9/21 crossover + consecutive green candles
- **Exit**: 6% take profit / 3% stop loss
- **Symbols**: Dynamic via screener (fallback: NVDA)
- **Position Sizing**: Equal allocation across active symbols

### Multi-Symbol Crypto Trading (24/7)
- **Timeframe**: 5-minute bars  
- **Schedule**: Every 5 minutes, 24/7 operation
- **Entry**: EMA 9/21 crossover + consecutive green candles
- **Exit**: 3% take profit / 1.5% stop loss (tighter for crypto volatility)
- **Symbols**: Dynamic via performance screener (fallback: LTC/USD)
- **Position Sizing**: Equal allocation across active symbols

## 🛡️ Risk Management

### Portfolio Allocation
- **Total Risk**: 10% maximum allocation (MAX_PORTFOLIO_PERCENT=0.10)
- **Stock Allocation**: 60% of total risk (6% of portfolio)
- **Crypto Allocation**: 40% of total risk (4% of portfolio)
- **Per Symbol**: Equal distribution within asset class

### Position Management
- **Entry Conditions**: EMA crossover + bullish candle confirmation
- **Risk Thresholds**: Different for stocks vs crypto due to volatility
- **Order Type**: Market orders with GTC time in force
- **Position Tracking**: Generic PositionTracker class (removed LTC-specific tracking)

## 🔧 Configuration Variables

### Screener Settings
- `MAX_STOCK_SYMBOLS=10`: Maximum stocks to trade simultaneously
- `MAX_CRYPTO_SYMBOLS=5`: Maximum cryptos to trade simultaneously  
- `MIN_DAILY_VOLUME=1000000`: Minimum $1M daily volume filter
- `MIN_PRICE=5.0`: Minimum $5 per share
- `MAX_PRICE=500.0`: Maximum $500 per share
- `SCREENER_UPDATE_HOURS=1`: Symbol refresh frequency

### Risk Management
- `MAX_PORTFOLIO_PERCENT=0.10`: 10% total portfolio risk
- `STOP_LOSS_PERCENT=0.005`: Legacy variable (now unused)
- `TAKE_PROFIT_1/2/3`: Legacy variables (now using fixed percentages)

## 💻 Development Guidelines

### Code Maintenance Rules
- **NO Legacy References**: Removed all LTC-specific functions and variables
- **Generic Approach**: All functions work with any symbol (stocks or crypto)
- **Screener First**: Always use dynamic symbol selection, fallback to static symbols only on error
- **Risk Management**: Different thresholds for stocks (6%/3%) vs crypto (3%/1.5%)
- **Market Hours**: Respect trading restrictions for stocks vs 24/7 crypto

### Security & Deployment
- **API Keys**: Never hardcode in files, use environment variables only
- **Live Trading**: All changes must be thoroughly tested before deployment
- **Docker**: Rebuild with --no-cache when code changes to prevent caching issues
- **Logging**: Comprehensive logging for all trading decisions and errors

## 📝 AI Assistant Instructions

### When User Says "Remember..."
- **Action**: Update this file with new information in appropriate section
- **Format**: Add timestamp and context for future reference
- **Location**: Add to "Memory Notes" section below

### Code Modification Rules
- **Always**: Check this file first for current system state and context
- **Never**: Revert to old dual-asset approach or LTC-specific code
- **Verify**: Current system is screener-based multi-symbol trading
- **Maintain**: Generic functions that work with any symbol

### Problem-Solving Approach
1. **Context Check**: Review this file for relevant background
2. **System Understanding**: Current state is screener-based, not dual-asset
3. **Safe Changes**: Test thoroughly before suggesting live trading modifications
4. **Documentation**: Update this file when system changes are made

## 📚 Memory Notes

### August 4, 2025 - Major System Transformation
- **Issue**: User noted "still a lot of references to LTC in the script"
- **Root Cause**: Code had both old LTC-specific functions AND new screener functions
- **Solution**: Removed ALL legacy LTC/NVDA specific functions
- **Result**: Clean screener-based system with generic symbol handling
- **Container Status**: Successfully rebuilt and running new system

### Key Functions Removed
- `run_ltc_scalping()`: Replaced with `trade_crypto_symbol()`
- `run_nvda_trading()`: Replaced with `trade_stock_symbol()`  
- `LTCPosition` class: Replaced with generic `PositionTracker`
- All LTC-specific position management and tracking

### Key Functions Added
- `screen_top_stock_movers()`: Dynamic stock selection
- `screen_top_crypto_movers()`: Dynamic crypto selection
- `update_symbol_lists()`: Periodic screener updates
- `run_multi_symbol_trading()`: Main execution loop
- Generic symbol trading functions for any asset

### Security Reminder - API Keys
- **CRITICAL**: Keep API keys in .env file ONLY, never hardcode in source files
- **Location**: Use ALPACA_LIVE_API_KEY and ALPACA_LIVE_API_SECRET environment variables
- **Docker**: API keys passed via environment variables in docker-compose.yml
- **Security**: Never commit real API keys to version control

### Deployment Reminder - Correct Directory
- **CRITICAL**: Always launch Docker from the correct folder: `d:\projects\Aplaca Projects\simple_swinger`
- **CRITICAL**: Always take Docker container down before rebuilding: `docker-compose down`
- **Command Sequence**: 
  1. `cd "d:\projects\Aplaca Projects\simple_swinger"`
  2. `docker-compose down`
  3. `docker-compose up --build`
- **Issue**: Launching from wrong directory causes "no configuration file provided: not found" error
- **Issue**: Not stopping container before rebuild can use old cached code
- **Docker Context**: docker-compose.yml and Dockerfile must be in current working directory

### API Documentation References
- **Alpaca Screener API**: https://alpaca.markets/sdks/python/api_reference/data/stock/screener.html
- **Alpaca Movers API**: https://docs.alpaca.markets/reference/movers-1
- **Fixed**: Updated to use `ScreenerClient` and `MarketMoversRequest` instead of deprecated classes
- **Note**: Screener functionality uses market movers API for finding gainers/losers

---
**Note**: This document serves as the AI assistant's memory and working context. Update it whenever the user asks to "remember" something or when significant system changes are made.
