<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Dual-Asset Trading Bot - Advanced Alpaca Trading System

This is a sophisticated dual-strategy trading bot using Docker and Portainer for production deployment.

## System Architecture
- **swing_trader.py**: Advanced dual-strategy trading engine with independent risk management
- **Dockerfile**: Production containerization with security hardening
- **docker-compose.yml**: Orchestration with placeholder API keys for secure deployment
- **Requirements**: Alpaca API, pandas, schedule, pytz for professional trading operations

## Trading Strategies

### LTC Scalping Strategy (24/7)
- **Timeframe**: 5-minute bars
- **Schedule**: Every 5 minutes, 24/7 operation
- **Entry**: EMA 9/21 crossover + consecutive green candles confirmation
- **Exit**: Scaled exits - 33% at 0.5%, 33% at 1.0%, 34% at 1.5%
- **Risk**: 0.5% stop loss, 10% max portfolio allocation
- **Position Tracking**: LTCPosition class for multiple concurrent positions

### NVDA Stock Strategy (Market Hours)
- **Timeframe**: 5-minute bars
- **Schedule**: Every 5 minutes during market hours (9:30 AM - 4:00 PM ET)
- **Entry**: EMA 9/21 crossover signals
- **Exit**: 6% take profit / 3% stop loss
- **Restriction**: Only trades during US stock market hours
- **Offset**: 2.5-minute offset from LTC strategy to prevent conflicts

## Development Guidelines
- **Security First**: Use placeholder API keys in docker-compose.yml, real keys only in .env
- **Live Trading**: This bot trades with REAL MONEY - all safety measures critical
- **Dual Strategy**: Maintain independent execution paths for crypto and stock trading
- **Error Handling**: Comprehensive try/catch blocks with detailed logging
- **Position Management**: Track multiple positions with different exit strategies
- **Market Hours**: Respect trading restrictions for stock markets vs 24/7 crypto

## Technical Implementation
- **Portfolio Management**: 10% maximum allocation per asset
- **Position Sizing**: Dynamic calculation based on portfolio value
- **Risk Management**: Independent stop losses and take profits per strategy
- **Scheduling**: Python schedule library with 5-minute intervals
- **Data Sources**: Alpaca historical data clients for stocks and crypto
- **Order Execution**: Market orders with GTC time in force

## Deployment Pattern
- **Portainer Deployment**: Stack-based deployment with environment variables
- **Security**: Never include real API keys in compose files
- **Logging**: Persistent volume for trading logs and performance tracking
- **Restart Policy**: unless-stopped for continuous operation
- **Environment**: All configuration via environment variables

## Code Structure
- **Global Position Tracking**: ltc_positions list for scalping management
- **Market Hours Check**: is_market_hours() function for stock trading restrictions
- **EMA Calculations**: calculate_ema() for technical analysis
- **Dual Execution**: Separate run_ltc_scalping() and run_nvda_trading() functions
- **Portfolio Monitoring**: get_portfolio_value() for allocation management
