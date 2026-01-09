# Swing Trader

Algorithmic day/swing trading system for Alpaca API. Designed for small accounts with aggressive risk management.

## Features

- **MACD Strategy**: Buy on MACD crossover + 2 green candles, exit on red candle + MACD cross down
- **Multi-asset support**: Stocks (active), Crypto (paused)
- **Risk-first design**: Position sizing based on risk, not capital
- **ATR-based stops**: Dynamic stop-loss based on volatility
- **Portfolio limits**: Drawdown protection, daily loss limits, position limits
- **Live dashboard**: Real-time status at http://localhost:8080

## Current Strategy

**Stock Trading (MACD Crossover)**
- Timeframe: 5-minute bars
- MACD settings: 8-17-9 (fast)
- Entry: MACD crosses above signal line + 2 consecutive green candles
- Exit: Red candle + MACD crosses below signal line
- Stop-loss: 2x ATR
- Signal check: Every 5 minutes during market hours

## Project Structure

```
swing-trader/
├── src/
│   ├── bot/
│   │   ├── main.py            # Main trading bot controller
│   │   ├── signals/           # Strategy implementations
│   │   │   ├── macd.py        # MACD crossover strategy
│   │   │   ├── breakout.py    # Donchian breakout (deprecated)
│   │   │   └── mean_reversion.py  # RSI+BB mean reversion
│   │   ├── screener.py        # Stock/crypto screeners
│   │   ├── scheduler.py       # Job scheduling
│   │   └── api.py             # Dashboard API
│   │
│   ├── core/
│   │   ├── alpaca_client.py   # Unified Alpaca API client
│   │   ├── position_manager.py # Track positions and P&L
│   │   └── order_executor.py  # Order placement with retries
│   │
│   ├── risk/
│   │   ├── position_sizer.py  # Risk-based position sizing
│   │   ├── stop_manager.py    # Stop-loss calculations
│   │   └── portfolio_limits.py # Portfolio-level risk limits
│   │
│   └── data/
│       └── indicators.py      # Technical indicators (MACD, RSI, BB, ATR, etc.)
│
├── config/
│   └── settings.py            # Pydantic settings management
│
├── deploy/
│   ├── deploy-remote.sh       # Remote deployment script
│   └── podman-compose.yml     # Container orchestration
│
└── tests/
    └── unit/                  # Unit tests
```

## Quick Start

### 1. Setup

```bash
cd swing-trader

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

### 2. Configure

Edit `.env` with your Alpaca credentials:

```env
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
TRADING_MODE=paper  # or 'live'
```

### 3. Run

```bash
python scripts/run_bot.py
```

Dashboard available at http://localhost:8080

### 4. Deploy to Remote Server

```bash
cd deploy
./deploy-remote.sh user@host --build
```

## Configuration

Key settings in `src/bot/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `macd_fast_period` | 8 | MACD fast EMA period |
| `macd_slow_period` | 17 | MACD slow EMA period |
| `macd_signal_period` | 9 | MACD signal line period |
| `stock_timeframe` | 5Min | Bar timeframe for signals |
| `stock_check_interval_minutes` | 5 | Signal check frequency |
| `stock_atr_stop_multiplier` | 2.0 | ATR multiplier for stops |
| `enable_crypto_trading` | false | Enable/disable crypto |

## Risk Management

- Max 2% account risk per trade
- Position size calculated from stop distance
- ATR-based stops adapt to volatility
- Portfolio limits: max drawdown, daily loss, position count

## License

MIT
