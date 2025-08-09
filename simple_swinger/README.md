# 🚀 Simple Swinger - Screener-Based Multi-Symbol Trading Bot

An intelligent, production-ready swing trading bot for Alpaca Markets using Docker and Portainer. Features dynamic symbol selection via market screeners for both stocks and cryptocurrency trading.

⚠️ **LIVE TRADING WARNING**: This bot trades with real money. Test thoroughly and monitor all trades.

## 🌟 Features

- **🔍 Smart Screener**: Automatically selects top 10 stocks + 5 cryptos using market movers API
- **📊 Multi-Symbol Trading**: Trades up to 15 symbols simultaneously with dynamic allocation
- **⏰ 24/7 Operation**: Stocks during market hours, crypto around the clock
- **🎯 EMA Strategy**: 9/21 EMA crossover with confirmation signals
- **🛡️ Risk Management**: Per-symbol stop losses and take profits
- **🐳 Docker Deployment**: Easy deployment with Portainer
- **💰 Live Trading**: Real money trading with Alpaca Markets

## 📁 Project Structure

```
simple_swinger/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── swing_trader.py
├── DEPLOYMENT.md
└── PORTAINER_DEPLOYMENT.md
```

## 🔧 Setup

### 1. Prerequisites
- Docker installed on your VPS
- Portainer running on your server
- Alpaca Markets account with API keys
- Crypto trading enabled on your Alpaca account

### 2. Configuration
Replace the API keys in `.env` file or Portainer environment variables:
```bash
# Live Trading API Keys
ALPACA_LIVE_API_KEY=your_live_api_key_here
ALPACA_LIVE_API_SECRET=your_live_secret_key_here

# Trading Configuration
STOCK_SYMBOL=NVDA                    # Fallback stock symbol
CRYPTO_SYMBOL=LTC/USD               # Fallback crypto symbol
MAX_STOCK_SYMBOLS=10                # Max stocks to trade simultaneously
MAX_CRYPTO_SYMBOLS=5                # Max cryptos to trade simultaneously
MAX_POSITION_PERCENT=0.20           # 20% max of portfolio per position
MAX_CASH_PER_TRADE=0.10            # 10% max of available cash per trade
SCREENER_UPDATE_HOURS=1             # Update symbol list every hour

# Risk Management
STOP_LOSS_PERCENT=0.005             # 0.5% stop loss
TAKE_PROFIT_1=0.005                 # 0.5% first take profit
TAKE_PROFIT_2=0.010                 # 1.0% second take profit
TAKE_PROFIT_3=0.015                 # 1.5% third take profit

# Screener Criteria
MIN_PRICE=5.0                       # Minimum $5 per share
MAX_PRICE=500.0                     # Maximum $500 per share
MIN_DAILY_VOLUME=1000000            # Minimum $1M daily volume
```

### 3. Deploy via Portainer
1. Log into Portainer
2. Go to **Stacks → Add Stack**
3. Name: `simple-swinger`
4. **Edit the API keys** in the compose content before deploying
5. Paste the updated `docker-compose.yml` content
6. Deploy the stack

## 📊 Screener-Based Multi-Symbol Strategy

### **🔍 Dynamic Symbol Selection:**
- **Stocks**: Top 10 market movers from Alpaca's screener API
- **Cryptos**: Top 5 performing cryptos based on 24h performance
- **Criteria**: $5-$500 price range, $1M+ daily volume
- **Updates**: Symbol list refreshes every hour

### **📈 Trading Strategy (Per Symbol):**
- **Timeframe**: 5-minute bars for all symbols
- **Indicators**: EMA 9/21 crossover with green candle confirmation
- **Entry**: Long on EMA crossover + bullish candle
- **Risk Management**: 
  - Stocks: 6% take profit / 3% stop loss
  - Crypto: 3% take profit / 1.5% stop loss (tighter due to volatility)

### **💼 Position Sizing:**
- **Per Position Limit**: Maximum 20% of total portfolio value
- **Per Trade Limit**: Maximum 10% of available cash
- **Actual Size**: Uses the smaller of the two limits
- **Multiple Positions**: Can hold up to 15 concurrent positions (10 stocks + 5 cryptos)

## 🛡️ Risk Management

### **🎯 Per-Symbol Risk Controls:**
- **Stop Loss**: Automatic exit on 3% loss (stocks) / 1.5% loss (crypto)
- **Take Profit**: Automatic exit on 6% gain (stocks) / 3% gain (crypto)
- **Position Sizing**: 20% portfolio max, 10% cash max per position
- **Max Exposure**: Up to 100% theoretically (15 positions × 20% each if fully allocated)

### **📊 Portfolio Protection:**
- **Diversification**: Up to 15 symbols (10 stocks + 5 cryptos)
- **Equal Weighting**: Risk spread across all positions
- **Market Hours**: Stocks only trade during market hours
- **24/7 Crypto**: Cryptocurrency trading continues around the clock

### **⚡ Real-Time Monitoring:**
- **5-minute cycles**: Continuous position monitoring
- **Automatic exits**: Stop loss/take profit override strategy signals
- **Symbol rotation**: Hourly screener updates for fresh opportunities

## 🕒 Trading Schedule

| Time Period | Assets | Active Symbols | Frequency |
|-------------|--------|----------------|-----------|
| **Market Hours** (9:30 AM - 4:00 PM ET, Mon-Fri) | Stocks + Crypto | Up to 15 symbols | Every 5 minutes |
| **After Hours** (All other times) | Crypto Only | Up to 5 symbols | Every 5 minutes |
| **Screener Updates** | Symbol Selection | Refresh lists | Every hour |

### **🔄 Operation Cycle:**
1. **Symbol Screening** (Hourly): Update active symbol lists
2. **Market Analysis** (5min): Check entry signals for all symbols  
3. **Position Management** (5min): Monitor exits for existing positions
4. **Risk Monitoring** (Continuous): Stop loss and take profit checks

## 📝 Logging & Monitoring

### View Logs in Portainer
- Navigate to: **Containers → alpaca-swing-bot → Logs**

### Log Files
- Container logs: `/app/logs/trading.log`
- Host logs: `./logs/trading.log` (mounted volume)

## 🔒 Security Notes

- **Do not expose** your bot container to the internet
- Use **firewalls** and limit open ports
- Store API keys securely

## ⚡ Features

- **🤖 Automated Trading**: Runs every 5 minutes with intelligent symbol selection
- **🔍 Market Screener**: Dynamic symbol selection based on market performance
- **📊 Multi-Symbol**: Trades up to 15 symbols simultaneously
- **🛡️ Risk Management**: Comprehensive stop loss and take profit controls
- **📱 Real-Time Monitoring**: Continuous position and market monitoring
- **🐳 Docker Ready**: Full containerization with Portainer support
- **📝 Comprehensive Logging**: Detailed logs for all trading activities
- **🔒 Secure**: API keys stored securely, never in repository

## ⚠️ Disclaimer

This software is for educational purposes only. Trading involves significant risk and you can lose money. Always:
- Test with paper trading first
- Monitor your bot closely
- Never risk more than you can afford to lose
