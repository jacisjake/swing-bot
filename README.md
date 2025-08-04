# 🚀 Simple Swinger - Multi-Asset Alpaca Trading Bot

A minimal, production-ready swing trading bot for Alpaca Markets using Docker and Portainer. Supports both stocks during market hours and cryptocurrency for 24/7 trading.

⚠️ **LIVE TRADING WARNING**: This bot trades with real money. Test thoroughly and monitor all trades.

## 🌟 Features

- **Multi-Asset Trading**: Automatically switches between stocks (market hours) and crypto (after hours)
- **Market Hours Detection**: Intelligent switching based on US stock market hours
- **24/7 Trading**: Never stops - trades stocks during market hours, crypto after hours
- **Docker Deployment**: Easy deployment with Portainer
- **Live Trading**: Real money trading with Alpaca Markets

## 📁 Project Structure

```
simple_swinger/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── swing_trader.py
```

## 🔧 Setup

### 1. Prerequisites
- Docker installed on your VPS
- Portainer running on your server
- Alpaca Markets account with API keys
- Crypto trading enabled on your Alpaca account

### 2. Configuration
Replace the API keys in `docker-compose.yml`:
```yaml
environment:
  # API Keys - REPLACE WITH YOUR ACTUAL KEYS
  - ALPACA_LIVE_API_KEY=your_live_api_key_here
  - ALPACA_LIVE_API_SECRET=your_live_secret_key_here
  # Trading Settings
  - STOCK_SYMBOL=NVDA                # Stock for market hours
  - CRYPTO_SYMBOL=LTC/USD            # Crypto for 24/7 trading
  - STOCK_QUANTITY=1                 # NVDA shares per trade
  - MAX_PORTFOLIO_PERCENT=0.10       # 10% max per asset
  - STOP_LOSS_PERCENT=0.005          # 0.5% stop loss
  - TAKE_PROFIT_1=0.005              # 0.5% first take profit
  - TAKE_PROFIT_2=0.010              # 1.0% second take profit
  - TAKE_PROFIT_3=0.015              # 1.5% third take profit
```

### 3. Deploy via Portainer
1. Log into Portainer
2. Go to **Stacks → Add Stack**
3. Name: `simple-swinger`
4. **Edit the API keys** in the compose content before deploying
5. Paste the updated `docker-compose.yml` content
6. Deploy the stack

## 📊 Dual Trading Strategy

### **LTC 5-Minute Scalping (24/7):**
- **Timeframe**: 5-minute candles
- **Indicators**: EMA 9/21 crossover + green candle confirmation
- **Exits**: Scaled (33%/33%/34%) at 0.5%/1.0%/1.5%
- **Risk**: 0.5% stop loss, 10% portfolio limit

### **NVDA Daily Trading (Market Hours):**
- **Timeframe**: Daily bars
- **Indicators**: EMA 9/21 crossover
- **Risk Management**: 6% take profit / 3% stop loss
- **Schedule**: Every 5 minutes (2.5min offset from LTC)
- **Order Type**: Market orders

## 🛡️ Risk Management

### **Automatic Exit Conditions (Priority Order):**
1. **🚨 Take Profit**: 6% gain → Force exit
2. **🚨 Stop Loss**: 3% loss → Force exit  
3. **📊 SMA Crossover**: Trend reversal → Regular exit

### **Configuration:**
- `STOP_LOSS_PERCENT=0.03` (3% maximum loss per trade)
- `TAKE_PROFIT_PERCENT=0.06` (6% profit target)
- Risk management overrides SMA signals when triggered

### **Example for LTC/USD:**
- Entry: $119.28
- Take Profit: $126.64 (+6%)
- Stop Loss: $115.70 (-3%)

## 🕒 Trading Schedule

| Time Period | Asset | Symbol | Data Frequency |
|-------------|-------|--------|----------------|
| **Market Hours** (9:30 AM - 4:00 PM ET, Mon-Fri) | Stock | NVDA | Daily bars |
| **After Hours** (All other times) | Crypto | LTC/USD | Hourly bars |

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

- **Automated Trading**: Runs every 10 minutes
- **Error Handling**: Comprehensive logging and error management
- **Live Trading**: Uses Alpaca live API (not paper trading)
- **Docker Ready**: Full containerization with Portainer support
- **Volume Mounting**: Persistent logs on host system

## ⚠️ Disclaimer

This software is for educational purposes only. Trading involves significant risk and you can lose money. Always:
- Test with paper trading first
- Monitor your bot closely
- Never risk more than you can afford to lose
