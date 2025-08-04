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
Replace the placeholder API keys in `docker-compose.yml`:
```yaml
environment:
  - ALPACA_LIVE_API_KEY=your_live_api_key
  - ALPACA_LIVE_API_SECRET=your_live_secret_key
  - STOCK_SYMBOL=NVDA           # Stock for market hours
  - CRYPTO_SYMBOL=LTC/USD       # Crypto for after hours  
  - STOCK_QUANTITY=1            # Number of stock shares
  - CRYPTO_QUANTITY=0.1         # Amount of crypto to trade
```

### 3. Deploy via Portainer
1. Log into Portainer
2. Go to **Stacks → Add Stack**
3. Name: `simple-swinger`
4. Paste the `docker-compose.yml` content
5. Deploy the stack

## 📊 Trading Strategy

- **Timeframe**: Daily bars for stocks, hourly for crypto
- **Indicators**: 10-day SMA vs 20-day SMA crossover
- **Market Hours**: NVDA stock (9:30 AM - 4:00 PM ET, Mon-Fri)
- **After Hours**: LTC/USD crypto (24/7 when market closed)
- **Execution**: Runs every 10 minutes
- **Order Type**: Market orders

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
