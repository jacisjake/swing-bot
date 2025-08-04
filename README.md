# 🚀 Simple Swinger - Alpaca Trading Bot

A minimal, production-ready swing trading bot for Alpaca Markets using Docker and Portainer.

⚠️ **LIVE TRADING WARNING**: This bot trades with real money. Test thoroughly and monitor all trades.

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

### 2. Configuration
Replace the placeholder API keys in `docker-compose.yml`:
```yaml
environment:
  - ALPACA_LIVE_API_KEY=your_live_api_key
  - ALPACA_LIVE_API_SECRET=your_live_secret_key
```

### 3. Deploy via Portainer
1. Log into Portainer
2. Go to **Stacks → Add Stack**
3. Name: `simple-swinger`
4. Paste the `docker-compose.yml` content
5. Deploy the stack

## 📊 Trading Strategy

- **Timeframe**: Daily bars
- **Indicators**: 10-day SMA vs 20-day SMA crossover
- **Symbol**: NVDA (configurable)
- **Execution**: Runs every 10 minutes
- **Order Type**: Market orders

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
