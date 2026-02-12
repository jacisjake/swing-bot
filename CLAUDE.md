# Project Context for Claude

## Deployment Environment

- **Remote server**: `jacisjake@ut.gitsum.rest`
- **Web server**: Caddy (reverse proxy)
- **Container runtime**: Podman (not Docker)
- **Deploy command**: `cd deploy && ./deploy-remote.sh jacisjake@ut.gitsum.rest --build`
- **Bot runs on port**: 8080 (internal)
- **Public URL**: https://ut.gitsum.rest (via Caddy reverse proxy)

## Caddy Configuration

To add a new site, edit `/etc/caddy/Caddyfile` on the server and reload:
```
sudo systemctl reload caddy
```

## Key Directories on Server

- `/opt/swing-trader/` - Application files
- `/opt/swing-trader/.env` - Environment variables (Alpaca keys)
- Container volumes for state/logs

## Trading Context

- **Broker**: Alpaca (live trading enabled)
- **Starting capital**: $400
- **Goal**: $4,000
- **Strategy**: Ross Cameron-style momentum pullback on low-float stocks
- **Timeframe**: 5-min bars
- **Target stocks**: $1-$10 price (prefer $2+), <20M float, 10%+ daily change, 5x+ relative volume
- **Trading window**: 7:00-10:00 AM ET (pre-market scanning from 6:00 AM)
- **Max trades/day**: 1 (cash account approach)
- **Position sizing**: Up to 90% of buying power, risk-constrained (2% max risk)
- **Scanner**: Alpaca screener API for top gainers, enriched with relative volume + float data
- **Float data**: Financial Modeling Prep (FMP) free API, yfinance fallback
- **Entry**: First pullback after surge, MACD must be positive above zero line
- **Exit**: MACD below zero, declining histogram, time-based (10:05 AM), or safety net (3:55 PM)
- **Market hours**: Active trading 7:00-10:00 AM ET, safety net close at 3:55 PM ET
