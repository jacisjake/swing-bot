# Local Testing Workflow for Live Trading Bot

## ⚠️ Important: Live Trading with Limited Funds

This bot uses **live trading** with limited account funds as the risk management strategy instead of paper trading.

## Pre-Testing Checklist

1. **Account Setup**
   - ✅ Limited fund account ($X,XXX max exposure)
   - ✅ Live API keys configured
   - ✅ Conservative position sizing (2% per asset for testing)

2. **Risk Management**
   - ✅ NVDA: 1 share max position
   - ✅ LTC: 2% portfolio max (vs 10% production)
   - ✅ Tight stop losses (0.5%)
   - ✅ Quick take profits (0.5%, 1.0%, 1.5%)

## Local Testing Steps

### 1. Quick Account Check
```bash
# Check current account status
Ctrl+Shift+P → "Tasks: Run Task" → "Check Account Balance & Positions"
```

### 2. Validate Bot Logic (No Trading)
```bash
# Dry run - validates logic without placing orders
Ctrl+Shift+P → "Tasks: Run Task" → "Dry Run - Validate Bot Logic Only"
```

### 3. Build Docker Image
```bash
# Build the trading bot image
Ctrl+Shift+P → "Tasks: Run Task" → "Build Trading Bot Docker Image"
```

### 4. Run Test with Ultra-Conservative Settings
```bash
# Run with 2% portfolio limit (vs 10% production)
Ctrl+Shift+P → "Tasks: Run Task" → "Test Live Trading Bot (Limited Funds)"
```

### 5. Monitor Test Run
```bash
# Watch logs in real-time
Ctrl+Shift+P → "Tasks: Run Task" → "View Trading Bot Logs"
```

### 6. Stop Testing
```bash
# Stop the test container
Ctrl+Shift+P → "Tasks: Run Task" → "Stop Test Trading Bot"
```

## What to Look For During Testing

### ✅ Successful Startup Indicators
- API keys loaded successfully
- Account access validated (shows portfolio value)
- Stock/Crypto data access confirmed
- Existing positions detected
- Market hours check working

### ✅ Trading Logic Working
- LTC: EMA crossover detection
- NVDA: Market hours respected (closed on weekends)
- Position sizing calculations correct
- Stop loss/take profit levels set properly

### ⚠️ Warning Signs to Stop Testing
- Unexpected large position sizes
- API errors persisting
- Orders failing repeatedly
- Position limits not being respected

## Production Deployment Workflow

1. **Local Test Passes** ✅
2. **Commit to Git**
   ```bash
   git add .
   git commit -m "Trading bot ready for production"
   git push origin main
   ```
3. **Deploy via Portainer**
   - Update stack from Git
   - Verify environment variables
   - Start production stack
   - Monitor initial cycles

## Emergency Procedures

### Stop All Trading Immediately
```bash
# Local testing
docker-compose -f docker-compose.test.yml down

# Production (if needed)
# In Portainer: Stop stack or individual container
```

### Check Current Positions
```bash
Ctrl+Shift+P → "Tasks: Run Task" → "Check Account Balance & Positions"
```

## Configuration Differences

| Setting | Testing | Production |
|---------|---------|------------|
| Portfolio Limit | 2% | 10% |
| Restart Policy | no | unless-stopped |
| Container Name | alpaca-swing-bot-test | alpaca-swing-bot |
| Monitoring | Manual | Automated |

## Ready to Test?

Run this command to start local testing:
```bash
Ctrl+Shift+P → "Tasks: Run Task" → "Test Live Trading Bot (Limited Funds)"
```

The bot will use live trading with ultra-conservative settings to minimize risk while validating the logic.
