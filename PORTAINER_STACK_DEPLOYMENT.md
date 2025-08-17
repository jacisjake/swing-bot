# Portainer Stack Deployment Guide

## 🚀 Deploy Trading Bot via Portainer

### Step 1: Create New Stack
1. **Login to Portainer** → **Stacks** → **Add Stack**
2. **Name**: `simple-swinger-bot`
3. **Build method**: Select **"Repository"**

### Step 2: Repository Configuration
- **Repository URL**: `https://github.com/jacisjake/swing-bot.git`
- **Reference**: `refs/heads/main`
- **Compose path**: `docker-compose.yml`

### Step 3: Environment Variables (CRITICAL!)

**In Portainer, set these environment variables:**

#### 🔑 **Required API Keys** (NEVER leave empty!)
```
ALPACA_LIVE_API_KEY=AKXXXXXXXXXXXXXXXX
ALPACA_LIVE_API_SECRET=your_secret_key_here
```

#### 📊 **Trading Configuration** (Optional - has defaults)
```
MAX_STOCK_SYMBOLS=10
MAX_CRYPTO_SYMBOLS=5
MAX_POSITION_PERCENT=0.20
MAX_CASH_PER_TRADE=0.10
SCREENER_UPDATE_HOURS=1
```

#### 🛡️ **Risk Management** (Optional - has defaults)
```
STOP_LOSS_PERCENT=0.005
TAKE_PROFIT_1=0.005
TAKE_PROFIT_2=0.010
TAKE_PROFIT_3=0.015
```

#### 🔍 **Screener Criteria** (Optional - has defaults)
```
MIN_PRICE=5.0
MAX_PRICE=500.0
MIN_DAILY_VOLUME=1000000
```

### Step 4: Deploy Stack
1. **Click "Deploy the stack"**
2. **Wait for build to complete** (first time takes 2-3 minutes)
3. **Check container logs** for successful startup

### Step 5: Verify Deployment

Look for these messages in the logs:
```
✅ API keys loaded: API_KEY=AKRG978Z..., SECRET_KEY=******
✅ Account access validated. Portfolio: $XXXXX.XX
✅ Stock data access validated
✅ Crypto data access validated
🔄 Running initial screener update...
📊 Selected X stocks: ['AAPL', 'MSFT', ...]
📊 Selected X cryptos: ['BTC/USD', 'ETH/USD', ...]
```

### Step 6: Monitor Operation

#### Container Status
- **Containers** → **alpaca-swing-bot** → **Logs**
- Should see trading cycles every 5 minutes
- Screener updates every hour

#### Key Log Messages to Watch For
- `🔄 Running multi-symbol trading cycle`
- `📊 [SYMBOL] Position found - Entry: $X.XX, Current: $Y.YY, P&L: Z.Z%`
- `🎯 [SYMBOL] Scale-out trigger: X.XX% >= Y.YY%`
- `💰 [SYMBOL] ✅ Scaled out X units at Y.YY% profit`

## 🚨 Troubleshooting

### Error: "env file .env not found"
- **Solution**: Use environment variables in Portainer (this guide)
- **Don't** try to upload .env files to Portainer

### Error: "forbidden" API messages
- **Cause**: Invalid or missing API keys
- **Solution**: Check ALPACA_LIVE_API_KEY and ALPACA_LIVE_API_SECRET in stack environment

### Container won't start
- **Check**: Stack environment variables are set
- **Verify**: Repository URL is correct
- **Try**: Force recreation of stack

### No trading activity
- **Check**: Market hours for stocks (9:30 AM - 4:00 PM ET)
- **Verify**: Crypto trading runs 24/7
- **Look for**: "No entry signal" messages (normal)

## 📈 Stack Updates

### To update the bot with latest code:
1. **Stacks** → **simple-swinger-bot** → **Editor**
2. **Click "Pull and redeploy"**
3. **Monitor logs** for successful restart

### Environment Variable Changes:
1. **Stacks** → **simple-swinger-bot** → **Editor**
2. **Modify environment variables**
3. **Update the stack**

## 🔒 Security Notes

- **Never commit** API keys to git
- **Use Portainer environment variables** for all secrets
- **Monitor logs** for any credential exposure
- **Rotate API keys** regularly

## 📊 Expected Performance

- **Stocks**: Trade during market hours only
- **Crypto**: 24/7 trading
- **Profit Taking**: Automatic at configured levels
- **Risk Management**: Stop losses and position limits active