# 🚀 VPS Deployment Instructions - Simple Swinger

## Prerequisites Setup (You'll need to do these)

### 1. Git Repository Setup
```bash
# Navigate to the simple_swinger folder
cd "d:\projects\Aplaca Projects\simple_swinger"

# Initialize git
git init
git add .
git commit -m "Initial commit - Simple Swinger Bot"

# Create GitHub repository and push
git remote add origin https://github.com/yourusername/simple-swinger.git
git branch -M main
git push -u origin main
```

### 2. VPS Preparation
```bash
# SSH into your VPS
ssh user@your-vps-ip

# Clone your repository
git clone https://github.com/yourusername/simple-swinger.git
cd simple-swinger
```

### 3. Portainer Configuration

**Stack Name:** `simple-swinger`

**Docker Compose Content:**
```yaml
version: "3.8"
services:
  alpaca-bot:
    build: .
    container_name: alpaca-swing-bot
    restart: unless-stopped
    environment:
      - ALPACA_LIVE_API_KEY=AKRG978ZNISA817UNE1C
      - ALPACA_LIVE_API_SECRET=1mCvTHCR95ZJAcmvElQbGUf1umowwbm5dnEoyfB0
      - TRADING_SYMBOL=NVDA
      - TRADING_QUANTITY=1
    volumes:
      - ./logs:/app/logs
      - /etc/localtime:/etc/localtime:ro
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### 4. Deployment Steps in Portainer

1. **Access Portainer**: Navigate to `http://your-vps-ip:9000`
2. **Go to Stacks**: Click "Stacks" in left sidebar
3. **Add Stack**: Click "Add stack" button
4. **Configuration**:
   - **Name**: `simple-swinger`
   - **Build method**: Select "Repository" 
   - **Repository URL**: `https://github.com/yourusername/simple-swinger`
   - **Compose path**: `docker-compose.yml`
5. **Deploy**: Click "Deploy the stack"

### 5. Monitoring Commands

```bash
# View live logs
docker logs -f alpaca-swing-bot

# Check container status
docker ps | grep alpaca

# View log files
tail -f logs/trading.log

# Stop bot (emergency)
docker stop alpaca-swing-bot

# Restart bot
docker restart alpaca-swing-bot
```

## 📊 Expected Log Output

When working correctly, you should see:
```
2025-08-04 09:30:01 - INFO - 🚀 Starting Alpaca Swing Trading Bot
2025-08-04 09:30:01 - INFO - Trading symbol: NVDA
2025-08-04 09:30:01 - INFO - ⚠️  LIVE TRADING MODE - Real money at risk!
2025-08-04 09:30:02 - INFO - 🚦 Running bot for NVDA
2025-08-04 09:30:03 - INFO - Fetched 60 bars for NVDA
2025-08-04 09:30:03 - INFO - Current price: $425.30, Short SMA: $423.15, Long SMA: $419.80
2025-08-04 09:30:03 - INFO - No current position in NVDA
2025-08-04 09:30:03 - INFO - 📈 Placed BUY order for NVDA - Order ID: abc123
2025-08-04 09:30:03 - INFO - Trading cycle completed successfully
```

## ⚠️ Safety Checklist

- [ ] Bot shows "LIVE TRADING MODE" in logs
- [ ] Correct symbol (NVDA) appears in logs
- [ ] API connection successful (no auth errors)
- [ ] Orders are being placed with correct quantity
- [ ] Container restarts automatically after reboot

## 🚨 Emergency Procedures

**To immediately stop trading:**
1. In Portainer: Containers → alpaca-swing-bot → Stop
2. Via SSH: `docker stop alpaca-swing-bot`
3. Manual intervention: Log into Alpaca web interface
