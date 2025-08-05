# Portainer Deployment Instructions

## 🚨 CRITICAL: API Keys Setup

The bot is failing because API keys are not properly configured. You have two options:

### Option 1: Portainer Environment Variables (RECOMMENDED)

1. In Portainer, when creating/updating the stack:
2. Go to **Environment Variables** section
3. Add these variables:

```
ALPACA_LIVE_API_KEY=AKRG978ZNISA817UNE1C
ALPACA_LIVE_API_SECRET=1mCvTHCR95ZJAcmvElQbGUf1umowwbm5dnEoyfB0
```

### Option 2: Direct Replacement in docker-compose.yml

Replace the placeholder values in docker-compose.yml:

```yaml
environment:
  - ALPACA_LIVE_API_KEY=AKRG978ZNISA817UNE1C
  - ALPACA_LIVE_API_SECRET=1mCvTHCR95ZJAcmvElQbGUf1umowwbm5dnEoyfB0
```

## Current Error Diagnosis

The error `{"message": "forbidden."}` indicates:
- ❌ API keys are not being loaded (showing as `your_live_api_key_here`)
- ❌ Alpaca API rejects requests with invalid credentials

## Verification Steps

After deployment with real API keys, the logs should show:
```
✅ API keys loaded: API_KEY=AKRG978Z..., SECRET_KEY=******
✅ Account access validated. Portfolio: $XXXXX.XX
```

Instead of:
```
❌ API keys not found in environment variables!
❌ API validation failed: {"message": "forbidden."}
```

## Next Steps

1. **Set real API keys** using one of the methods above
2. **Redeploy the stack** in Portainer
3. **Check logs** for successful API validation
4. **Verify trading operation** begins properly

## Security Note

- **Never commit real API keys** to git repositories
- **Use Portainer environment variables** for production deployment
- **Keep API keys secure** and rotate them regularly
