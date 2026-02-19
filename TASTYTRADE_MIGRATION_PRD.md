# Product Requirements Document: Alpaca to tastytrade Migration

## Executive Summary
Complete replacement of Alpaca broker integration with tastytrade in the swing-trader bot. No broker abstraction layer - direct tastytrade implementation only. TradingView remains the sole scanner.

## Business Context
- **Current State**: Bot uses Alpaca for trading and market data
- **Target State**: Full tastytrade integration with official SDK
- **Timeline**: Phased migration with verification gates
- **Risk Level**: High - production trading system
- **Rollback Strategy**: Git revert to pre-migration commit

## Technical Architecture

### Chosen SDK
- **Package**: `tastytrade-sdk` v1.2.0 (official by tastytrade)
- **Rationale**: Long-term maintenance stability over community alternatives
- **API Style**: Thin wrapper providing `api.get()`/`api.post()` + `market_data.subscribe()`

### Key API Endpoints
```
GET    /accounts/{acct}/balances/USD     # Account equity, buying power
GET    /accounts/{acct}/positions         # Open positions
POST   /accounts/{acct}/orders            # Place orders
DELETE /accounts/{acct}/orders/{id}       # Cancel orders
GET    /customers/{id}/orders             # Order history
GET    /instruments/equities/{symbol}     # Asset info
GET    /market-metrics                    # Current quotes
POST   /accounts/{acct}/orders/dry-run    # Order validation
```

### Streaming Architecture
- **Protocol**: DXLink WebSocket (6-step handshake: SETUP → AUTHORIZE → CHANNEL_REQUEST → FEED_SETUP → FEED_SUBSCRIPTION → KEEPALIVE)
- **Events**: Quote, Candle, Trade, TradeETH, Greeks, Profile, Summary
- **Format**: `AAPL{=5m}` for 5-minute bars (native support)
- **Historical Backfill**: DXLink Candle events support `fromTime` (Unix epoch) to backfill historical bars on connect — no REST API or yfinance needed
- **Threading**: SDK runs own thread, callbacks bridge to asyncio
- **Keepalive**: Every 30 seconds
- **Order Fills**: Poll `/accounts/{acct}/orders/live` every 5 seconds

### Key Differences from Alpaca

| Feature | Alpaca | tastytrade | Migration Impact |
|---------|--------|------------|------------------|
| Auth | API keys | Username/password or OAuth2 (15-min token, must refresh) | New env vars, token refresh logic |
| Trailing stops | Native | Not supported | Already code-side |
| Historical bars | REST API | DXLink `fromTime` backfill | Stream historical candles on connect |
| Screener | Native API | None | TradingView only |
| News feed | Native API | None | External sources |
| Order streaming | WebSocket | Polling required | 5-sec poll loop |
| Rate limits | 200/min | Unspecified (429 on abuse) | Adaptive throttle, minimize calls |
| Bar aggregation | 1-min → 5-min | Native 5-min | Simplify code |
| Fractional shares | Native qty | Notional Market orders (eligibility check required) | Check `is-fractional-quantity-eligible` |
| API key format | Dasherized JSON keys | Dasherized JSON keys (`time-in-force`, `order-type`) | SDK may abstract |
| Sandbox | Persistent | Resets daily (positions/balances wiped) | No multi-day sandbox tests |

## Migration Plan

### Phase 1: Foundation (6 tasks)
- [ ] **1.1** Update `config/settings.py` - Add tastytrade fields, remove Alpaca
- [ ] **1.2** Create `.env.example` - New credential template
- [ ] **1.3** Create `src/core/tastytrade_client.py` - REST wrapper with Alpaca-compatible interface
- [ ] **1.4** Create `src/core/tastytrade_ws.py` - DXLink streaming + order polling
- [ ] **1.5** Update `src/core/__init__.py` - Export new classes
- [ ] **1.6** Update `requirements.txt` - Swap dependencies

### Phase 2: Integration (9 tasks)
- [ ] **2.1** Update `src/core/order_executor.py` - Swap client type
- [ ] **2.2** Update `src/bot/config.py` - Inherit new settings
- [ ] **2.3** Simplify `src/bot/stream_handler.py` - Remove 1-min aggregation, news
- [ ] **2.4** Update `src/bot/monitor.py` - Schedule-based market clock
- [ ] **2.5** Update `src/bot/scheduler.py` - Remove Alpaca clock
- [ ] **2.6** Update `src/bot/screener.py` - Remove Alpaca fallback
- [ ] **2.7** Update `src/bot/processor.py` - Cash account settlement
- [ ] **2.8** Update `src/bot/main.py` - Wire new clients
- [ ] **2.9** Update `src/bot/api.py` - Dashboard adjustments

### Phase 3: Cleanup (2 tasks)
- [ ] **3.1** Delete `src/core/alpaca_client.py`
- [ ] **3.2** Delete `src/core/ws_client.py`

### Verification Gates (4 tasks)
- [ ] **V.1** Unit tests pass (`pytest tests/unit/`)
- [ ] **V.2** Sandbox smoke test (connection, streaming, dashboard)
- [ ] **V.3** End-to-end sandbox (signal → order → fill → exit)
- [ ] **V.4** Production deployment

## Implementation Details

### 1. TastytradeClient (REST Wrapper)
```python
class TastytradeClient:
    """Alpaca-compatible interface over tastytrade REST API"""
    
    # Account methods
    get_account() -> dict              # Equity, buying power, cash
    get_positions() -> list[dict]      # Open positions
    
    # Order methods
    submit_market_order()               # Market buy/sell
    submit_notional_order()             # Fractional/dollar-amount orders
    submit_limit_order()                # Limit orders
    submit_stop_limit_order()           # Stop-limit orders
    cancel_order()                      # Cancel by ID
    get_orders()                        # Order history
    
    # Market data
    get_bars() -> pd.DataFrame          # Via DXLink historical candle backfill
    get_latest_price() -> float         # Current quote
    check_fractional_eligible(symbol)   # Check is-fractional-quantity-eligible
    
    # Assets
    get_asset() -> dict                 # Symbol info
    is_market_open() -> bool            # Schedule-based
```

### 2. TastytradeWSClient (Streaming)
```python
class TastytradeWSClient:
    """DXLink streaming + order fill polling"""
    
    # Callbacks (Alpaca-compatible signatures)
    on_bar(callback)                   # 5-min candles
    on_quote(callback)                  # Bid/ask updates
    on_trade_update(callback)           # Order fills (via polling)
    
    # Subscriptions
    subscribe(bars=[], quotes=[])       # Symbol lists
    unsubscribe(bars=[], quotes=[])     # Remove symbols
    
    # Historical backfill
    backfill(symbols, from_time)        # DXLink fromTime candle request

    # Internal
    run_trade_loop()                    # 5-sec order polling
    _handle_candle()                    # Normalize to Alpaca format
    _handle_quote()                     # Normalize to Alpaca format
```

### 3. Environment Variables
```bash
# Login credentials (used by SDK's tasty.login())
TT_USERNAME=trader@email.com
TT_PASSWORD=your_password

# Account config
TT_ACCOUNT_NUMBER=your_account_number
TRADING_MODE=paper  # or live

# Sandbox URLs (production uses defaults)
# TT_API_URL=https://api.cert.tastyworks.com       # sandbox REST
# TT_STREAMER_URL=wss://streamer.cert.tastyworks.com # sandbox WS
```

### 4. Authentication Details
- **SDK login**: `tasty.login(login, password)` creates a session token
- **Token lifetime**: 15 minutes — must refresh before expiry
- **Token refresh**: Generates a new token (cannot extend existing)
- **Header**: `Authorization: Bearer [access_token]`
- **OAuth2 app**: Optional — only needed for third-party integrations, not personal use
- **Required headers**: `User-Agent: swing-trader/1.0`, `Content-Type: application/json`, `Accept: application/json`

### 5. Fractional / Notional Orders
- **Order type**: `Notional Market` — uses `value` (dollar amount) instead of `quantity`
- **Eligibility**: Must check `is-fractional-quantity-eligible` on the instrument before placing
- **Constraints**: Single-leg only, Day TIF only, equities only
- **Fallback**: If not eligible, round down to whole shares
```json
{
  "time-in-force": "Day",
  "order-type": "Notional Market",
  "value": 350,
  "value-effect": "Debit",
  "legs": [{
    "instrument-type": "Equity",
    "symbol": "AAPL",
    "action": "Buy to Open"
  }]
}
```

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| No order streaming | Delayed fill detection | Poll every 5 sec; only poll when orders are pending |
| No trailing stops | Strategy impact | Already implemented code-side |
| Token expiry (15 min) | Auth failure mid-session | Proactive refresh before expiry, retry on 401 |
| DXLink disconnect | Lost market data | Auto-reconnect with exponential backoff, re-subscribe |
| Unspecified rate limits | 429 throttling | Adaptive backoff, minimize REST calls, use streaming |
| No news API | Missing catalysts | External RSS/FMP scanner |
| Fractional ineligible | Can't size small positions | Check eligibility, fall back to whole shares |
| Sandbox daily reset | Can't test multi-day | Accept limitation; test day-level flows only |
| Dasherized API keys | Serialization bugs | Validate payload format in unit tests |

## Testing Strategy

### Unit Tests
- Position manager (broker-agnostic) 
- Order sizer (broker-agnostic)
- Stop manager (broker-agnostic)
- Portfolio limits (broker-agnostic)

### Integration Tests
- Login + token refresh cycle (verify 15-min refresh works)
- REST API calls (orders, positions, account)
- DXLink streaming (quotes, candles)
- DXLink historical backfill (fromTime candle request)
- Order fill detection (polling)
- Market hours detection (schedule)
- Fractional eligibility check + notional order placement
- DXLink reconnection after disconnect

### End-to-End Tests
1. **Entry Signal**: TradingView scan → validation → order
2. **Order Lifecycle**: Submit → fill detection → position tracking
3. **Exit Conditions**: MACD cross → stop adjustment → close
4. **Dashboard**: Account data, positions, P&L display

### Sandbox Limitations
- Positions/balances reset every 24 hours — run full E2E tests within a single session
- Quotes are 15-minute delayed — don't rely on price accuracy for signal testing
- Use sandbox for connectivity/flow verification, not strategy validation

## Rollout Plan

### Pre-Production
1. Create sandbox account at tastytrade developer portal
2. Get sandbox credentials (username/password) for testing
3. Note sandbox URLs: `api.cert.tastyworks.com` / `streamer.cert.tastyworks.com`
4. Branch: `feature/tastytrade-migration`
5. Tag pre-migration commit: `git tag pre-tastytrade-migration`

### Production Cutover
1. Market close Friday afternoon
2. Deploy new code: `./deploy-remote.sh`
3. Update server `.env` with production credentials
4. Smoke test over weekend
5. Live trading Monday 7:00 AM ET

### Rollback Procedure
```bash
# If issues detected (uses tag set in pre-production step)
git revert --no-commit pre-tastytrade-migration..HEAD
git commit -m "Rollback: tastytrade migration"
./deploy-remote.sh --build
# Restore Alpaca credentials in .env
```

## Success Metrics
- [ ] All unit tests pass
- [ ] Sandbox trading executes without errors
- [ ] Order fill detection < 3 seconds
- [ ] 5-min bar streaming stable
- [ ] Dashboard displays accurate data
- [ ] First production trade successful
- [ ] 5 consecutive trading days without issues

## Dependencies
- Sandbox account at tastytrade developer portal
- Production tastytrade login credentials for go-live
- `tastytrade-sdk` v1.2.0 (`pip install tastytrade-sdk`)

## Notes for Future Sessions
- All tasks numbered for easy reference (1.1, 2.3, etc.)
- Each task is atomic and verifiable
- Phase gates prevent premature progression
- Rollback plan documented and tested
- Migration preserves existing business logic