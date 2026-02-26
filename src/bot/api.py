"""
Simple web API for bot monitoring dashboard.
"""

import secrets
from datetime import datetime
from typing import Optional
from urllib.parse import urlencode

import pandas as pd
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from config import settings


app = FastAPI(title="Momentum Day Trader", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global reference to bot (set by main.py)
_bot = None


def set_bot(bot):
    """Set the bot reference for API access."""
    global _bot
    _bot = bot


class PositionResponse(BaseModel):
    symbol: str
    side: str
    qty: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


class StatusResponse(BaseModel):
    running: bool
    mode: str
    market_open: bool
    equity: float
    buying_power: float
    total_pnl: float
    total_pnl_pct: float
    position_count: int
    last_sync: Optional[str]


@app.get("/")
async def dashboard():
    """Serve the dashboard HTML."""
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/api/status")
async def get_status() -> dict:
    """Get bot status."""
    if not _bot:
        return {"error": "Bot not initialized"}

    # Check auth first — if not authenticated, return minimal status
    if not _bot.client.is_authenticated:
        return {
            "running": _bot._running,
            "authenticated": False,
            "mode": _bot.config.trading_mode.value,
            "error": "Not authenticated. Complete OAuth setup via dashboard.",
        }

    try:
        account = _bot.client.get_account()
        equity = float(account.get("equity", 0))
        buying_power = float(account.get("buying_power", 0))

        # Fetch positions directly from broker for accurate P&L
        positions = _bot.client.get_positions()
        unrealized_pnl = sum(p["unrealized_pl"] for p in positions)
        total_cost = sum(p["cost_basis"] for p in positions)

        # Get stats from broker order history (no local ledger needed)
        stats = _bot.client.get_trade_stats()

        last_sync_dt = _bot.bot_state.get_job_timestamp("broker_sync")
        last_sync = last_sync_dt.isoformat() if last_sync_dt else None

        # Experiment tracking: use actual broker equity
        # $250 = 0% progress, $25000 = 100% progress
        starting_capital = 250.0
        goal = 25000.0
        total_pnl = equity - starting_capital  # Actual P&L from broker equity
        progress_pct = ((equity - starting_capital) / (goal - starting_capital)) * 100

        return {
            "running": _bot._running,
            "authenticated": True,
            "mode": _bot.config.trading_mode.value,
            "is_trading_day": _bot.scheduler.is_trading_day(),
            "market_open": _bot.scheduler.is_market_open(),
            "in_trading_window": _bot.scheduler.is_in_trading_window(),
            "in_premarket": _bot.scheduler.is_in_premarket(),
            "equity": equity,
            "buying_power": buying_power,
            # Experiment tracking - based on actual broker equity
            "starting_capital": starting_capital,
            "goal": goal,
            "realized_pnl": stats["total_realized_pnl"],
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "current_value": equity,  # Use actual broker equity
            "progress_pct": progress_pct,  # Can be negative if below $400
            "remaining": goal - equity,
            # Stats
            "total_trades": stats["total_trades"],
            "win_count": stats["win_count"],
            "loss_count": stats["loss_count"],
            "win_rate": stats["win_rate"],
            "position_count": len(positions),
            # Day trading
            "trades_today": _bot._daily_trades_today,
            "max_daily_trades": _bot.config.max_daily_trades,
            "scanner_hits": len(_bot._scanner_results),
            "full_day_trading": _bot.config.full_day_trading,
            "trading_window": "9:30-15:55 ET" if _bot.config.full_day_trading else f"{_bot.config.trading_window_start}-{_bot.config.trading_window_end} ET",
            "last_sync": last_sync,
            "timestamp": datetime.now().isoformat(),
            # WebSocket streaming status
            "ws_data_connected": _bot.ws_client.data_connected if hasattr(_bot, "ws_client") else False,
            "ws_trade_connected": _bot.ws_client.trade_connected if hasattr(_bot, "ws_client") else False,
            "ws_subscribed_symbols": sorted(list(_bot.ws_client._subscribed_bars)) if hasattr(_bot, "ws_client") else [],
            # Press release scanner
            "pr_catalyst_count": _bot.press_release_scanner.get_status()["positive_hits"] if hasattr(_bot, "press_release_scanner") else 0,
            "pr_catalyst_symbols": _bot.press_release_scanner.get_catalyst_symbols(positive_only=True)[:10] if hasattr(_bot, "press_release_scanner") else [],
            # Regime gate
            "regime_gate_enabled": _bot.config.enable_regime_gate,
            "regime_category": _bot.regime_detector.category if hasattr(_bot, "regime_detector") else None,
            "regime_label": _bot.regime_detector.label if hasattr(_bot, "regime_detector") else None,
            "regime_confidence": _bot.regime_detector.confidence if hasattr(_bot, "regime_detector") else 0,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/positions")
async def get_positions() -> list[dict]:
    """Get all positions directly from broker."""
    if not _bot:
        return []

    try:
        # Fetch directly from broker for accurate real-time data
        positions = _bot.client.get_positions()
        results = []
        for p in positions:
            symbol = p["symbol"]
            is_crypto = p["asset_class"] == "crypto"
            qty = p["qty"]

            # Prefer real-time streaming price from position manager
            # (broker REST close-price is often previous day's close)
            current_price = p["current_price"]
            live_pos = _bot.position_manager.get_position(symbol)
            if live_pos and live_pos.current_price > 0:
                current_price = live_pos.current_price

            cost_basis = qty * p["avg_entry_price"]
            market_value = qty * current_price

            if p["side"] == "long":
                unrealized_pl = market_value - cost_basis
            else:
                unrealized_pl = cost_basis - market_value
            unrealized_plpc = (unrealized_pl / cost_basis) if cost_basis > 0 else 0

            # Estimate fees for net P&L display
            fee_rate = 0.0025 if is_crypto else 0.0
            fees = market_value * fee_rate
            net_proceeds = market_value - fees
            net_pnl = net_proceeds - cost_basis

            results.append({
                "symbol": symbol,
                "side": p["side"],
                "qty": qty,
                "entry_price": p["avg_entry_price"],
                "current_price": current_price,
                "unrealized_pnl": unrealized_pl,
                "unrealized_pnl_pct": unrealized_plpc * 100,
                "cost_basis": cost_basis,
                "market_value": market_value,
                "is_crypto": is_crypto,
                "fees": fees,
                "net_proceeds": net_proceeds,
                "net_pnl": net_pnl,
                "strategy": live_pos.strategy if live_pos and live_pos.strategy else "",
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]


@app.get("/api/watchlists")
async def get_watchlists() -> dict:
    """Get scanner results as the watchlist (scanner-driven)."""
    if not _bot:
        return {"stocks": [], "crypto": []}

    # Use scanner results as the stock watchlist
    stocks = [
        {
            "symbol": c.symbol,
            "price": c.price,
            "change": c.change_pct,
            "relative_volume": c.relative_volume,
            "float_millions": c.float_shares / 1e6 if c.float_shares else None,
            "has_catalyst": c.has_catalyst,
            "news_headline": c.news_headline,
            "news_url": c.news_url,
            "news_count": c.news_count,
        }
        for c in _bot._scanner_results
    ]

    # Add any open positions not already in the list
    position_symbols = {c.symbol for c in _bot._scanner_results}
    for pos in _bot.position_manager.get_open_positions():
        if pos.symbol not in position_symbols:
            try:
                price = _bot.client.get_latest_price(pos.symbol)
                stocks.insert(0, {
                    "symbol": pos.symbol,
                    "price": price,
                    "change": 0,
                    "relative_volume": 0,
                    "float_millions": None,
                })
            except Exception:
                pass

    return {
        "stocks": stocks,
        "crypto": [],  # Crypto disabled for day trading
    }


@app.get("/api/jobs")
async def get_jobs() -> list[dict]:
    """Get scheduled jobs."""
    if not _bot:
        return []

    return _bot.scheduler.get_jobs()


@app.get("/api/signals")
async def get_signals() -> dict:
    """Get signal history and stats."""
    if not _bot:
        return {}

    state = _bot.bot_state.get_state_summary()
    return {
        "active_signals": state.get("active_signals", 0),
        "metrics": state.get("metrics", {}),
        "job_timestamps": state.get("job_timestamps", {}),
    }


@app.get("/api/scanner")
async def get_scanner_results() -> list[dict]:
    """Get latest momentum scanner results."""
    if not _bot:
        return []

    try:
        return [
            {
                "symbol": c.symbol,
                "price": c.price,
                "change_pct": c.change_pct,
                "volume": c.volume,
                "relative_volume": c.relative_volume,
                "float_millions": c.float_shares / 1e6 if c.float_shares else None,
                "high_of_day": c.high_of_day,
                "has_catalyst": c.has_catalyst,
                "news_headline": c.news_headline,
                "news_url": c.news_url,
                "news_count": c.news_count,
                "news_source": c.news_source,
                "passes_all": c.passes_all_filters,
                "filter_failures": c.filter_failures,
            }
            for c in _bot._scanner_results
        ]
    except Exception as e:
        return [{"error": str(e)}]


@app.get("/api/ws/status")
async def get_ws_status() -> dict:
    """Get detailed WebSocket connection status."""
    if not _bot:
        return {"error": "Bot not initialized"}

    try:
        result = {"connected": False}

        if hasattr(_bot, "ws_client"):
            ws_status = _bot.ws_client.get_status()
            result = {
                "data_connected": ws_status.get("data_connected", False),
                "trade_connected": ws_status.get("trade_connected", False),
                "subscribed_bars": sorted(list(_bot.ws_client._subscribed_bars)),
                "subscribed_quotes": sorted(list(_bot.ws_client._subscribed_quotes)),
            }

        if hasattr(_bot, "stream_handler"):
            stream_status = _bot.stream_handler.get_status()
            result["stream_handler"] = stream_status

        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/news/{symbol}")
async def get_symbol_news(symbol: str, limit: int = 10) -> list[dict]:
    """Get recent news for a symbol."""
    if not _bot:
        return []

    try:
        return _bot.client.get_news(symbol=symbol.upper(), limit=limit, hours_back=24)
    except Exception as e:
        return [{"error": str(e)}]


@app.get("/api/press-releases")
async def get_press_releases() -> dict:
    """Get press release scanner results and status."""
    if not _bot:
        return {"error": "Bot not initialized"}

    try:
        status = _bot.press_release_scanner.get_status()
        hits = [h.to_dict() for h in _bot.press_release_scanner.hits[:50]]
        return {
            "status": status,
            "hits": hits,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/trades")
async def get_trades(limit: int = 50) -> list[dict]:
    """Get trade history from broker closed orders."""
    if not _bot:
        return []

    try:
        # Use the client wrapper method
        orders = _bot.client.get_orders(status="closed")

        # Filter and limit filled orders
        trades = []
        for order in orders[:limit]:
            if order.get("filled_qty", 0) > 0:
                trades.append({
                    "id": order.get("id"),
                    "symbol": order.get("symbol"),
                    "side": order.get("side"),
                    "qty": order.get("filled_qty"),
                    "price": order.get("filled_avg_price"),
                    "total": (order.get("filled_qty", 0) or 0) * (order.get("filled_avg_price", 0) or 0),
                    "status": order.get("status"),
                    "filled_at": order.get("created_at"),
                })

        return trades
    except Exception as e:
        return [{"error": str(e)}]


@app.get("/api/trades/ledger")
async def get_trade_ledger(limit: int = 50) -> dict:
    """Get trade history with P&L from broker order history."""
    if not _bot:
        return {"error": "Bot not initialized"}

    try:
        # Get stats directly from broker order history
        stats = _bot.client.get_trade_stats()

        # Get current equity for experiment progress
        account = _bot.client.get_account()
        equity = float(account.get("equity", 0))
        starting_capital = 250.0
        goal = 25000.0

        return {
            "trades": stats.get("trades", []),
            "stats": {
                "total_trades": stats["total_trades"],
                "total_realized_pnl": stats["total_realized_pnl"],
                "win_count": stats["win_count"],
                "loss_count": stats["loss_count"],
                "win_rate": stats["win_rate"],
                "avg_win": stats["avg_win"],
                "avg_loss": stats["avg_loss"],
            },
            "experiment": {
                "starting_capital": starting_capital,
                "goal": goal,
                "current_value": equity,
                "total_pnl": equity - starting_capital,
                "progress_pct": ((equity - starting_capital) / (goal - starting_capital)) * 100,
            },
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/sparklines")
async def get_sparklines() -> dict:
    """Get mini price history for sparklines (last 20 closes for each watchlist symbol)."""
    if not _bot:
        return {"error": "Bot not initialized"}

    try:
        # Use scanner results as stock symbols
        stock_symbols = [c.symbol for c in _bot._scanner_results]

        # Also include open positions
        for pos in _bot.position_manager.get_open_positions():
            if pos.symbol not in stock_symbols:
                stock_symbols.append(pos.symbol)

        result = {"stocks": {}, "crypto": {}}

        # Fetch 5-min bars for stocks
        for symbol in stock_symbols:
            try:
                bars = _bot.client.get_bars(symbol, timeframe="5Min", limit=20)
                if bars is not None and not bars.empty:
                    closes = [float(c) for c in bars["close"].tolist()]
                    result["stocks"][symbol] = closes
            except Exception:
                pass

        return result
    except Exception as e:
        return {"error": str(e)}


class WebhookPayload(BaseModel):
    """TradingView webhook payload."""
    symbol: str
    action: str = "add_watchlist"  # add_watchlist, remove_watchlist
    source: str = "tradingview"


@app.post("/api/webhook")
async def tradingview_webhook(payload: WebhookPayload) -> dict:
    """
    Receive TradingView webhook alerts.

    Note: Watchlist is now scanner-driven. Webhooks are logged but
    the scanner controls what gets traded.
    """
    if not _bot:
        return {"error": "Bot not initialized", "status": "error"}

    symbol = payload.symbol.upper().strip()
    action = payload.action.lower()
    source = payload.source

    from loguru import logger
    logger.info(f"[Webhook] {source}: {action} {symbol}")

    return {
        "status": "ok",
        "message": f"Received webhook for {symbol} (scanner-driven mode, watchlist not modified)",
        "symbol": symbol,
    }


@app.post("/api/scan/trigger")
async def trigger_scan() -> dict:
    """Manually trigger a momentum scan."""
    if not _bot:
        return {"error": "Bot not initialized"}

    from loguru import logger
    logger.info("[API] Manual scan triggered")
    await _bot._run_momentum_scan()
    results = [c.to_dict() for c in _bot._scanner_results] if _bot._scanner_results else []
    return {"status": "ok", "candidates": len(results), "results": results}


@app.post("/api/scan/press-releases")
async def trigger_press_release_scan() -> dict:
    """Manually trigger a press release scan."""
    if not _bot:
        return {"error": "Bot not initialized"}

    from loguru import logger
    logger.info("[API] Manual press release scan triggered")
    await _bot._run_press_release_scan()
    status = _bot.press_release_scanner.get_status()
    return {"status": "ok", **status}


@app.get("/api/regime")
async def get_regime() -> dict:
    """Get current HMM regime status."""
    if not _bot:
        return {"error": "Bot not initialized"}

    try:
        status = _bot.regime_detector.get_status()
        status["enabled"] = _bot.config.enable_regime_gate
        return status
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/settings/regime-gate")
async def toggle_regime_gate(enabled: bool = True) -> dict:
    """Toggle HMM regime gate on/off."""
    if not _bot:
        return {"error": "Bot not initialized"}

    _bot.config.enable_regime_gate = enabled

    # Train on first enable if not yet trained
    if enabled and not _bot.regime_detector.trained:
        _bot.regime_detector.refresh()

    status = _bot.regime_detector.get_status()
    from loguru import logger
    logger.info(f"[API] Regime gate {'enabled' if enabled else 'disabled'}")
    return {
        "status": "ok",
        "regime_gate_enabled": enabled,
        "regime": status,
    }


@app.post("/api/regime/refresh")
async def refresh_regime() -> dict:
    """Manually refresh regime detection."""
    if not _bot:
        return {"error": "Bot not initialized"}

    success = _bot.regime_detector.refresh()
    status = _bot.regime_detector.get_status()
    return {
        "status": "ok" if success else "error",
        "regime": status,
    }


@app.post("/api/settings/full-day-trading")
async def toggle_full_day_trading(enabled: bool = True) -> dict:
    """Toggle between early window (7-10 AM) and full day (9:30 AM - 3:55 PM) trading."""
    if not _bot:
        return {"error": "Bot not initialized"}

    _bot.scheduler.set_full_day_trading(enabled)

    # Update monitor's time-based exit window to match
    from datetime import time as dt_time
    if enabled:
        _bot.monitor._window_end = dt_time(15, 55)
    else:
        parts = _bot.config.trading_window_end.split(":")
        _bot.monitor._window_end = dt_time(int(parts[0]), int(parts[1]))

    window = "9:30 AM - 3:55 PM ET" if enabled else f"{_bot.config.trading_window_start}-{_bot.config.trading_window_end} ET"
    return {
        "status": "ok",
        "full_day_trading": enabled,
        "trading_window": window,
    }


@app.get("/api/asset/{symbol}")
async def get_asset_info(symbol: str) -> dict:
    """Get asset name and details from broker."""
    if not _bot:
        return {"error": "Bot not initialized"}

    try:
        asset = _bot.client.get_asset(symbol)
        return {
            "symbol": symbol,
            "name": asset.get("name", symbol),
            "exchange": asset.get("exchange", ""),
            "asset_class": asset.get("class", ""),
        }
    except Exception as e:
        # Fallback for crypto or unknown assets
        return {
            "symbol": symbol,
            "name": symbol,
            "exchange": "",
            "asset_class": "",
            "error": str(e),
        }


@app.get("/api/bars/{symbol}")
async def get_bars(symbol: str, timeframe: str = "5Min", limit: int = 100) -> dict:
    """Get OHLCV bars with MACD indicators for charting."""
    if not _bot:
        return {"error": "Bot not initialized"}

    try:
        from src.data.indicators import macd

        # Fetch bars from broker
        bars = _bot.client.get_bars(symbol, timeframe=timeframe, limit=limit)

        if bars is None or bars.empty:
            return {"error": f"No data for {symbol}"}

        # Calculate MACD
        close = bars["close"]
        macd_line, signal_line, histogram = macd(close, 8, 17, 9)

        # Format for TradingView Lightweight Charts
        candles = []
        for idx, row in bars.iterrows():
            candles.append({
                "time": int(idx.timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            })

        macd_data = []
        for i, (idx, _) in enumerate(bars.iterrows()):
            if not pd.isna(macd_line.iloc[i]):
                macd_data.append({
                    "time": int(idx.timestamp()),
                    "macd": float(macd_line.iloc[i]),
                    "signal": float(signal_line.iloc[i]),
                    "histogram": float(histogram.iloc[i]),
                })

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": candles,
            "macd": macd_data,
        }
    except Exception as e:
        return {"error": str(e)}


# =========================================================================
# OAuth2 Endpoints
# =========================================================================

# CSRF state tokens for OAuth flow (in-memory, short-lived)
_oauth_states: dict[str, float] = {}

TASTYTRADE_AUTH_URL = "https://my.tastytrade.com/auth.html"
TASTYTRADE_TOKEN_URL = (
    "https://api.cert.tastyworks.com/oauth/token"
    if settings.is_paper
    else "https://api.tastyworks.com/oauth/token"
)


def _get_oauth_redirect_uri(request: Request) -> str:
    """Build the OAuth redirect URI, respecting reverse proxy headers."""
    # Prefer explicit config (required for reverse proxy with path prefix)
    if settings.tt_oauth_redirect_uri:
        return settings.tt_oauth_redirect_uri
    # Fallback: build from request (works for direct access without proxy)
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.url.netloc)
    return f"{proto}://{host}/oauth/callback"


@app.get("/oauth/authorize")
async def oauth_authorize(request: Request):
    """Redirect user to tastytrade OAuth login page."""
    if not settings.tt_client_id:
        return {"error": "TT_CLIENT_ID not configured in .env"}

    redirect_uri = _get_oauth_redirect_uri(request)

    # Generate CSRF state token
    state = secrets.token_urlsafe(32)
    _oauth_states[state] = datetime.now().timestamp()

    params = {
        "response_type": "code",
        "client_id": settings.tt_client_id,
        "redirect_uri": redirect_uri,
        "scope": "read trade",
        "state": state,
    }
    auth_url = f"{TASTYTRADE_AUTH_URL}?{urlencode(params)}"
    return RedirectResponse(url=auth_url)


@app.get("/oauth/callback")
async def oauth_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    """Handle tastytrade OAuth callback — exchange code for tokens."""
    from loguru import logger

    if error:
        logger.error(f"OAuth error: {error}")
        return HTMLResponse(
            content=f"<h2>OAuth Error</h2><p>{error}</p>"
            '<p><a href="/">Back to dashboard</a></p>',
            status_code=400,
        )

    # Validate CSRF state
    if state not in _oauth_states:
        return HTMLResponse(
            content="<h2>Invalid state</h2><p>CSRF validation failed. Try again.</p>"
            '<p><a href="/">Back to dashboard</a></p>',
            status_code=400,
        )
    del _oauth_states[state]

    # Clean up expired states (older than 10 min)
    cutoff = datetime.now().timestamp() - 600
    expired = [k for k, v in _oauth_states.items() if v < cutoff]
    for k in expired:
        del _oauth_states[k]

    if not code:
        return HTMLResponse(
            content="<h2>No authorization code</h2>"
            '<p><a href="/">Back to dashboard</a></p>',
            status_code=400,
        )

    # Exchange authorization code for tokens
    redirect_uri = _get_oauth_redirect_uri(request)

    try:
        resp = requests.post(
            TASTYTRADE_TOKEN_URL,
            json={
                "grant_type": "authorization_code",
                "client_id": settings.tt_client_id,
                "client_secret": settings.tt_client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
            },
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        token_data = resp.json()
    except Exception as e:
        logger.error(f"OAuth token exchange failed: {e}")
        return HTMLResponse(
            content=f"<h2>Token exchange failed</h2><p>{e}</p>"
            '<p><a href="/">Back to dashboard</a></p>',
            status_code=500,
        )

    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        return HTMLResponse(
            content="<h2>No refresh token in response</h2>"
            '<p><a href="/">Back to dashboard</a></p>',
            status_code=500,
        )

    # Store the refresh token and authenticate the client
    if _bot:
        _bot.client.set_refresh_token(refresh_token)
        logger.info("OAuth flow complete — client authenticated")

    return HTMLResponse(
        content='<html><body style="background:#0d1117;color:#3fb950;font-family:sans-serif;'
        'display:flex;justify-content:center;align-items:center;height:100vh">'
        "<div><h2>Connected to tastytrade!</h2>"
        '<p style="color:#c9d1d9">Refresh token saved. Redirecting to dashboard...</p>'
        '</div><script>setTimeout(()=>window.location="/",2000)</script></body></html>'
    )


@app.get("/api/auth/status")
async def get_auth_status() -> dict:
    """Get authentication status."""
    authenticated = _bot.client.is_authenticated if _bot else False
    auth_mode = "oauth" if (settings.has_oauth or settings.tt_client_secret) else "legacy"
    has_client_id = bool(settings.tt_client_id)

    return {
        "authenticated": authenticated,
        "auth_mode": auth_mode,
        "has_client_id": has_client_id,
        "can_start_oauth": has_client_id and bool(settings.tt_client_secret),
    }


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Momentum Day Trader</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
            line-height: 1.5;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #58a6ff; margin-bottom: 20px; }
        h2 { color: #8b949e; font-size: 14px; text-transform: uppercase; margin: 20px 0 10px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .two-column-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        .two-column-grid h2 { margin-top: 0; }
        .watchlist-card { max-height: 400px; overflow-y: auto; }
        @media (max-width: 900px) { .two-column-grid { grid-template-columns: 1fr; } }
        .card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
        }
        .card-title { color: #8b949e; font-size: 12px; margin-bottom: 5px; }
        .card-value { font-size: 24px; font-weight: 600; }
        .card-subtitle { color: #8b949e; font-size: 12px; margin-top: 4px; }
        .progress-bar {
            height: 6px;
            background: #21262d;
            border-radius: 3px;
            margin-top: 8px;
            overflow: visible;
            position: relative;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #3fb950, #58a6ff);
            border-radius: 3px;
            transition: width 0.5s ease;
        }
        .progress-fill.negative {
            background: #f85149;
            position: absolute;
            right: 100%;
            border-radius: 3px 0 0 3px;
        }
        .progress-marker {
            position: absolute;
            top: -4px;
            width: 2px;
            height: 14px;
            background: #8b949e;
        }
        .positive { color: #3fb950; }
        .negative { color: #f85149; }
        .neutral { color: #8b949e; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #30363d; }
        th { color: #8b949e; font-weight: 500; font-size: 12px; text-transform: uppercase; }
        .status-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-dot.running { background: #3fb950; }
        .status-dot.stopped { background: #f85149; }
        .tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
        }
        .tag.live { background: #f8514922; color: #f85149; }
        .tag.paper { background: #3fb95022; color: #3fb950; }
        .watchlist-table {
            width: 100%;
            border-collapse: collapse;
        }
        .watchlist-table th {
            text-align: left;
            padding: 8px 12px;
            border-bottom: 1px solid #30363d;
            color: #8b949e;
            font-weight: 500;
            font-size: 12px;
        }
        .watchlist-table td {
            padding: 8px 12px;
            border-bottom: 1px solid #21262d;
            font-size: 13px;
        }
        .watchlist-table tr {
            cursor: pointer;
            transition: background 0.2s;
        }
        .watchlist-table tr:hover {
            background: #21262d;
        }
        .watchlist-table .symbol-cell {
            font-family: monospace;
            font-weight: 600;
        }
        .watchlist-table .price-cell {
            text-align: right;
        }
        .watchlist-table .change-cell {
            text-align: right;
            width: 80px;
        }
        .watchlist-table .sparkline-cell {
            width: 60px;
            text-align: center;
        }
        .sparkline {
            width: 50px;
            height: 16px;
            vertical-align: middle;
        }
        .sparkline polyline {
            fill: none;
            stroke-width: 1.5;
            stroke-linecap: round;
            stroke-linejoin: round;
        }
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .modal-overlay.active { display: flex; }
        .modal {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            width: 90%;
            max-width: 1000px;
            max-height: 90vh;
            overflow: hidden;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            border-bottom: 1px solid #30363d;
        }
        .modal-title { font-size: 18px; font-weight: 600; }
        .modal-close {
            background: none;
            border: none;
            color: #8b949e;
            font-size: 24px;
            cursor: pointer;
        }
        .modal-close:hover { color: #c9d1d9; }
        .modal-nav-button {
            background: #21262d;
            border: 1px solid #30363d;
            color: #c9d1d9;
            padding: 5px 10px;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .modal-nav-button:hover {
            background: #30363d;
        }
        #chart-container { height: 400px; }
        #macd-container { height: 150px; border-top: 1px solid #30363d; }
        .refresh-note {
            color: #8b949e;
            font-size: 12px;
            margin-top: 20px;
            text-align: center;
        }
        .toggle-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 6px;
        }
        .toggle-label {
            color: #8b949e;
            font-size: 12px;
        }
        .toggle-switch {
            position: relative;
            width: 36px;
            height: 20px;
            flex-shrink: 0;
        }
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            inset: 0;
            background: #30363d;
            border-radius: 20px;
            transition: 0.2s;
        }
        .toggle-slider:before {
            content: "";
            position: absolute;
            height: 14px;
            width: 14px;
            left: 3px;
            bottom: 3px;
            background: #8b949e;
            border-radius: 50%;
            transition: 0.2s;
        }
        .toggle-switch input:checked + .toggle-slider {
            background: #238636;
        }
        .toggle-switch input:checked + .toggle-slider:before {
            transform: translateX(16px);
            background: #fff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Momentum Day Trader <span id="ws-indicator" style="font-size:12px;vertical-align:middle;margin-left:10px;padding:3px 10px;border-radius:12px;background:#f8514922;color:#f85149">DXLink &#x25CF; Disconnected</span></h1>

        <div id="auth-banner" style="display:none;margin-bottom:20px">
            <div class="card" style="border-color:#d29922;text-align:center;padding:20px">
                <div style="font-size:18px;color:#d29922;margin-bottom:10px">Not Connected to tastytrade</div>
                <p style="color:#8b949e;margin-bottom:15px">Complete OAuth setup to start trading</p>
                <a href="oauth/authorize" style="display:inline-block;background:#238636;color:#fff;padding:10px 24px;border-radius:6px;text-decoration:none;font-weight:600">Connect tastytrade Account</a>
            </div>
        </div>

        <div class="grid" id="status-grid">
            <div class="card">
                <div class="card-title">Status</div>
                <div class="card-value" id="status">Loading...</div>
                <div class="card-subtitle" id="trading-window">--</div>
                <div class="toggle-row">
                    <label class="toggle-switch">
                        <input type="checkbox" id="full-day-toggle" onchange="toggleFullDay(this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                    <span class="toggle-label">Full day trading</span>
                </div>
            </div>
            <div class="card">
                <div class="card-title">Progress to $4,000</div>
                <div class="card-value" id="progress">--</div>
                <div class="progress-bar"><div class="progress-fill" id="progress-bar"></div></div>
            </div>
            <div class="card">
                <div class="card-title">Total P&L (Realized + Open)</div>
                <div class="card-value" id="total-pnl">--</div>
                <div class="card-subtitle" id="pnl-breakdown">--</div>
            </div>
            <div class="card">
                <div class="card-title">Win / Loss</div>
                <div class="card-value" id="win-loss">--</div>
                <div class="card-subtitle" id="win-loss-detail">--</div>
            </div>
            <div class="card">
                <div class="card-title">Today</div>
                <div class="card-value" id="trades-today">--</div>
                <div class="card-subtitle" id="scanner-info">--</div>
            </div>
            <div class="card">
                <div class="card-title">Market Regime</div>
                <div class="card-value" id="regime-status">--</div>
                <div class="card-subtitle" id="regime-detail">--</div>
                <div class="toggle-row">
                    <label class="toggle-switch">
                        <input type="checkbox" id="regime-gate-toggle" onchange="toggleRegimeGate(this.checked)">
                        <span class="toggle-slider"></span>
                    </label>
                    <span class="toggle-label">Regime gate</span>
                </div>
            </div>
        </div>

        <div class="two-column-grid">
            <div>
                <h2>Positions</h2>
                <div class="card">
                    <table>
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Qty</th>
                                <th>Entry</th>
                                <th>Current</th>
                                <th>P&L</th>
                            </tr>
                        </thead>
                        <tbody id="positions-table">
                            <tr><td colspan="5">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
            <div>
                <h2>Scanner Results</h2>
                <div class="card watchlist-card">
                    <table class="watchlist-table">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th></th>
                                <th class="price-cell">Price</th>
                                <th class="change-cell">Change</th>
                                <th>Catalyst</th>
                            </tr>
                        </thead>
                        <tbody id="stock-watchlist">
                            <tr><td colspan="5">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <h2>Press Release Catalysts <span id="pr-count" style="font-size:12px;color:#8b949e;font-weight:normal"></span></h2>
        <div class="card">
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Headline</th>
                        <th>Source</th>
                        <th>Sentiment</th>
                    </tr>
                </thead>
                <tbody id="pr-table">
                    <tr><td colspan="4" style="color:#8b949e">No press releases yet</td></tr>
                </tbody>
            </table>
        </div>

        <h2>Scheduled Jobs</h2>
        <div class="card">
            <table>
                <thead>
                    <tr>
                        <th>Job</th>
                        <th>Next Run</th>
                    </tr>
                </thead>
                <tbody id="jobs-table">
                    <tr><td colspan="2">Loading...</td></tr>
                </tbody>
            </table>
        </div>

        <p class="refresh-note">Auto-refreshes every 60 seconds</p>
    </div>

    <div class="modal-overlay" id="chart-modal">
        <div class="modal">
            <div class="modal-header">
                <button class="modal-nav-button" onclick="showPrevChart()">&lt; Prev</button>
                <span class="modal-title" id="chart-title">Symbol</span>
                <button class="modal-nav-button" onclick="showNextChart()">Next &gt;</button>
                <button class="modal-close" onclick="closeChart()">&times;</button>
            </div>
            <div id="chart-container"></div>
            <div id="macd-container"></div>
        </div>
    </div>

    <script>
        function formatCurrency(val) {
            return '$' + val.toFixed(2);
        }

        function formatPnl(val, pct) {
            const sign = val >= 0 ? '+' : '';
            const cls = val >= 0 ? 'positive' : 'negative';
            return `<span class="${cls}">${sign}${formatCurrency(val)} (${sign}${pct.toFixed(2)}%)</span>`;
        }

        // Generate SVG sparkline from price array
        function sparklineSvg(prices, color) {
            if (!prices || prices.length < 2) return '';
            const width = 50, height = 16, padding = 2;
            const min = Math.min(...prices);
            const max = Math.max(...prices);
            const range = max - min || 1;
            const points = prices.map((p, i) => {
                const x = padding + (i / (prices.length - 1)) * (width - 2 * padding);
                const y = height - padding - ((p - min) / range) * (height - 2 * padding);
                return `${x.toFixed(1)},${y.toFixed(1)}`;
            }).join(' ');
            return `<svg class="sparkline" viewBox="0 0 ${width} ${height}"><polyline points="${points}" stroke="${color}"/></svg>`;
        }

        let sparklineData = { stocks: {}, crypto: {} };

        async function fetchSparklines() {
            try {
                sparklineData = await fetch('api/sparklines').then(r => r.json());
            } catch (e) {
                console.error('Sparkline fetch error:', e);
            }
        }

        async function fetchData() {
            try {
                const [status, positions, watchlists, jobs, prData] = await Promise.all([
                    fetch('api/status').then(r => r.json()),
                    fetch('api/positions').then(r => r.json()),
                    fetch('api/watchlists').then(r => r.json()),
                    fetch('api/jobs').then(r => r.json()),
                    fetch('api/press-releases').then(r => r.json()).catch(() => ({hits: []})),
                ]);

                // Store watchlist data globally
                currentWatchlistData.stocks = watchlists.stocks || [];
                currentWatchlistData.crypto = watchlists.crypto || [];

                // Store positions globally for chart entry price markers
                currentPositionsData = positions;

                // Auth banner
                const authBanner = document.getElementById('auth-banner');
                if (status.authenticated === false && !status.running) {
                    authBanner.style.display = 'block';
                } else {
                    authBanner.style.display = 'none';
                }

                // Status
                const statusEl = document.getElementById('status');
                const dot = status.running ? 'running' : 'stopped';
                const mode = status.mode || 'unknown';
                statusEl.innerHTML = `<span class="status-dot ${dot}"></span>${status.running ? 'Running' : 'Stopped'} <span class="tag ${mode}">${mode.toUpperCase()}</span>`;

                // WebSocket connection indicator
                const wsIndicator = document.getElementById('ws-indicator');
                const wsDataOk = status.ws_data_connected || false;
                const wsTradeOk = status.ws_trade_connected || false;
                const wsSymCount = (status.ws_subscribed_symbols || []).length;
                if (wsDataOk && wsTradeOk) {
                    wsIndicator.style.background = '#3fb95022';
                    wsIndicator.style.color = '#3fb950';
                    wsIndicator.innerHTML = `DXLink &#x25CF; Connected (${wsSymCount} symbols)`;
                } else if (wsDataOk || wsTradeOk) {
                    wsIndicator.style.background = '#d2992222';
                    wsIndicator.style.color = '#d29922';
                    wsIndicator.innerHTML = `DXLink &#x25CF; Partial`;
                } else {
                    wsIndicator.style.background = '#f8514922';
                    wsIndicator.style.color = '#f85149';
                    wsIndicator.innerHTML = `DXLink &#x25CF; Disconnected`;
                }

                // Progress toward goal ($250 = 0%, $25000 = 100%)
                const currentVal = status.current_value || status.equity || 250;
                const goal = status.goal || 25000;
                const startingCapital = status.starting_capital || 250;
                const progressPct = status.progress_pct || 0;

                // Show current value with color based on P&L
                const pnlClass = currentVal >= startingCapital ? 'positive' : 'negative';
                const pnlSign = currentVal >= startingCapital ? '+' : '';
                const pnlAmount = currentVal - startingCapital;
                document.getElementById('progress').innerHTML =
                    `<span class="${pnlClass}">${formatCurrency(currentVal)}</span> ` +
                    `<span class="neutral" style="font-size:14px">/ ${formatCurrency(goal)}</span>` +
                    `<br><span class="${pnlClass}" style="font-size:14px">${pnlSign}${formatCurrency(pnlAmount)} from start</span>`;

                // Handle progress bar - negative shows red bar extending left
                const progressBar = document.getElementById('progress-bar');
                if (progressPct >= 0) {
                    progressBar.className = 'progress-fill';
                    progressBar.style.width = `${Math.min(100, progressPct)}%`;
                    progressBar.style.right = '';
                } else {
                    progressBar.className = 'progress-fill negative';
                    progressBar.style.width = `${Math.min(10, Math.abs(progressPct))}%`;
                }

                // Total P&L with breakdown
                const totalPnl = status.total_pnl || 0;
                const realizedPnl = status.realized_pnl || 0;
                const unrealizedPnl = status.unrealized_pnl || 0;
                const totalPnlPct = startingCapital > 0 ? (totalPnl / startingCapital * 100) : 0;
                document.getElementById('total-pnl').innerHTML = formatPnl(totalPnl, totalPnlPct);
                document.getElementById('pnl-breakdown').innerHTML = `Realized: ${formatPnl(realizedPnl, 0).replace(/\(.*\)/, '')} | Open: ${formatPnl(unrealizedPnl, 0).replace(/\(.*\)/, '')}`;

                // Win/Loss record
                const winCount = status.win_count || 0;
                const lossCount = status.loss_count || 0;
                const winRate = status.win_rate || 0;
                const totalTrades = status.total_trades || 0;
                const wlClass = winCount >= lossCount ? 'positive' : 'negative';
                document.getElementById('win-loss').innerHTML =
                    `<span class="${wlClass}">${winCount}W - ${lossCount}L</span>`;
                document.getElementById('win-loss-detail').innerHTML =
                    `${winRate.toFixed(0)}% win rate (${totalTrades} trades)`;

                // Trades today + scanner info
                const tradesToday = status.trades_today || 0;
                const maxDaily = status.max_daily_trades || 10;
                const scannerHits = status.scanner_hits || 0;
                document.getElementById('trades-today').innerHTML =
                    `${tradesToday}/${maxDaily} trades`;
                document.getElementById('scanner-info').innerHTML =
                    `${scannerHits} scanner hits`;

                // Trading window status
                const windowEl = document.getElementById('trading-window');
                const tradingWindow = status.trading_window || '--';
                const isTradingDay = status.is_trading_day !== false;
                if (!isTradingDay) {
                    windowEl.innerHTML = `<span style="color:#f85149">Holiday</span> ${tradingWindow}`;
                } else if (status.in_trading_window) {
                    windowEl.innerHTML = `<span class="positive">Active</span> ${tradingWindow}`;
                } else if (status.in_premarket) {
                    windowEl.innerHTML = `<span style="color:#d29922">Pre-market</span> ${tradingWindow}`;
                } else {
                    windowEl.innerHTML = `<span class="neutral">Closed</span> ${tradingWindow}`;
                }

                // Sync full day toggle with server state
                const fullDayToggle = document.getElementById('full-day-toggle');
                if (fullDayToggle && !fullDayToggle._userClicked) {
                    fullDayToggle.checked = !!status.full_day_trading;
                }

                // Regime gate status
                const regimeEl = document.getElementById('regime-status');
                const regimeDetail = document.getElementById('regime-detail');
                const regimeToggle = document.getElementById('regime-gate-toggle');
                if (regimeToggle && !regimeToggle._userClicked) {
                    regimeToggle.checked = !!status.regime_gate_enabled;
                }
                const regimeLabel = status.regime_label || '--';
                const regimeCat = status.regime_category || 'neutral';
                const regimeConf = status.regime_confidence || 0;
                const regimeColor = regimeCat === 'bullish' ? '#3fb950'
                    : regimeCat === 'bearish' ? '#f85149' : '#d29922';
                const regimeIcon = regimeCat === 'bullish' ? '&#x25B2;'
                    : regimeCat === 'bearish' ? '&#x25BC;' : '&#x25CF;';
                regimeEl.innerHTML = `<span style="color:${regimeColor}">${regimeIcon} ${regimeLabel}</span>`;
                const gateStatus = status.regime_gate_enabled ? 'Active' : 'Off';
                regimeDetail.innerHTML = `${(regimeConf * 100).toFixed(0)}% conf | Gate: ${gateStatus}`;

                // Positions (clickable to show chart)
                const posTable = document.getElementById('positions-table');
                if (positions.length === 0) {
                    posTable.innerHTML = '<tr><td colspan="5" class="neutral">No open positions</td></tr>';
                } else {
                    posTable.innerHTML = positions.map(p => {
                        return `
                        <tr onclick="showChart('${p.symbol}')" style="cursor:pointer">
                            <td><strong>${p.symbol}</strong></td>
                            <td>${p.qty.toFixed(4)}</td>
                            <td>${formatCurrency(p.entry_price)}</td>
                            <td>${formatCurrency(p.current_price)}</td>
                            <td>${formatPnl(p.unrealized_pnl, p.unrealized_pnl_pct)}</td>
                        </tr>
                    `}).join('');
                }

                // Watchlists (clickable to show chart) with sparklines
                const renderWatchlist = (elemId, list, sparklines) => {
                    const elem = document.getElementById(elemId);
                    if (!list || list.length === 0) {
                        elem.innerHTML = '<tr><td colspan="5" class="neutral">None</td></tr>';
                        return;
                    }
                    elem.innerHTML = list.map(item => {
                        const priceDisplay = item.price > 0 ? formatCurrency(item.price) : '--.--';
                        const changeDisplay = item.change !== 0 ? `${item.change > 0 ? '+' : ''}${item.change.toFixed(2)}%` : '--';
                        const color = item.change > 0 ? '#3fb950' : (item.change < 0 ? '#f85149' : '#8b949e');
                        const colorClass = item.change > 0 ? 'positive' : (item.change < 0 ? 'negative' : 'neutral');
                        const sparkline = sparklineSvg(sparklines[item.symbol], color);
                        const headline = (item.news_headline || '').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
                        const newsUrl = item.news_url || '';
                        const catalystBadge = item.has_catalyst
                            ? (newsUrl
                                ? `<a href="${newsUrl}" target="_blank" rel="noopener" onclick="event.stopPropagation()" style="background:#d2992222;color:#d29922;padding:2px 6px;border-radius:3px;font-size:10px;text-decoration:none;cursor:pointer" title="${headline}">NEWS (${item.news_count})</a>`
                                : `<span style="background:#d2992222;color:#d29922;padding:2px 6px;border-radius:3px;font-size:10px;cursor:help" title="${headline}">NEWS (${item.news_count})</span>`)
                            : '<span class="neutral" style="font-size:10px">--</span>';

                        return `
                            <tr onclick="showChart('${item.symbol}')">
                                <td class="symbol-cell">${item.symbol}</td>
                                <td class="sparkline-cell">${sparkline}</td>
                                <td class="price-cell">${priceDisplay}</td>
                                <td class="change-cell ${colorClass}">${changeDisplay}</td>
                                <td>${catalystBadge}</td>
                            </tr>
                        `;
                    }).join('');
                };
                renderWatchlist('stock-watchlist', watchlists.stocks, sparklineData.stocks || {});

                // Jobs
                const jobsTable = document.getElementById('jobs-table');
                if (jobs.length === 0) {
                    jobsTable.innerHTML = '<tr><td colspan="2" class="neutral">No scheduled jobs</td></tr>';
                } else {
                    jobsTable.innerHTML = jobs.map(j => {
                        let nextRunDisplay = '-';
                        if (j.next_run) {
                            const nextDate = new Date(j.next_run);
                            const now = new Date();
                            const diffMs = nextDate - now;
                            const diffMins = Math.round(diffMs / 60000);
                            if (diffMins < 1) {
                                nextRunDisplay = '<span class="positive">< 1 min</span>';
                            } else if (diffMins < 60) {
                                nextRunDisplay = `in ${diffMins} min`;
                            } else {
                                const hours = Math.floor(diffMins / 60);
                                const mins = diffMins % 60;
                                nextRunDisplay = `in ${hours}h ${mins}m`;
                            }
                            nextRunDisplay += ` <span class="neutral" style="font-size:11px">(${nextDate.toLocaleTimeString()})</span>`;
                        }
                        return `
                            <tr>
                                <td>${j.name}</td>
                                <td>${nextRunDisplay}</td>
                            </tr>
                        `;
                    }).join('');
                }

                // Press releases table
                const prTable = document.getElementById('pr-table');
                const prCount = document.getElementById('pr-count');
                const prHits = (prData && prData.hits) || [];
                if (prHits.length === 0) {
                    prTable.innerHTML = '<tr><td colspan="4" style="color:#8b949e">No press releases detected yet</td></tr>';
                    prCount.textContent = '';
                } else {
                    const posCount = prHits.filter(h => h.sentiment === 'positive').length;
                    prCount.textContent = `(${prHits.length} total, ${posCount} positive)`;
                    prTable.innerHTML = prHits.slice(0, 20).map(h => {
                        const sentCls = h.sentiment === 'positive' ? 'positive'
                            : h.sentiment === 'negative' ? 'negative' : 'neutral';
                        const sentIcon = h.sentiment === 'positive' ? '&#x2191;'
                            : h.sentiment === 'negative' ? '&#x2193;' : '&#x25CF;';
                        const headline = h.headline.length > 80 ? h.headline.substring(0, 80) + '...' : h.headline;
                        return `
                            <tr>
                                <td><strong>${h.symbol}</strong></td>
                                <td style="font-size:12px">${headline}</td>
                                <td style="font-size:11px;color:#8b949e">${h.source}</td>
                                <td class="${sentCls}">${sentIcon} ${h.sentiment}</td>
                            </tr>
                        `;
                    }).join('');
                }

            } catch (e) {
                console.error('Fetch error:', e);
            }
        }

        // Initial load
        fetchSparklines();
        fetchData();

        // Refresh data every 30s, sparklines every 60s (less frequent - more expensive)
        setInterval(fetchData, 60000);
        setInterval(fetchSparklines, 60000);

        // Chart functionality
        let candleChart = null;
        let macdChart = null;
        let candleSeries = null;
        let macdLine = null;
        let signalLine = null;
        let histogramSeries = null;
        let chartRefreshInterval = null;
        const CHART_REFRESH_MS = 30000; // 30 seconds
        let currentWatchlistData = { stocks: [], crypto: [] };
        let currentPositionsData = [];
        let currentChartSymbol = null;
        let currentWatchlist = null; // 'stocks' or 'crypto'
        let currentSymbolIndex = -1;

        async function toggleFullDay(enabled) {
            const toggle = document.getElementById('full-day-toggle');
            toggle._userClicked = true;
            try {
                const resp = await fetch(`api/settings/full-day-trading?enabled=${enabled}`, { method: 'POST' });
                const data = await resp.json();
                if (data.error) {
                    toggle.checked = !enabled;
                }
            } catch (e) {
                toggle.checked = !enabled;
            }
            // Let next fetchData cycle pick up the new state
            setTimeout(() => { toggle._userClicked = false; fetchData(); }, 500);
        }

        async function toggleRegimeGate(enabled) {
            const toggle = document.getElementById('regime-gate-toggle');
            toggle._userClicked = true;
            try {
                const resp = await fetch(`api/settings/regime-gate?enabled=${enabled}`, { method: 'POST' });
                const data = await resp.json();
                if (data.error) {
                    toggle.checked = !enabled;
                }
            } catch (e) {
                toggle.checked = !enabled;
            }
            setTimeout(() => { toggle._userClicked = false; fetchData(); }, 500);
        }

        async function refreshChartData(symbol) {
            // Fetch latest bars and update series in-place (no chart rebuild)
            try {
                const data = await fetch(`api/bars/${symbol}?timeframe=5Min&limit=100`).then(r => r.json());
                if (data.error || !candleSeries) return;
                candleSeries.setData(data.candles);
                if (macdLine) macdLine.setData(data.macd.map(d => ({ time: d.time, value: d.macd })));
                if (signalLine) signalLine.setData(data.macd.map(d => ({ time: d.time, value: d.signal })));
                if (histogramSeries) histogramSeries.setData(data.macd.map(d => ({
                    time: d.time, value: d.histogram,
                    color: d.histogram >= 0 ? '#3fb950' : '#f85149',
                })));
            } catch (e) { /* silent refresh failure */ }
        }

        async function showChart(symbol) {
            document.getElementById('chart-modal').classList.add('active');
            document.getElementById('chart-title').textContent = symbol + ' - Loading...';

            // Stop any existing refresh
            if (chartRefreshInterval) { clearInterval(chartRefreshInterval); chartRefreshInterval = null; }

            currentChartSymbol = symbol;
            currentSymbolIndex = currentWatchlistData.stocks.findIndex(item => item.symbol === symbol);
            if (currentSymbolIndex !== -1) {
                currentWatchlist = 'stocks';
            } else {
                currentSymbolIndex = currentWatchlistData.crypto.findIndex(item => item.symbol === symbol);
                if (currentSymbolIndex !== -1) {
                    currentWatchlist = 'crypto';
                } else {
                    currentWatchlist = null;
                    currentSymbolIndex = -1;
                }
            }

            // Clear existing charts
            document.getElementById('chart-container').innerHTML = '<p style="padding:20px;color:#8b949e">Loading chart...</p>';
            document.getElementById('macd-container').innerHTML = '';

            try {
                // Fetch asset info and bars in parallel
                const [assetInfo, data] = await Promise.all([
                    fetch(`api/asset/${symbol}`).then(r => r.json()),
                    fetch(`api/bars/${symbol}?timeframe=5Min&limit=100`).then(r => r.json())
                ]);

                // Update title with company name
                const companyName = assetInfo.name || symbol;
                const exchange = assetInfo.exchange ? ` (${assetInfo.exchange})` : '';
                document.getElementById('chart-title').innerHTML = `<strong>${symbol}</strong> - ${companyName}${exchange} <span style="color:#8b949e;font-size:12px">5Min | auto-refresh 30s</span>`;
                if (data.error) {
                    document.getElementById('chart-container').innerHTML = `<p style="padding:20px;color:#f85149">${data.error}</p>`;
                    return;
                }

                // Clear loading message
                document.getElementById('chart-container').innerHTML = '';

                // Create candlestick chart
                candleChart = LightweightCharts.createChart(document.getElementById('chart-container'), {
                    layout: { background: { color: '#161b22' }, textColor: '#c9d1d9' },
                    grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
                    width: document.getElementById('chart-container').clientWidth,
                    height: 400,
                    timeScale: { timeVisible: true, secondsVisible: false },
                });
                candleSeries = candleChart.addCandlestickSeries({
                    upColor: '#3fb950', downColor: '#f85149',
                    borderUpColor: '#3fb950', borderDownColor: '#f85149',
                    wickUpColor: '#3fb950', wickDownColor: '#f85149',
                });
                candleSeries.setData(data.candles);

                // Mark entry price for open positions
                const pos = currentPositionsData.find(p => p.symbol === symbol);
                if (pos && pos.entry_price) {
                    candleSeries.createPriceLine({
                        price: pos.entry_price,
                        color: '#58a6ff',
                        lineWidth: 1,
                        lineStyle: LightweightCharts.LineStyle.Dashed,
                        axisLabelVisible: true,
                        title: `Entry $${pos.entry_price.toFixed(2)}`,
                    });
                }

                // Create MACD chart
                macdChart = LightweightCharts.createChart(document.getElementById('macd-container'), {
                    layout: { background: { color: '#161b22' }, textColor: '#c9d1d9' },
                    grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
                    width: document.getElementById('macd-container').clientWidth,
                    height: 150,
                    timeScale: { timeVisible: true, secondsVisible: false },
                });

                // MACD line (blue)
                macdLine = macdChart.addLineSeries({ color: '#58a6ff', lineWidth: 1 });
                macdLine.setData(data.macd.map(d => ({ time: d.time, value: d.macd })));

                // Signal line (orange)
                signalLine = macdChart.addLineSeries({ color: '#d29922', lineWidth: 1 });
                signalLine.setData(data.macd.map(d => ({ time: d.time, value: d.signal })));

                // Histogram (green/red bars)
                histogramSeries = macdChart.addHistogramSeries({
                    color: '#3fb950',
                    priceFormat: { type: 'price' },
                });
                histogramSeries.setData(data.macd.map(d => ({
                    time: d.time,
                    value: d.histogram,
                    color: d.histogram >= 0 ? '#3fb950' : '#f85149',
                })));

                // Fit content
                candleChart.timeScale().fitContent();
                macdChart.timeScale().fitContent();

                // Sync time scales
                candleChart.timeScale().subscribeVisibleTimeRangeChange(() => {
                    const range = candleChart.timeScale().getVisibleRange();
                    if (range) macdChart.timeScale().setVisibleRange(range);
                });

                // Start auto-refresh
                chartRefreshInterval = setInterval(() => refreshChartData(symbol), CHART_REFRESH_MS);

            } catch (e) {
                document.getElementById('chart-container').innerHTML = `<p style="padding:20px;color:#f85149">Error: ${e}</p>`;
            }
        }

        function closeChart() {
            document.getElementById('chart-modal').classList.remove('active');
            if (chartRefreshInterval) { clearInterval(chartRefreshInterval); chartRefreshInterval = null; }
            if (candleChart) { candleChart.remove(); candleChart = null; }
            if (macdChart) { macdChart.remove(); macdChart = null; }
            candleSeries = null; macdLine = null; signalLine = null; histogramSeries = null;
            currentChartSymbol = null;
            currentWatchlist = null;
            currentSymbolIndex = -1;
        }

        function showNextChart() {
            if (currentWatchlist && currentSymbolIndex !== -1) {
                const watchlist = currentWatchlistData[currentWatchlist];
                const nextIndex = (currentSymbolIndex + 1) % watchlist.length;
                showChart(watchlist[nextIndex].symbol);
            }
        }

        function showPrevChart() {
            if (currentWatchlist && currentSymbolIndex !== -1) {
                const watchlist = currentWatchlistData[currentWatchlist];
                const prevIndex = (currentSymbolIndex - 1 + watchlist.length) % watchlist.length;
                showChart(watchlist[prevIndex].symbol);
            }
        }

        // Close on overlay click
        document.getElementById('chart-modal').addEventListener('click', (e) => {
            if (e.target.id === 'chart-modal') closeChart();
        });

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeChart();
        });
    </script>
</body>
</html>
"""
