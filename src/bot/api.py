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
    <title>Swing Trader</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        :root {
            --bg-canvas: #FAFAFA;
            --bg-card: #FFFFFF;
            --bg-sidebar: #0A0A0A;
            --text-primary: #0A0A0A;
            --text-secondary: #6B7280;
            --text-muted: #9CA3AF;
            --text-inverse: #FAFAFA;
            --accent: #DC2626;
            --accent-light: #FEE2E2;
            --success: #16A34A;
            --success-light: #DCFCE7;
            --warning: #D97706;
            --warning-light: #FEF3C7;
            --border: #E5E7EB;
            --border-light: #F3F4F6;
            --font-heading: 'Sora', sans-serif;
            --font-mono: 'IBM Plex Mono', monospace;
        }
        body {
            font-family: var(--font-mono);
            background: var(--bg-canvas);
            color: var(--text-primary);
            line-height: 1.5;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        /* Sidebar */
        .sidebar {
            width: 64px;
            background: var(--bg-sidebar);
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px 0;
            gap: 16px;
            flex-shrink: 0;
        }
        .sidebar-logo {
            width: 40px;
            height: 40px;
            background: var(--accent);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-inverse);
            font-family: var(--font-heading);
            font-weight: 700;
            font-size: 14px;
        }
        /* Main content */
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .content {
            flex: 1;
            overflow-y: auto;
            padding: 32px;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }
        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            font-family: var(--font-heading);
            font-size: 24px;
            font-weight: 600;
            color: var(--text-primary);
        }
        .header-right {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .status-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            font-weight: 600;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        .market-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            font-size: 10px;
            font-weight: 600;
        }
        /* Auth banner */
        #auth-banner {
            display: none;
        }
        #auth-banner .auth-card {
            background: var(--bg-card);
            border: 1px solid var(--warning);
            padding: 20px;
            text-align: center;
        }
        #auth-banner a {
            display: inline-block;
            background: var(--success);
            color: #fff;
            padding: 10px 24px;
            text-decoration: none;
            font-weight: 600;
            font-size: 13px;
        }
        /* KPI Grid */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
        }
        @media (max-width: 1000px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 600px) { .kpi-grid { grid-template-columns: 1fr; } }
        .kpi-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .kpi-label {
            font-size: 10px;
            font-weight: 500;
            color: var(--text-muted);
            letter-spacing: 1.2px;
            text-transform: uppercase;
        }
        .kpi-value {
            font-size: 28px;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1.1;
        }
        .kpi-sub {
            font-size: 11px;
            color: var(--text-secondary);
        }
        .progress-bar {
            height: 4px;
            background: var(--border);
            overflow: visible;
            position: relative;
        }
        .progress-fill {
            height: 100%;
            background: var(--accent);
            transition: width 0.5s ease;
        }
        .progress-fill.negative-fill {
            background: var(--accent);
            position: absolute;
            right: 100%;
        }
        .positive { color: var(--success); }
        .negative { color: var(--accent); }
        .neutral { color: var(--text-muted); }
        /* Content columns */
        .content-row {
            display: flex;
            gap: 16px;
            min-height: 300px;
        }
        @media (max-width: 900px) { .content-row { flex-direction: column; } }
        /* Section panels */
        .section {
            background: var(--bg-card);
            border: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 20px;
        }
        .section-title {
            font-family: var(--font-heading);
            font-size: 14px;
            font-weight: 600;
            color: var(--text-primary);
        }
        .section-count {
            padding: 2px 8px;
            font-size: 11px;
            font-weight: 600;
        }
        .section-divider {
            height: 1px;
            background: var(--border);
        }
        .section-body {
            flex: 1;
            overflow-y: auto;
        }
        .positions-section {
            flex: 1;
            min-height: 200px;
        }
        .scanner-section {
            width: 360px;
            flex-shrink: 0;
            min-height: 200px;
        }
        @media (max-width: 900px) { .scanner-section { width: 100%; } }
        /* Tables */
        table { width: 100%; border-collapse: collapse; }
        th {
            padding: 10px 20px;
            text-align: left;
            font-size: 10px;
            font-weight: 500;
            color: var(--text-muted);
            letter-spacing: 0.8px;
            text-transform: uppercase;
        }
        td {
            padding: 12px 20px;
            font-size: 13px;
            border-bottom: 1px solid var(--border-light);
        }
        tbody tr { cursor: pointer; transition: background 0.15s; }
        tbody tr:hover { background: var(--border-light); }
        .symbol-cell { font-weight: 600; }
        .side-badge {
            display: inline-block;
            padding: 2px 6px;
            font-size: 10px;
            font-weight: 600;
            text-align: center;
        }
        .side-badge.long { background: var(--success-light); color: var(--success); }
        .side-badge.short { background: var(--accent-light); color: var(--accent); }
        .strategy-cell {
            font-size: 11px;
            color: var(--text-muted);
        }
        /* Watchlist items */
        .watch-item {
            display: flex;
            align-items: center;
            padding: 12px 20px;
            gap: 12px;
            cursor: pointer;
            transition: background 0.15s;
            border-bottom: 1px solid var(--border-light);
        }
        .watch-item:hover { background: var(--border-light); }
        .watch-item-info {
            flex: 1;
        }
        .watch-item-symbol {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-primary);
        }
        .watch-item-detail {
            font-size: 10px;
            color: var(--text-secondary);
        }
        .news-badge {
            display: inline-block;
            padding: 2px 6px;
            background: var(--accent-light);
            color: var(--accent);
            font-size: 9px;
            font-weight: 700;
            text-decoration: none;
        }
        .news-badge:hover { opacity: 0.8; }
        /* Extra sections below main content */
        .extra-sections {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .extra-sections .section-title-row {
            font-family: var(--font-heading);
            font-size: 13px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        /* Status bar */
        .status-bar {
            display: flex;
            align-items: center;
            padding: 0 16px;
            height: 36px;
            background: var(--bg-card);
            border-top: 1px solid var(--border);
            gap: 16px;
            font-size: 10px;
            color: var(--text-muted);
            flex-shrink: 0;
        }
        .status-bar-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .status-bar-spacer { flex: 1; }
        .status-bar-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
        }
        /* Sparklines */
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
        /* Modal */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.5);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }
        .modal-overlay.active { display: flex; }
        .modal {
            background: var(--bg-card);
            border: 1px solid var(--border);
            width: 90%;
            max-width: 1000px;
            max-height: 90vh;
            overflow: hidden;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            border-bottom: 1px solid var(--border);
            gap: 12px;
        }
        .modal-title {
            font-family: var(--font-heading);
            font-size: 18px;
            font-weight: 600;
            color: var(--text-primary);
            flex: 1;
        }
        .modal-close {
            background: none;
            border: none;
            color: var(--text-muted);
            font-size: 24px;
            cursor: pointer;
        }
        .modal-close:hover { color: var(--text-primary); }
        .modal-nav-button {
            background: var(--bg-canvas);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 5px 10px;
            cursor: pointer;
            font-family: var(--font-mono);
            font-size: 12px;
            transition: background 0.2s;
        }
        .modal-nav-button:hover { background: var(--border-light); }
        .chart-info-bar {
            display: flex;
            gap: 24px;
            padding: 10px 20px;
            border-bottom: 1px solid var(--border);
            font-size: 10px;
        }
        .chart-info-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .chart-info-label { color: var(--text-muted); }
        .chart-info-value { font-weight: 600; font-size: 12px; }
        #chart-container { height: 400px; }
        #macd-container { height: 150px; border-top: 1px solid var(--border); }
    </style>
</head>
<body>
    <nav class="sidebar">
        <div class="sidebar-logo">ST</div>
    </nav>

    <div class="main">
        <div class="content">
            <div class="header">
                <h1>Dashboard</h1>
                <div class="header-right">
                    <div class="status-badge" id="ws-indicator">
                        <span class="status-dot" style="background:var(--accent)"></span>
                        <span style="color:var(--accent)">Disconnected</span>
                    </div>
                    <div class="market-badge" id="market-badge" style="background:var(--success-light);color:var(--success)">
                        MARKET OPEN
                    </div>
                </div>
            </div>

            <div id="auth-banner">
                <div class="auth-card">
                    <div style="font-size:18px;color:var(--warning);margin-bottom:10px">Not Connected to tastytrade</div>
                    <p style="color:var(--text-secondary);margin-bottom:15px">Complete OAuth setup to start trading</p>
                    <a href="oauth/authorize">Connect tastytrade Account</a>
                </div>
            </div>

            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Progress</div>
                    <div class="kpi-value" id="progress">--</div>
                    <div class="kpi-sub" id="progress-sub">$250 &rarr; $25,000</div>
                    <div class="progress-bar"><div class="progress-fill" id="progress-bar"></div></div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Total P&amp;L</div>
                    <div class="kpi-value" id="total-pnl">--</div>
                    <div class="kpi-sub" id="pnl-breakdown">--</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Win / Loss</div>
                    <div class="kpi-value" id="win-loss">--</div>
                    <div class="kpi-sub" id="win-loss-detail">--</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Regime</div>
                    <div class="kpi-value" id="regime-status">--</div>
                    <div class="kpi-sub" id="regime-detail">--</div>
                </div>
            </div>

            <div class="content-row">
                <div class="section positions-section">
                    <div class="section-header">
                        <span class="section-title">Open Positions</span>
                        <span class="section-count" id="pos-count" style="background:var(--accent-light);color:var(--accent)">0</span>
                    </div>
                    <div class="section-divider"></div>
                    <div class="section-body">
                        <table>
                            <thead>
                                <tr>
                                    <th>Symbol</th>
                                    <th>Side</th>
                                    <th>Entry</th>
                                    <th>Current</th>
                                    <th>P&amp;L</th>
                                    <th>Strategy</th>
                                </tr>
                            </thead>
                            <tbody id="positions-table">
                                <tr><td colspan="6" style="color:var(--text-muted)">Loading...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="section scanner-section">
                    <div class="section-header">
                        <span class="section-title">Scanner</span>
                        <span class="section-count" id="scanner-count" style="background:#EEF2FF;color:#4F46E5">0</span>
                    </div>
                    <div class="section-divider"></div>
                    <div class="section-body" id="stock-watchlist">
                        <div style="padding:20px;color:var(--text-muted);font-size:12px">Loading...</div>
                    </div>
                </div>
            </div>

            <div class="extra-sections">
                <div class="section">
                    <div class="section-header">
                        <span class="section-title">Press Releases <span id="pr-count" style="font-size:11px;color:var(--text-muted);font-weight:normal"></span></span>
                    </div>
                    <div class="section-divider"></div>
                    <div class="section-body">
                        <table>
                            <thead>
                                <tr><th>Symbol</th><th>Headline</th><th>Source</th><th>Sentiment</th></tr>
                            </thead>
                            <tbody id="pr-table">
                                <tr><td colspan="4" style="color:var(--text-muted)">No press releases yet</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="section">
                    <div class="section-header">
                        <span class="section-title">Scheduled Jobs</span>
                    </div>
                    <div class="section-divider"></div>
                    <div class="section-body">
                        <table>
                            <thead><tr><th>Job</th><th>Next Run</th></tr></thead>
                            <tbody id="jobs-table">
                                <tr><td colspan="2" style="color:var(--text-muted)">Loading...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <div class="status-bar">
            <div class="status-bar-item" id="stream-status">
                <span class="status-bar-dot" style="background:var(--text-muted)"></span>
                <span>Connecting...</span>
            </div>
            <div class="status-bar-spacer"></div>
            <div class="status-bar-item" id="regime-bar">
                <span class="status-bar-dot" style="background:var(--text-muted)"></span>
                <span>Regime: --</span>
            </div>
            <div class="status-bar-item" id="clock">--:--:-- ET</div>
        </div>
    </div>

    <div class="modal-overlay" id="chart-modal">
        <div class="modal">
            <div class="modal-header">
                <button class="modal-nav-button" onclick="showPrevChart()">&lt; Prev</button>
                <span class="modal-title" id="chart-title">Symbol</span>
                <span style="color:var(--text-muted);font-size:10px">auto-refresh 30s</span>
                <button class="modal-nav-button" onclick="showNextChart()">Next &gt;</button>
                <button class="modal-close" onclick="closeChart()">&times;</button>
            </div>
            <div class="chart-info-bar" id="chart-info-bar" style="display:none">
                <div class="chart-info-item"><span class="chart-info-label">Entry</span><span class="chart-info-value" id="chart-entry">--</span></div>
                <div class="chart-info-item"><span class="chart-info-label">Stop</span><span class="chart-info-value" id="chart-stop" style="color:var(--accent)">--</span></div>
                <div class="chart-info-item"><span class="chart-info-label">Strategy</span><span class="chart-info-value" id="chart-strategy">--</span></div>
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

        // Update clock in status bar
        function updateClock() {
            const now = new Date();
            const et = now.toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour12: false });
            document.getElementById('clock').textContent = et + ' ET';
        }
        setInterval(updateClock, 1000);
        updateClock();

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

                currentWatchlistData.stocks = watchlists.stocks || [];
                currentWatchlistData.crypto = watchlists.crypto || [];
                currentPositionsData = positions;

                // Auth banner
                const authBanner = document.getElementById('auth-banner');
                if (status.authenticated === false && !status.running) {
                    authBanner.style.display = 'block';
                } else {
                    authBanner.style.display = 'none';
                }

                // WebSocket / streaming status (status bar)
                const wsDataOk = status.ws_data_connected || false;
                const wsTradeOk = status.ws_trade_connected || false;
                const wsSymCount = (status.ws_subscribed_symbols || []).length;
                const streamEl = document.getElementById('stream-status');
                const wsIndicator = document.getElementById('ws-indicator');
                if (wsDataOk && wsTradeOk) {
                    streamEl.innerHTML = `<span class="status-bar-dot" style="background:var(--success)"></span><span>Streaming connected (${wsSymCount})</span>`;
                    wsIndicator.innerHTML = `<span class="status-dot" style="background:var(--success)"></span><span style="color:var(--success)">LIVE</span>`;
                } else if (wsDataOk || wsTradeOk) {
                    streamEl.innerHTML = `<span class="status-bar-dot" style="background:var(--warning)"></span><span>Streaming partial</span>`;
                    wsIndicator.innerHTML = `<span class="status-dot" style="background:var(--warning)"></span><span style="color:var(--warning)">PARTIAL</span>`;
                } else {
                    streamEl.innerHTML = `<span class="status-bar-dot" style="background:var(--accent)"></span><span>Disconnected</span>`;
                    wsIndicator.innerHTML = `<span class="status-dot" style="background:var(--accent)"></span><span style="color:var(--accent)">OFFLINE</span>`;
                }

                // Market badge
                const marketBadge = document.getElementById('market-badge');
                const isTradingDay = status.is_trading_day !== false;
                if (!isTradingDay) {
                    marketBadge.style.background = 'var(--accent-light)';
                    marketBadge.style.color = 'var(--accent)';
                    marketBadge.textContent = 'HOLIDAY';
                } else if (status.market_open) {
                    marketBadge.style.background = 'var(--success-light)';
                    marketBadge.style.color = 'var(--success)';
                    marketBadge.textContent = 'MARKET OPEN';
                } else {
                    marketBadge.style.background = 'var(--border-light)';
                    marketBadge.style.color = 'var(--text-muted)';
                    marketBadge.textContent = 'MARKET CLOSED';
                }

                // KPI: Progress
                const currentVal = status.current_value || status.equity || 250;
                const startingCapital = status.starting_capital || 250;
                const progressPct = status.progress_pct || 0;
                const pnlClass = currentVal >= startingCapital ? 'positive' : 'negative';
                document.getElementById('progress').innerHTML = `<span class="${pnlClass}">${formatCurrency(currentVal)}</span>`;
                document.getElementById('progress-sub').innerHTML = `$${startingCapital.toFixed(0)} &rarr; $25,000`;
                const progressBar = document.getElementById('progress-bar');
                if (progressPct >= 0) {
                    progressBar.className = 'progress-fill';
                    progressBar.style.width = `${Math.min(100, progressPct)}%`;
                } else {
                    progressBar.className = 'progress-fill negative-fill';
                    progressBar.style.width = `${Math.min(10, Math.abs(progressPct))}%`;
                }

                // KPI: Total P&L
                const totalPnl = status.total_pnl || 0;
                const realizedPnl = status.realized_pnl || 0;
                const unrealizedPnl = status.unrealized_pnl || 0;
                const totalSign = totalPnl >= 0 ? '+' : '';
                const totalCls = totalPnl >= 0 ? 'positive' : 'negative';
                document.getElementById('total-pnl').innerHTML = `<span class="${totalCls}">${totalSign}${formatCurrency(totalPnl)}</span>`;
                const rSign = realizedPnl >= 0 ? '+' : '';
                const uSign = unrealizedPnl >= 0 ? '+' : '';
                document.getElementById('pnl-breakdown').innerHTML = `R: ${rSign}$${realizedPnl.toFixed(2)} | U: ${uSign}$${unrealizedPnl.toFixed(2)}`;

                // KPI: Win/Loss
                const winCount = status.win_count || 0;
                const lossCount = status.loss_count || 0;
                const winRate = status.win_rate || 0;
                const totalTrades = status.total_trades || 0;
                document.getElementById('win-loss').innerHTML =
                    `<span class="positive">${winCount}W</span> <span class="neutral" style="font-weight:400">&mdash;</span> <span class="negative">${lossCount}L</span>`;
                document.getElementById('win-loss-detail').innerHTML =
                    `${winRate.toFixed(0)}% win rate &middot; ${totalTrades} total`;

                // KPI: Regime (read-only)
                const regimeLabel = status.regime_label || '--';
                const regimeCat = status.regime_category || 'neutral';
                const regimeConf = status.regime_confidence || 0;
                const regimeColor = regimeCat === 'bullish' ? 'var(--success)'
                    : regimeCat === 'bearish' ? 'var(--accent)' : 'var(--warning)';
                document.getElementById('regime-status').innerHTML = `<span style="color:${regimeColor}">${regimeLabel.toUpperCase()}</span>`;
                document.getElementById('regime-detail').textContent =
                    `${(regimeConf * 100).toFixed(0)}% confidence`;

                // Status bar: regime
                const regimeBar = document.getElementById('regime-bar');
                const regimeDotColor = regimeCat === 'bullish' ? 'var(--success)'
                    : regimeCat === 'bearish' ? 'var(--accent)' : 'var(--warning)';
                regimeBar.innerHTML = `<span class="status-bar-dot" style="background:${regimeDotColor}"></span><span>Regime: ${regimeLabel}</span>`;

                // Positions table
                const posTable = document.getElementById('positions-table');
                document.getElementById('pos-count').textContent = positions.length;
                if (positions.length === 0) {
                    posTable.innerHTML = '<tr><td colspan="6" style="color:var(--text-muted)">No open positions</td></tr>';
                } else {
                    posTable.innerHTML = positions.map(p => {
                        const pnlSign = p.unrealized_pnl >= 0 ? '+' : '';
                        const pnlCls = p.unrealized_pnl >= 0 ? 'positive' : 'negative';
                        const sideClass = p.side === 'long' ? 'long' : 'short';
                        return `
                        <tr onclick="showChart('${p.symbol}')">
                            <td class="symbol-cell">${p.symbol}</td>
                            <td><span class="side-badge ${sideClass}">${p.side.toUpperCase()}</span></td>
                            <td>${formatCurrency(p.entry_price)}</td>
                            <td style="font-weight:600">${formatCurrency(p.current_price)}</td>
                            <td class="${pnlCls}" style="font-weight:600">${pnlSign}${formatCurrency(p.unrealized_pnl)}</td>
                            <td class="strategy-cell">${p.strategy || '--'}</td>
                        </tr>`;
                    }).join('');
                }

                // Scanner / watchlist
                const watchlistEl = document.getElementById('stock-watchlist');
                const scannerCount = document.getElementById('scanner-count');
                const stocks = watchlists.stocks || [];
                scannerCount.textContent = stocks.length;
                if (stocks.length === 0) {
                    watchlistEl.innerHTML = '<div style="padding:20px;color:var(--text-muted);font-size:12px">No scanner results</div>';
                } else {
                    watchlistEl.innerHTML = stocks.map(item => {
                        const changeDisplay = item.change !== 0 ? `${item.change > 0 ? '+' : ''}${item.change.toFixed(1)}%` : '--';
                        const headline = (item.news_headline || '').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
                        const newsUrl = item.news_url || '';
                        const newsBadge = item.has_catalyst
                            ? (newsUrl
                                ? `<a href="${newsUrl}" target="_blank" rel="noopener" onclick="event.stopPropagation()" class="news-badge" title="${headline}">NEWS</a>`
                                : `<span class="news-badge" title="${headline}">NEWS</span>`)
                            : '';
                        return `
                        <div class="watch-item" onclick="showChart('${item.symbol}')">
                            <div class="watch-item-info">
                                <div class="watch-item-symbol">${item.symbol}</div>
                                <div class="watch-item-detail">${item.price > 0 ? formatCurrency(item.price) : '--'} &middot; ${changeDisplay}</div>
                            </div>
                            ${newsBadge}
                        </div>`;
                    }).join('');
                }

                // Jobs
                const jobsTable = document.getElementById('jobs-table');
                if (jobs.length === 0) {
                    jobsTable.innerHTML = '<tr><td colspan="2" style="color:var(--text-muted)">No scheduled jobs</td></tr>';
                } else {
                    jobsTable.innerHTML = jobs.map(j => {
                        let nextRunDisplay = '-';
                        if (j.next_run) {
                            const nextDate = new Date(j.next_run);
                            const diffMins = Math.round((nextDate - new Date()) / 60000);
                            if (diffMins < 1) nextRunDisplay = '<span class="positive">&lt; 1 min</span>';
                            else if (diffMins < 60) nextRunDisplay = `in ${diffMins} min`;
                            else nextRunDisplay = `in ${Math.floor(diffMins/60)}h ${diffMins%60}m`;
                            nextRunDisplay += ` <span class="neutral" style="font-size:11px">(${nextDate.toLocaleTimeString()})</span>`;
                        }
                        return `<tr><td>${j.name}</td><td>${nextRunDisplay}</td></tr>`;
                    }).join('');
                }

                // Press releases
                const prTable = document.getElementById('pr-table');
                const prCount = document.getElementById('pr-count');
                const prHits = (prData && prData.hits) || [];
                if (prHits.length === 0) {
                    prTable.innerHTML = '<tr><td colspan="4" style="color:var(--text-muted)">No press releases detected yet</td></tr>';
                    prCount.textContent = '';
                } else {
                    const posCount = prHits.filter(h => h.sentiment === 'positive').length;
                    prCount.textContent = `(${prHits.length} total, ${posCount} positive)`;
                    prTable.innerHTML = prHits.slice(0, 20).map(h => {
                        const sentCls = h.sentiment === 'positive' ? 'positive' : h.sentiment === 'negative' ? 'negative' : 'neutral';
                        const sentIcon = h.sentiment === 'positive' ? '&#x2191;' : h.sentiment === 'negative' ? '&#x2193;' : '&#x25CF;';
                        const headline = h.headline.length > 80 ? h.headline.substring(0, 80) + '...' : h.headline;
                        return `<tr><td><strong>${h.symbol}</strong></td><td style="font-size:12px">${headline}</td><td style="font-size:11px;color:var(--text-muted)">${h.source}</td><td class="${sentCls}">${sentIcon} ${h.sentiment}</td></tr>`;
                    }).join('');
                }

            } catch (e) {
                console.error('Fetch error:', e);
            }
        }

        // Initial load
        fetchSparklines();
        fetchData();
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
        const CHART_REFRESH_MS = 30000;
        let currentWatchlistData = { stocks: [], crypto: [] };
        let currentPositionsData = [];
        let currentChartSymbol = null;
        let currentWatchlist = null;
        let currentSymbolIndex = -1;

        async function refreshChartData(symbol) {
            try {
                const data = await fetch(`api/bars/${symbol}?timeframe=5Min&limit=100`).then(r => r.json());
                if (data.error || !candleSeries) return;
                candleSeries.setData(data.candles);
                if (macdLine) macdLine.setData(data.macd.map(d => ({ time: d.time, value: d.macd })));
                if (signalLine) signalLine.setData(data.macd.map(d => ({ time: d.time, value: d.signal })));
                if (histogramSeries) histogramSeries.setData(data.macd.map(d => ({
                    time: d.time, value: d.histogram,
                    color: d.histogram >= 0 ? '#16A34A' : '#DC2626',
                })));
            } catch (e) { /* silent */ }
        }

        async function showChart(symbol) {
            document.getElementById('chart-modal').classList.add('active');
            document.getElementById('chart-title').textContent = symbol + ' - Loading...';
            if (chartRefreshInterval) { clearInterval(chartRefreshInterval); chartRefreshInterval = null; }

            currentChartSymbol = symbol;
            currentSymbolIndex = currentWatchlistData.stocks.findIndex(item => item.symbol === symbol);
            currentWatchlist = currentSymbolIndex !== -1 ? 'stocks' : null;

            // Show position info bar if we have a position
            const pos = currentPositionsData.find(p => p.symbol === symbol);
            const infoBar = document.getElementById('chart-info-bar');
            if (pos) {
                infoBar.style.display = 'flex';
                document.getElementById('chart-entry').textContent = formatCurrency(pos.entry_price);
                document.getElementById('chart-strategy').textContent = pos.strategy || '--';
                document.getElementById('chart-stop').textContent = '--';
            } else {
                infoBar.style.display = 'none';
            }

            document.getElementById('chart-container').innerHTML = '<p style="padding:20px;color:var(--text-muted)">Loading chart...</p>';
            document.getElementById('macd-container').innerHTML = '';

            try {
                const [assetInfo, data] = await Promise.all([
                    fetch(`api/asset/${symbol}`).then(r => r.json()),
                    fetch(`api/bars/${symbol}?timeframe=5Min&limit=100`).then(r => r.json())
                ]);

                const companyName = assetInfo.name || symbol;
                const exchange = assetInfo.exchange ? ` (${assetInfo.exchange})` : '';
                document.getElementById('chart-title').innerHTML = `<strong>${symbol}</strong> &mdash; ${companyName}${exchange}`;

                if (data.error) {
                    document.getElementById('chart-container').innerHTML = `<p style="padding:20px;color:var(--accent)">${data.error}</p>`;
                    return;
                }

                document.getElementById('chart-container').innerHTML = '';

                candleChart = LightweightCharts.createChart(document.getElementById('chart-container'), {
                    layout: { background: { color: '#FFFFFF' }, textColor: '#6B7280' },
                    grid: { vertLines: { color: '#F3F4F6' }, horzLines: { color: '#F3F4F6' } },
                    width: document.getElementById('chart-container').clientWidth,
                    height: 400,
                    timeScale: { timeVisible: true, secondsVisible: false },
                });
                candleSeries = candleChart.addCandlestickSeries({
                    upColor: '#16A34A', downColor: '#DC2626',
                    borderUpColor: '#16A34A', borderDownColor: '#DC2626',
                    wickUpColor: '#16A34A', wickDownColor: '#DC2626',
                });
                candleSeries.setData(data.candles);

                if (pos && pos.entry_price) {
                    candleSeries.createPriceLine({
                        price: pos.entry_price,
                        color: '#3B82F6',
                        lineWidth: 1,
                        lineStyle: LightweightCharts.LineStyle.Dashed,
                        axisLabelVisible: true,
                        title: `Entry $${pos.entry_price.toFixed(2)}`,
                    });
                }

                macdChart = LightweightCharts.createChart(document.getElementById('macd-container'), {
                    layout: { background: { color: '#FFFFFF' }, textColor: '#6B7280' },
                    grid: { vertLines: { color: '#F3F4F6' }, horzLines: { color: '#F3F4F6' } },
                    width: document.getElementById('macd-container').clientWidth,
                    height: 150,
                    timeScale: { timeVisible: true, secondsVisible: false },
                });

                macdLine = macdChart.addLineSeries({ color: '#3B82F6', lineWidth: 1 });
                macdLine.setData(data.macd.map(d => ({ time: d.time, value: d.macd })));

                signalLine = macdChart.addLineSeries({ color: '#D97706', lineWidth: 1 });
                signalLine.setData(data.macd.map(d => ({ time: d.time, value: d.signal })));

                histogramSeries = macdChart.addHistogramSeries({ color: '#16A34A', priceFormat: { type: 'price' } });
                histogramSeries.setData(data.macd.map(d => ({
                    time: d.time, value: d.histogram,
                    color: d.histogram >= 0 ? '#16A34A' : '#DC2626',
                })));

                candleChart.timeScale().fitContent();
                macdChart.timeScale().fitContent();
                candleChart.timeScale().subscribeVisibleTimeRangeChange(() => {
                    const range = candleChart.timeScale().getVisibleRange();
                    if (range) macdChart.timeScale().setVisibleRange(range);
                });

                chartRefreshInterval = setInterval(() => refreshChartData(symbol), CHART_REFRESH_MS);

            } catch (e) {
                document.getElementById('chart-container').innerHTML = `<p style="padding:20px;color:var(--accent)">Error: ${e}</p>`;
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
                showChart(watchlist[(currentSymbolIndex + 1) % watchlist.length].symbol);
            }
        }

        function showPrevChart() {
            if (currentWatchlist && currentSymbolIndex !== -1) {
                const watchlist = currentWatchlistData[currentWatchlist];
                showChart(watchlist[(currentSymbolIndex - 1 + watchlist.length) % watchlist.length].symbol);
            }
        }

        document.getElementById('chart-modal').addEventListener('click', (e) => {
            if (e.target.id === 'chart-modal') closeChart();
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeChart();
        });
    </script>
</body>
</html>
"""
