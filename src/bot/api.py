"""
Simple web API for bot monitoring dashboard.
"""

from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


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

    try:
        account = _bot.client.get_account()
        equity = float(account.get("equity", 0))
        buying_power = float(account.get("buying_power", 0))

        # Fetch positions directly from Alpaca for accurate P&L
        positions = _bot.client.get_positions()
        unrealized_pnl = sum(p["unrealized_pl"] for p in positions)
        total_cost = sum(p["cost_basis"] for p in positions)

        # Get stats from Alpaca order history (no local ledger needed)
        stats = _bot.client.get_trade_stats()

        last_sync_dt = _bot.bot_state.get_job_timestamp("broker_sync")
        last_sync = last_sync_dt.isoformat() if last_sync_dt else None

        # Experiment tracking: use actual Alpaca equity
        # $400 = 0% progress, $4000 = 100% progress
        starting_capital = 400.0
        goal = 4000.0
        total_pnl = equity - starting_capital  # Actual P&L from Alpaca equity
        progress_pct = ((equity - starting_capital) / (goal - starting_capital)) * 100

        return {
            "running": _bot._running,
            "mode": _bot.config.trading_mode.value,
            "is_trading_day": _bot.scheduler.is_trading_day(),
            "market_open": _bot.scheduler.is_market_open(),
            "in_trading_window": _bot.scheduler.is_in_trading_window(),
            "in_premarket": _bot.scheduler.is_in_premarket(),
            "equity": equity,
            "buying_power": buying_power,
            # Experiment tracking - based on actual Alpaca equity
            "starting_capital": starting_capital,
            "goal": goal,
            "realized_pnl": stats["total_realized_pnl"],
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "current_value": equity,  # Use actual Alpaca equity
            "progress_pct": progress_pct,  # Can be negative if below $400
            "remaining": goal - equity,
            # Stats
            "total_trades": stats["total_trades"],
            "win_rate": stats["win_rate"],
            "position_count": len(positions),
            # Day trading
            "trades_today": _bot._daily_trades_today,
            "max_daily_trades": _bot.config.max_daily_trades,
            "scanner_hits": len(_bot._scanner_results),
            "trading_window": f"{_bot.config.trading_window_start}-{_bot.config.trading_window_end} ET",
            "last_sync": last_sync,
            "timestamp": datetime.now().isoformat(),
            # WebSocket streaming status
            "ws_data_connected": _bot.ws_client.data_connected if hasattr(_bot, "ws_client") else False,
            "ws_trade_connected": _bot.ws_client.trade_connected if hasattr(_bot, "ws_client") else False,
            "ws_subscribed_symbols": sorted(list(_bot.ws_client._subscribed_bars)) if hasattr(_bot, "ws_client") else [],
            # Press release scanner
            "pr_catalyst_count": _bot.press_release_scanner.get_status()["positive_hits"] if hasattr(_bot, "press_release_scanner") else 0,
            "pr_catalyst_symbols": _bot.press_release_scanner.get_catalyst_symbols(positive_only=True)[:10] if hasattr(_bot, "press_release_scanner") else [],
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/positions")
async def get_positions() -> list[dict]:
    """Get all positions directly from Alpaca."""
    if not _bot:
        return []

    try:
        # Fetch directly from Alpaca for accurate real-time data
        positions = _bot.client.get_positions()
        results = []
        for p in positions:
            is_crypto = p["asset_class"] == "crypto"
            market_value = p["market_value"]

            # Estimate fees for net P&L display
            fee_rate = 0.0025 if is_crypto else 0.0
            fees = market_value * fee_rate
            net_proceeds = market_value - fees
            net_pnl = net_proceeds - p["cost_basis"]

            results.append({
                "symbol": p["symbol"],
                "side": p["side"],
                "qty": p["qty"],
                "entry_price": p["avg_entry_price"],
                "current_price": p["current_price"],
                "unrealized_pnl": p["unrealized_pl"],
                "unrealized_pnl_pct": p["unrealized_plpc"] * 100,
                "cost_basis": p["cost_basis"],
                "market_value": market_value,
                "is_crypto": is_crypto,
                "fees": fees,
                "net_proceeds": net_proceeds,
                "net_pnl": net_pnl,
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
                "news_connected": ws_status.get("news_connected", False),
                "feed": ws_status.get("feed", ""),
                "paper": ws_status.get("paper", True),
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
    """Get trade history from Alpaca closed orders."""
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
    """Get trade history with P&L from Alpaca order history."""
    if not _bot:
        return {"error": "Bot not initialized"}

    try:
        # Get stats directly from Alpaca order history
        stats = _bot.client.get_trade_stats()

        # Get current equity for experiment progress
        account = _bot.client.get_account()
        equity = float(account.get("equity", 0))
        starting_capital = 400.0
        goal = 4000.0

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


@app.get("/api/asset/{symbol}")
async def get_asset_info(symbol: str) -> dict:
    """Get asset name and details from Alpaca."""
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

        # Fetch bars from Alpaca
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
    </style>
</head>
<body>
    <div class="container">
        <h1>Momentum Day Trader <span id="ws-indicator" style="font-size:12px;vertical-align:middle;margin-left:10px;padding:3px 10px;border-radius:12px;background:#f8514922;color:#f85149">WSS &#x25CF; Disconnected</span></h1>

        <div class="grid" id="status-grid">
            <div class="card">
                <div class="card-title">Status</div>
                <div class="card-value" id="status">Loading...</div>
                <div class="card-subtitle" id="trading-window">--</div>
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
                <div class="card-title">Today</div>
                <div class="card-value" id="trades-today">--</div>
                <div class="card-subtitle" id="scanner-info">--</div>
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

        <p class="refresh-note">Auto-refreshes every 30 seconds</p>
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
                    wsIndicator.innerHTML = `WSS &#x25CF; Connected (${wsSymCount} symbols)`;
                } else if (wsDataOk || wsTradeOk) {
                    wsIndicator.style.background = '#d2992222';
                    wsIndicator.style.color = '#d29922';
                    wsIndicator.innerHTML = `WSS &#x25CF; Partial`;
                } else {
                    wsIndicator.style.background = '#f8514922';
                    wsIndicator.style.color = '#f85149';
                    wsIndicator.innerHTML = `WSS &#x25CF; Disconnected`;
                }

                // Progress toward goal ($400 = 0%, $4000 = 100%)
                const currentVal = status.current_value || status.equity || 400;
                const goal = status.goal || 4000;
                const startingCapital = status.starting_capital || 400;
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

                // Trades today + scanner info
                const tradesToday = status.trades_today || 0;
                const maxDaily = status.max_daily_trades || 1;
                const scannerHits = status.scanner_hits || 0;
                const winRate = status.win_rate || 0;
                const totalTrades = status.total_trades || 0;
                document.getElementById('trades-today').innerHTML =
                    `${tradesToday}/${maxDaily} trades`;
                document.getElementById('scanner-info').innerHTML =
                    `${scannerHits} scanner hits | ${winRate.toFixed(0)}% win rate (${totalTrades} total)`;

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
                        const catalystBadge = item.has_catalyst
                            ? `<span style="background:#d2992222;color:#d29922;padding:2px 6px;border-radius:3px;font-size:10px;cursor:help" title="${headline}">NEWS (${item.news_count})</span>`
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
        setInterval(fetchData, 30000);
        setInterval(fetchSparklines, 60000);

        // Chart functionality
        let candleChart = null;
        let macdChart = null;
        let currentWatchlistData = { stocks: [], crypto: [] };
        let currentChartSymbol = null;
        let currentWatchlist = null; // 'stocks' or 'crypto'
        let currentSymbolIndex = -1;

        async function showChart(symbol) {
            document.getElementById('chart-modal').classList.add('active');
            document.getElementById('chart-title').textContent = symbol + ' - Loading...';

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
                document.getElementById('chart-title').innerHTML = `<strong>${symbol}</strong> - ${companyName}${exchange} <span style="color:#8b949e;font-size:12px">5Min</span>`;
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
                const candleSeries = candleChart.addCandlestickSeries({
                    upColor: '#3fb950', downColor: '#f85149',
                    borderUpColor: '#3fb950', borderDownColor: '#f85149',
                    wickUpColor: '#3fb950', wickDownColor: '#f85149',
                });
                candleSeries.setData(data.candles);

                // Create MACD chart
                macdChart = LightweightCharts.createChart(document.getElementById('macd-container'), {
                    layout: { background: { color: '#161b22' }, textColor: '#c9d1d9' },
                    grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
                    width: document.getElementById('macd-container').clientWidth,
                    height: 150,
                    timeScale: { timeVisible: true, secondsVisible: false },
                });

                // MACD line (blue)
                const macdLine = macdChart.addLineSeries({ color: '#58a6ff', lineWidth: 1 });
                macdLine.setData(data.macd.map(d => ({ time: d.time, value: d.macd })));

                // Signal line (orange)
                const signalLine = macdChart.addLineSeries({ color: '#d29922', lineWidth: 1 });
                signalLine.setData(data.macd.map(d => ({ time: d.time, value: d.signal })));

                // Histogram (green/red bars)
                const histogram = macdChart.addHistogramSeries({
                    color: '#3fb950',
                    priceFormat: { type: 'price' },
                });
                histogram.setData(data.macd.map(d => ({
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

            } catch (e) {
                document.getElementById('chart-container').innerHTML = `<p style="padding:20px;color:#f85149">Error: ${e}</p>`;
            }
        }

        function closeChart() {
            document.getElementById('chart-modal').classList.remove('active');
            if (candleChart) { candleChart.remove(); candleChart = null; }
            if (macdChart) { macdChart.remove(); macdChart = null; }
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
