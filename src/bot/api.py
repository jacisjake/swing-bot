"""
Simple web API for bot monitoring dashboard.
"""

from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


app = FastAPI(title="Swing Trader Bot", version="1.0.0")

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
    stop_loss: Optional[float]
    take_profit: Optional[float]


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

        positions = _bot.position_manager.get_open_positions()
        total_pnl = sum(p.unrealized_pnl for p in positions)
        total_cost = sum(p.cost_basis for p in positions)
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

        last_sync_dt = _bot.bot_state.get_job_timestamp("broker_sync")
        last_sync = last_sync_dt.isoformat() if last_sync_dt else None

        return {
            "running": _bot._running,
            "mode": _bot.config.trading_mode.value,
            "market_open": _bot.scheduler.is_market_open(),
            "equity": equity,
            "buying_power": buying_power,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "position_count": len(positions),
            "last_sync": last_sync,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/positions")
async def get_positions() -> list[dict]:
    """Get all positions."""
    if not _bot:
        return []

    try:
        positions = _bot.position_manager.get_open_positions()
        results = []
        for p in positions:
            market_value = p.market_value
            is_crypto = p.symbol.endswith("USD") or "/" in p.symbol

            # Alpaca fees: ~0.25% for crypto, $0 for stocks
            fee_rate = 0.0025 if is_crypto else 0.0
            fees = market_value * fee_rate
            net_proceeds = market_value - fees
            net_pnl = net_proceeds - p.cost_basis

            results.append({
                "symbol": p.symbol,
                "side": p.side.value,
                "qty": p.qty,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "unrealized_pnl": p.unrealized_pnl,
                "unrealized_pnl_pct": p.unrealized_pnl_pct * 100,
                "stop_loss": p.stop_loss,
                "take_profit": p.take_profit,
                "cost_basis": p.cost_basis,
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
    """Get current watchlists."""
    if not _bot:
        return {"stocks": [], "crypto": []}

    return {
        "stocks": _bot._stock_watchlist,
        "crypto": _bot._crypto_watchlist,
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


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Swing Trader Dashboard</title>
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
        .card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
        }
        .card-title { color: #8b949e; font-size: 12px; margin-bottom: 5px; }
        .card-value { font-size: 24px; font-weight: 600; }
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
        .watchlist {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .symbol {
            background: #21262d;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 13px;
            font-family: monospace;
        }
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
        <h1>Swing Trader Dashboard</h1>

        <div class="grid" id="status-grid">
            <div class="card">
                <div class="card-title">Status</div>
                <div class="card-value" id="status">Loading...</div>
            </div>
            <div class="card">
                <div class="card-title">Equity</div>
                <div class="card-value" id="equity">--</div>
            </div>
            <div class="card">
                <div class="card-title">Buying Power</div>
                <div class="card-value" id="buying-power">--</div>
            </div>
            <div class="card">
                <div class="card-title">Total P&L</div>
                <div class="card-value" id="total-pnl">--</div>
            </div>
        </div>

        <h2>Positions</h2>
        <div class="card">
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Qty</th>
                        <th>Entry</th>
                        <th>Current</th>
                        <th>Value</th>
                        <th>P&L</th>
                        <th>Net if Sold</th>
                        <th>Stop</th>
                        <th>Target</th>
                    </tr>
                </thead>
                <tbody id="positions-table">
                    <tr><td colspan="9">Loading...</td></tr>
                </tbody>
            </table>
        </div>

        <h2>Stock Watchlist</h2>
        <div class="card">
            <div class="watchlist" id="stock-watchlist">Loading...</div>
        </div>

        <h2>Crypto Watchlist</h2>
        <div class="card">
            <div class="watchlist" id="crypto-watchlist">Loading...</div>
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

    <script>
        function formatCurrency(val) {
            return '$' + val.toFixed(2);
        }

        function formatPnl(val, pct) {
            const sign = val >= 0 ? '+' : '';
            const cls = val >= 0 ? 'positive' : 'negative';
            return `<span class="${cls}">${sign}${formatCurrency(val)} (${sign}${pct.toFixed(2)}%)</span>`;
        }

        async function fetchData() {
            try {
                const [status, positions, watchlists, jobs] = await Promise.all([
                    fetch('api/status').then(r => r.json()),
                    fetch('api/positions').then(r => r.json()),
                    fetch('api/watchlists').then(r => r.json()),
                    fetch('api/jobs').then(r => r.json()),
                ]);

                // Status
                const statusEl = document.getElementById('status');
                const dot = status.running ? 'running' : 'stopped';
                const mode = status.mode || 'unknown';
                statusEl.innerHTML = `<span class="status-dot ${dot}"></span>${status.running ? 'Running' : 'Stopped'} <span class="tag ${mode}">${mode.toUpperCase()}</span>`;

                document.getElementById('equity').textContent = formatCurrency(status.equity || 0);
                document.getElementById('buying-power').textContent = formatCurrency(status.buying_power || 0);
                document.getElementById('total-pnl').innerHTML = formatPnl(status.total_pnl || 0, status.total_pnl_pct || 0);

                // Positions
                const posTable = document.getElementById('positions-table');
                if (positions.length === 0) {
                    posTable.innerHTML = '<tr><td colspan="9" class="neutral">No open positions</td></tr>';
                } else {
                    posTable.innerHTML = positions.map(p => {
                        const netPnlPct = p.cost_basis > 0 ? (p.net_pnl / p.cost_basis * 100) : 0;
                        const feeNote = p.is_crypto ? ` <span class="neutral" style="font-size:10px">(${formatCurrency(p.fees)} fee)</span>` : '';
                        return `
                        <tr>
                            <td><strong>${p.symbol}</strong></td>
                            <td>${p.qty.toFixed(4)}</td>
                            <td>${formatCurrency(p.entry_price)}</td>
                            <td>${formatCurrency(p.current_price)}</td>
                            <td>${formatCurrency(p.market_value)}</td>
                            <td>${formatPnl(p.unrealized_pnl, p.unrealized_pnl_pct)}</td>
                            <td>${formatPnl(p.net_pnl, netPnlPct)}${feeNote}</td>
                            <td>${p.stop_loss ? formatCurrency(p.stop_loss) : '-'}</td>
                            <td>${p.take_profit ? formatCurrency(p.take_profit) : '-'}</td>
                        </tr>
                    `}).join('');
                }

                // Watchlists
                document.getElementById('stock-watchlist').innerHTML =
                    (watchlists.stocks || []).map(s => `<span class="symbol">${s}</span>`).join('') || 'None';
                document.getElementById('crypto-watchlist').innerHTML =
                    (watchlists.crypto || []).map(s => `<span class="symbol">${s}</span>`).join('') || 'None';

                // Jobs
                const jobsTable = document.getElementById('jobs-table');
                if (jobs.length === 0) {
                    jobsTable.innerHTML = '<tr><td colspan="2" class="neutral">No scheduled jobs</td></tr>';
                } else {
                    jobsTable.innerHTML = jobs.map(j => `
                        <tr>
                            <td>${j.name}</td>
                            <td>${j.next_run ? new Date(j.next_run).toLocaleString() : '-'}</td>
                        </tr>
                    `).join('');
                }

            } catch (e) {
                console.error('Fetch error:', e);
            }
        }

        fetchData();
        setInterval(fetchData, 30000);
    </script>
</body>
</html>
"""
