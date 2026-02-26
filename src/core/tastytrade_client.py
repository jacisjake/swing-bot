"""
Unified tastytrade API client.

Replaces AlpacaClient. Uses tastytrade REST API for account/orders/positions
and DXLink (via tastytrade SDK) for market data streaming and bar backfill.

Mode (paper/live) controlled by configuration:
  paper  → api.cert.tastyworks.com  (sandbox)
  live   → api.tastyworks.com       (production)
"""

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time as dt_time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from loguru import logger

from config import settings

# Persistent token storage path (survives container restarts via volume mount)
TOKEN_FILE = Path("data/oauth_token.json")


# NYSE holidays — schedule-based market clock (no broker API needed)
NYSE_HOLIDAYS: set[date] = {
    # 2024
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19),
    date(2024, 3, 29), date(2024, 5, 27), date(2024, 6, 19),
    date(2024, 7, 4), date(2024, 9, 2), date(2024, 11, 28),
    date(2024, 12, 25),
    # 2025
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19),
    date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 27),
    date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15),
    date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18),
    date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25),
    date(2027, 12, 24),
}

# Thread pool for running async DXLink calls from sync context
_thread_pool = ThreadPoolExecutor(max_workers=2)


class TastytradeClient:
    """
    Unified tastytrade client for all trading operations.

    API-compatible drop-in replacement for AlpacaClient:
    same method signatures, same return formats.
    """

    PROD_URL = "https://api.tastyworks.com"
    SANDBOX_URL = "https://api.cert.tastyworks.com"
    TOKEN_REFRESH_SECONDS = 780  # 13 min (token expires at 15)

    def __init__(self):
        """Initialize tastytrade client based on settings."""
        self._base_url = self.SANDBOX_URL if settings.is_paper else self.PROD_URL
        self._acct = settings.tt_account_number
        self._http = requests.Session()
        self._token: Optional[str] = None
        self._token_time: float = 0.0
        self._authenticated = False

        # OAuth state
        self._use_oauth = settings.has_oauth or bool(settings.tt_client_secret)
        self._refresh_token: Optional[str] = settings.tt_refresh_token

        # Try loading refresh token from persistent file if not in env
        if self._use_oauth and not self._refresh_token:
            self._refresh_token = self._load_refresh_token()

        # tastytrade SDK session (for DXLink streaming in get_bars)
        self._sdk_session = None

        # Bar cache: (symbol, timeframe) -> (timestamp, DataFrame)
        # Prevents redundant DXLink connections during scan cycles
        self._bar_cache: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
        self._bar_cache_ttl = 60  # seconds

        if self._use_oauth and self._refresh_token:
            self._login_oauth()
        elif settings.has_legacy_auth:
            self._login_legacy()
        else:
            logger.warning(
                "No tastytrade credentials configured. "
                "Complete OAuth setup via dashboard or set TT_REFRESH_TOKEN in .env"
            )

        mode = "PAPER" if settings.is_paper else "LIVE"
        auth = "OAuth" if self._use_oauth else "session"
        status = "authenticated" if self._authenticated else "NOT AUTHENTICATED"
        logger.info(f"TastytradeClient initialized: {mode} mode, {auth} auth, {status}")

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    # =========================================================================
    # Authentication
    # =========================================================================

    def _login(self) -> None:
        """Authenticate using the configured method."""
        if self._use_oauth and self._refresh_token:
            self._login_oauth()
        elif settings.has_legacy_auth:
            self._login_legacy()

    def _login_legacy(self) -> None:
        """Authenticate with username/password session token."""
        resp = self._http.post(
            f"{self._base_url}/sessions",
            json={
                "login": settings.tt_username,
                "password": settings.tt_password,
            },
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        self._token = data["session-token"]
        self._http.headers["Authorization"] = self._token
        self._token_time = time.time()
        self._authenticated = True
        logger.debug("tastytrade session authenticated (legacy)")

    def _login_oauth(self) -> None:
        """Authenticate using OAuth2 refresh token grant."""
        payload = {
            "grant_type": "refresh_token",
            "client_secret": settings.tt_client_secret,
            "refresh_token": self._refresh_token,
        }
        # Include client_id if available
        if settings.tt_client_id:
            payload["client_id"] = settings.tt_client_id

        resp = requests.post(
            f"{self._base_url}/oauth/token",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

        self._token = data["access_token"]
        self._http.headers["Authorization"] = f"Bearer {self._token}"
        self._token_time = time.time()
        self._authenticated = True

        # Update refresh token if a new one was issued
        new_refresh = data.get("refresh_token")
        if new_refresh:
            self._refresh_token = new_refresh
            self._save_refresh_token()

        logger.debug("tastytrade session authenticated (OAuth)")

    def set_refresh_token(self, refresh_token: str) -> None:
        """Set a new refresh token (from OAuth callback) and authenticate."""
        self._refresh_token = refresh_token
        self._use_oauth = True
        self._save_refresh_token()
        # Reset SDK session so it picks up new creds
        self._sdk_session = None
        self._login_oauth()

    def _ensure_token(self) -> None:
        """Refresh session token if close to expiry."""
        if time.time() - self._token_time > self.TOKEN_REFRESH_SECONDS:
            self._login()

    def _load_refresh_token(self) -> Optional[str]:
        """Load refresh token from persistent file."""
        try:
            if TOKEN_FILE.exists():
                data = json.loads(TOKEN_FILE.read_text())
                token = data.get("refresh_token")
                if token:
                    logger.debug("Loaded refresh token from file")
                    return token
        except Exception as e:
            logger.warning(f"Failed to load refresh token: {e}")
        return None

    def _save_refresh_token(self) -> None:
        """Persist refresh token to file."""
        try:
            TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
            TOKEN_FILE.write_text(json.dumps({
                "refresh_token": self._refresh_token,
                "updated_at": datetime.now().isoformat(),
            }))
            logger.debug("Saved refresh token to file")
        except Exception as e:
            logger.warning(f"Failed to save refresh token: {e}")

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """Authenticated GET request with 401 retry."""
        self._ensure_token()
        resp = self._http.get(f"{self._base_url}{path}", params=params)
        if resp.status_code == 401:
            self._login()
            resp = self._http.get(f"{self._base_url}{path}", params=params)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json_data: Optional[dict] = None) -> dict:
        """Authenticated POST request with 401 retry."""
        self._ensure_token()
        resp = self._http.post(
            f"{self._base_url}{path}",
            json=json_data,
            headers={"Content-Type": "application/json"},
        )
        if resp.status_code == 401:
            self._login()
            resp = self._http.post(
                f"{self._base_url}{path}",
                json=json_data,
                headers={"Content-Type": "application/json"},
            )
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> bool:
        """Authenticated DELETE request with 401 retry."""
        self._ensure_token()
        resp = self._http.delete(f"{self._base_url}{path}")
        if resp.status_code == 401:
            self._login()
            resp = self._http.delete(f"{self._base_url}{path}")
        return resp.status_code in (200, 204)

    def _get_sdk_session(self):
        """Get or create tastytrade SDK session for DXLink operations."""
        if self._sdk_session is None:
            from tastytrade import Session
            if self._use_oauth and self._refresh_token:
                # OAuth: Session(provider_secret, refresh_token)
                self._sdk_session = Session(
                    settings.tt_client_secret,
                    self._refresh_token,
                    is_test=settings.is_paper,
                )
            else:
                # Legacy: Session(username, password)
                self._sdk_session = Session(
                    settings.tt_username,
                    settings.tt_password,
                    is_test=settings.is_paper,
                )
        return self._sdk_session

    # =========================================================================
    # Account Methods
    # =========================================================================

    def get_account(self) -> dict:
        """Get account information including equity and buying power."""
        data = self._get(f"/accounts/{self._acct}/balances")["data"]
        return {
            "equity": float(data.get("net-liquidating-value", 0)),
            "buying_power": float(data.get("equity-buying-power", 0)),
            "cash": float(data.get("cash-balance", 0)),
            "portfolio_value": float(data.get("net-liquidating-value", 0)),
            "pattern_day_trader": False,
            "daytrade_count": 0,  # Cash account, no PDT tracking
            "trading_blocked": False,
            "account_blocked": False,
        }

    def get_buying_power(self) -> float:
        """Get current buying power."""
        return float(self.get_account()["buying_power"])

    def get_equity(self) -> float:
        """Get current account equity."""
        return float(self.get_account()["equity"])

    # =========================================================================
    # Position Methods
    # =========================================================================

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        data = self._get(f"/accounts/{self._acct}/positions")["data"]
        items = data.get("items", []) if isinstance(data, dict) else data

        positions = []
        for pos in items:
            qty_raw = float(pos.get("quantity", 0))
            direction = pos.get("quantity-direction", "Long")
            qty = qty_raw if direction == "Long" else -qty_raw

            avg_price = float(pos.get("average-open-price", 0))
            close_price = float(pos.get("close-price", 0) or avg_price)
            abs_qty = abs(qty)
            cost_basis = abs_qty * avg_price
            market_value = abs_qty * close_price

            if qty >= 0:
                unrealized_pl = market_value - cost_basis
            else:
                unrealized_pl = cost_basis - market_value

            unrealized_plpc = (unrealized_pl / cost_basis) if cost_basis > 0 else 0

            positions.append({
                "symbol": pos.get("symbol", pos.get("underlying-symbol", "")),
                "qty": abs_qty,
                "side": "long" if qty >= 0 else "short",
                "market_value": market_value,
                "cost_basis": cost_basis,
                "unrealized_pl": unrealized_pl,
                "unrealized_plpc": unrealized_plpc,
                "current_price": close_price,
                "avg_entry_price": avg_price,
                "asset_class": pos.get("instrument-type", "Equity").lower(),
            })

        return positions

    def get_position(self, symbol: str) -> Optional[dict]:
        """Get position for a specific symbol, or None if not held."""
        for p in self.get_positions():
            if p["symbol"] == symbol:
                return p
        return None

    def has_position(self, symbol: str) -> bool:
        """Check if we have an open position in a symbol."""
        return self.get_position(symbol) is not None

    # =========================================================================
    # Market Data Methods
    # =========================================================================

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        limit: int = 100,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get historical bars for a symbol via yfinance.

        Args:
            symbol: Stock ticker (e.g., "AAPL")
            timeframe: "1Min", "5Min", "15Min", "1Hour", "1Day"
            limit: Number of bars to retrieve
            end: End datetime (defaults to now)

        Returns:
            DataFrame with columns: open, high, low, close, volume, vwap
        """
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "vwap"]
        )

        # Check cache first
        cache_key = (symbol, timeframe)
        cached = self._bar_cache.get(cache_key)
        if cached and end is None:
            cache_time, cache_df = cached
            if time.time() - cache_time < self._bar_cache_ttl and len(cache_df) >= limit:
                return cache_df.tail(limit)

        try:
            df = self._fetch_bars_yfinance(symbol, timeframe, limit, end)
            if not df.empty and end is None:
                self._bar_cache[cache_key] = (time.time(), df)
            return df
        except Exception as e:
            logger.error(f"get_bars failed for {symbol}: {e}")
            return empty

    def _fetch_bars_yfinance(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        end: Optional[datetime],
    ) -> pd.DataFrame:
        """Fetch historical bars from yfinance (no WebSocket needed)."""
        import yfinance as yf

        # Map timeframe to yfinance interval + period
        tf_lower = timeframe.lower().replace(" ", "")
        yf_intervals = {
            "1min": ("1m", "1d"),     # 1-min: max 7 days
            "5min": ("5m", "5d"),     # 5-min: max 60 days
            "15min": ("15m", "5d"),
            "30min": ("30m", "5d"),
            "1hour": ("1h", "30d"),
            "4hour": ("1h", "60d"),   # yf has no 4h; fetch 1h, resample later
            "1day": ("1d", "6mo"),
            "1week": ("1wk", "2y"),
        }

        if tf_lower not in yf_intervals:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        interval, period = yf_intervals[tf_lower]

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "vwap"]
            )

        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Keep only OHLCV columns, add vwap placeholder
        cols = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]]
        if "vwap" not in df.columns:
            df["vwap"] = 0.0

        # Ensure UTC timezone on index
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # Filter by end date if specified
        if end:
            end_utc = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
            df = df[df.index <= end_utc]

        return df.tail(limit)

    def _create_sdk_session(self):
        """Create a fresh SDK session (not cached).

        Used by get_bars() which runs in a thread pool via asyncio.run().
        Each asyncio.run() creates and closes its own event loop, so the
        session must be fresh — a cached session's HTTP client would be
        bound to a closed loop from a previous asyncio.run() call.
        """
        from tastytrade import Session
        if self._use_oauth and self._refresh_token:
            return Session(
                settings.tt_client_secret,
                self._refresh_token,
                is_test=settings.is_paper,
            )
        else:
            return Session(
                settings.tt_username,
                settings.tt_password,
                is_test=settings.is_paper,
            )

    async def _fetch_candles_async(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        end: Optional[datetime],
    ) -> pd.DataFrame:
        """Async DXLink candle fetch."""
        from tastytrade import DXLinkStreamer
        from tastytrade.dxfeed import Candle

        period = self._map_timeframe_to_dxlink(timeframe)

        end_dt = end or datetime.now(timezone.utc)
        start_dt = self._calculate_start_time(timeframe, limit, end_dt)

        # Fresh session each call — asyncio.run() closes its event loop,
        # so a cached session's HTTP client would be on a dead loop.
        session = self._create_sdk_session()
        candles: list[dict] = []

        async with DXLinkStreamer(session) as streamer:
            await streamer.subscribe_candle(
                symbols=[symbol],
                interval=period,
                start_time=start_dt,
            )

            deadline = asyncio.get_event_loop().time() + 15  # 15s timeout
            async for candle in streamer.listen(Candle):
                candles.append({
                    "open": float(candle.open) if candle.open else 0,
                    "high": float(candle.high) if candle.high else 0,
                    "low": float(candle.low) if candle.low else 0,
                    "close": float(candle.close) if candle.close else 0,
                    "volume": int(candle.volume) if candle.volume else 0,
                    "vwap": float(candle.vwap) if hasattr(candle, "vwap") and candle.vwap else 0,
                    "timestamp": candle.time if hasattr(candle, "time") else end_dt,
                })
                if len(candles) >= limit or asyncio.get_event_loop().time() > deadline:
                    break

        if not candles:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "vwap"]
            )

        df = pd.DataFrame(candles)
        df.index = pd.to_datetime(df.pop("timestamp"), utc=True)
        df.sort_index(inplace=True)
        # Remove rows with all zeros (padding from DXLink)
        df = df[(df["close"] > 0)]
        return df.tail(limit)

    def _calculate_start_time(
        self, timeframe: str, limit: int, end: datetime
    ) -> datetime:
        """Calculate start time for bar request based on timeframe and limit."""
        tf_lower = timeframe.lower().replace(" ", "")
        multipliers = {
            "1min": timedelta(minutes=1),
            "5min": timedelta(minutes=5),
            "15min": timedelta(minutes=15),
            "30min": timedelta(minutes=30),
            "1hour": timedelta(hours=1),
            "4hour": timedelta(hours=4),
            "1day": timedelta(days=1),
            "1week": timedelta(weeks=1),
        }
        delta = multipliers.get(tf_lower, timedelta(days=1))
        # Fetch extra to account for non-trading hours
        return end - (delta * limit * 2)

    def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.

        Uses the equity snapshot endpoint for a fast, single REST call.
        Falls back to recent bars if that fails.
        """
        # Try equity snapshot (single REST call, no DXLink)
        try:
            data = self._get(
                f"/market-data/stocks/quotes/{symbol}"
            )["data"]
            # Try last trade price first, then midpoint of bid/ask
            last = data.get("last", 0) or data.get("lastTrade", 0)
            if last and float(last) > 0:
                return float(last)
            bid = float(data.get("bid", 0) or 0)
            ask = float(data.get("ask", 0) or 0)
            if bid > 0 and ask > 0:
                return round((bid + ask) / 2, 4)
            if ask > 0:
                return float(ask)
        except Exception:
            pass

        # Fallback: get from most recent bar (slower — DXLink backfill)
        bars = self.get_bars(symbol, timeframe="5Min", limit=1)
        if not bars.empty:
            return float(bars["close"].iloc[-1])

        raise ValueError(f"No price data for {symbol}")

    def get_latest_quotes_with_change(self, symbols: list[str]) -> dict:
        """Get the latest price and daily change for a list of symbols."""
        if not symbols:
            return {}

        results = {}
        for symbol in symbols:
            try:
                bars = self.get_bars(symbol, timeframe="1Day", limit=2)
                price = 0.0
                change = 0.0
                if len(bars) >= 2:
                    price = float(bars["close"].iloc[-1])
                    prev_close = float(bars["close"].iloc[-2])
                    change = price - prev_close
                elif len(bars) == 1:
                    price = float(bars["close"].iloc[-1])

                results[symbol] = {"price": price, "change": change}
            except Exception as e:
                logger.debug(f"Quote fetch failed for {symbol}: {e}")
                results[symbol] = {"price": 0.0, "change": 0.0}

        return results

    def get_news(
        self,
        symbol: str,
        limit: int = 5,
        hours_back: int = 12,
    ) -> list[dict]:
        """
        Get recent news articles for a symbol.

        tastytrade has no news API — always returns empty list.
        News catalysts come from the press release scanner instead.
        """
        return []

    @staticmethod
    def _map_timeframe_to_dxlink(timeframe: str) -> str:
        """Map timeframe string to DXLink candle period."""
        mapping = {
            "1min": "1m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "1hour": "1h",
            "4hour": "4h",
            "1day": "1d",
            "1week": "1w",
        }
        tf_lower = timeframe.lower().replace(" ", "")
        result = mapping.get(tf_lower)
        if result is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        return result

    def get_multi_timeframe_bars(
        self,
        symbol: str,
        timeframes: list[str],
        limit: int = 100,
        end: Optional[datetime] = None,
    ) -> dict[str, pd.DataFrame]:
        """Get historical bars for multiple timeframes at once."""
        results = {}
        for tf in timeframes:
            try:
                bars = self.get_bars(symbol, timeframe=tf, limit=limit, end=end)
                results[tf] = bars
            except Exception as e:
                logger.warning(f"Failed to fetch {tf} bars for {symbol}: {e}")
                results[tf] = pd.DataFrame()
        return results

    # =========================================================================
    # Order Methods
    # =========================================================================

    @staticmethod
    def _map_side_to_action(side: str, opening: bool = True) -> str:
        """Map buy/sell to tastytrade order actions."""
        if side.lower() == "buy":
            return "Buy to Open" if opening else "Buy to Close"
        else:
            return "Sell to Open" if opening else "Sell to Close"

    def _build_equity_leg(self, symbol: str, qty: float, action: str) -> dict:
        """Build an equity order leg in tastytrade format."""
        return {
            "instrument-type": "Equity",
            "symbol": symbol,
            "quantity": str(Decimal(str(qty))),
            "action": action,
        }

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
    ) -> dict:
        """Submit a market order."""
        is_sell = side.lower() == "sell"
        action = "Sell to Close" if is_sell else "Buy to Open"
        leg = self._build_equity_leg(symbol, qty, action)

        order_payload = {
            "time-in-force": "Day",
            "order-type": "Market",
            "legs": [leg],
        }

        data = self._post(
            f"/accounts/{self._acct}/orders", json_data=order_payload
        )["data"]
        order = data.get("order", data)
        logger.info(f"Market order submitted: {side} {qty} {symbol}")
        return self._order_to_dict(order)

    def submit_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        extended_hours: bool = False,
    ) -> dict:
        """Submit a limit order."""
        is_sell = side.lower() == "sell"
        action = "Sell to Close" if is_sell else "Buy to Open"
        leg = self._build_equity_leg(symbol, qty, action)

        # tastytrade: negative price = debit (buying), positive = credit (selling)
        price_value = -round(limit_price, 2) if not is_sell else round(limit_price, 2)

        order_payload = {
            "time-in-force": "Day",
            "order-type": "Limit",
            "price": str(Decimal(str(price_value))),
            "legs": [leg],
        }

        data = self._post(
            f"/accounts/{self._acct}/orders", json_data=order_payload
        )["data"]
        order = data.get("order", data)
        logger.info(f"Limit order submitted: {side} {qty} {symbol} @ ${limit_price}")
        return self._order_to_dict(order)

    def submit_stop_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_price: float,
        limit_price: float,
    ) -> dict:
        """Submit a stop-limit order (typically for stop-losses)."""
        is_sell = side.lower() == "sell"
        action = "Sell to Close" if is_sell else "Buy to Open"
        leg = self._build_equity_leg(symbol, qty, action)

        price_value = -round(limit_price, 2) if not is_sell else round(limit_price, 2)

        order_payload = {
            "time-in-force": "Day",
            "order-type": "Stop Limit",
            "stop-trigger": str(Decimal(str(round(stop_price, 2)))),
            "price": str(Decimal(str(price_value))),
            "legs": [leg],
        }

        data = self._post(
            f"/accounts/{self._acct}/orders", json_data=order_payload
        )["data"]
        order = data.get("order", data)
        logger.info(
            f"Stop-limit order submitted: {side} {qty} {symbol} "
            f"stop=${stop_price} limit=${limit_price}"
        )
        return self._order_to_dict(order)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        try:
            success = self._delete(
                f"/accounts/{self._acct}/orders/{order_id}"
            )
            if success:
                logger.info(f"Order cancelled: {order_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        orders = self.get_orders(status="open")
        count = 0
        for order in orders:
            if self.cancel_order(order["id"]):
                count += 1
        logger.info(f"Cancelled {count} orders")
        return count

    def get_orders(self, status: str = "open") -> list[dict]:
        """Get orders by status (open, closed, all)."""
        if status == "open":
            data = self._get(f"/accounts/{self._acct}/orders/live")["data"]
        else:
            data = self._get(f"/accounts/{self._acct}/orders")["data"]

        items = data.get("items", []) if isinstance(data, dict) else data
        return [self._order_to_dict(o) for o in items]

    def _order_to_dict(self, order: dict) -> dict:
        """Convert tastytrade order response to standard dict format."""
        legs = order.get("legs", [])
        first_leg = legs[0] if legs else {}
        fills = first_leg.get("fills", [])

        # Calculate filled qty and avg price from fills
        filled_qty = sum(float(f.get("quantity", 0)) for f in fills)
        filled_value = sum(
            float(f.get("fill-price", 0)) * float(f.get("quantity", 0))
            for f in fills
        )
        filled_avg_price = (filled_value / filled_qty) if filled_qty > 0 else None

        # Map tastytrade status to standard status
        raw_status = order.get("status", "").lower().replace(" ", "_")
        status_map = {
            "received": "new",
            "routed": "accepted",
            "in_flight": "accepted",
            "live": "new",
            "filled": "filled",
            "cancelled": "cancelled",
            "expired": "expired",
            "rejected": "rejected",
            "partially_filled": "partially_filled",
            "contingent": "new",
        }
        mapped_status = status_map.get(raw_status, raw_status)

        # Map side from action
        action = first_leg.get("action", "")
        side = "buy" if "Buy" in action else "sell"

        return {
            "id": str(order.get("id", "")),
            "symbol": first_leg.get("symbol", order.get("underlying-symbol", "")),
            "qty": float(first_leg.get("quantity", 0)),
            "filled_qty": filled_qty,
            "side": side,
            "type": order.get("order-type", "").lower().replace(" ", "_"),
            "status": mapped_status,
            "limit_price": (
                abs(float(order["price"])) if order.get("price") else None
            ),
            "stop_price": (
                float(order["stop-trigger"]) if order.get("stop-trigger") else None
            ),
            "filled_avg_price": filled_avg_price,
            "created_at": order.get("received-at", order.get("updated-at")),
            "submitted_at": order.get("received-at"),
        }

    # =========================================================================
    # Asset Methods
    # =========================================================================

    def get_asset(self, symbol: str) -> dict:
        """Get asset information including name, exchange, etc."""
        try:
            data = self._get(f"/instruments/equities/{symbol}")["data"]
            return {
                "symbol": data.get("symbol", symbol),
                "name": data.get("description", symbol),
                "exchange": data.get("listed-market", ""),
                "class": "us_equity",
                "tradable": data.get("is-tradeable", True),
                "fractionable": data.get(
                    "is-fractional-quantity-eligible", False
                ),
                "status": "active" if data.get("is-tradeable") else "inactive",
            }
        except Exception as e:
            return {"symbol": symbol, "name": symbol, "error": str(e)}

    def is_fractionable(self, symbol: str) -> bool:
        """Check if an asset supports fractional shares."""
        try:
            asset = self.get_asset(symbol)
            return asset.get("fractionable", False)
        except Exception:
            return False

    def is_market_open(self) -> bool:
        """
        Check if the stock market is currently open.

        Schedule-based using NYSE hours + static holiday list.
        No broker API call needed.
        """
        import pytz

        et = pytz.timezone("America/New_York")
        now_et = datetime.now(et)

        # Weekend
        if now_et.weekday() >= 5:
            return False

        # Holiday
        if now_et.date() in NYSE_HOLIDAYS:
            return False

        # Regular market hours: 9:30 AM - 4:00 PM ET
        current_time = now_et.time()
        return dt_time(9, 30) <= current_time < dt_time(16, 0)

    # =========================================================================
    # Trade Stats
    # =========================================================================

    def get_trade_stats(self, since: str = "2026-02-24") -> dict:
        """
        Calculate trading stats from tastytrade transaction history.

        Matches buy/sell transactions by symbol to calculate realized P&L.

        Args:
            since: Only include transactions on or after this date (YYYY-MM-DD).
                   Defaults to 2026-02-24 (experiment tracking start).
        """
        try:
            params = {"per-page": 500}
            if since:
                params["start-date"] = since
            data = self._get(
                f"/accounts/{self._acct}/transactions",
                params=params,
            )["data"]
            items = data.get("items", []) if isinstance(data, dict) else data
        except Exception as e:
            logger.error(f"Failed to get transactions: {e}")
            return self._empty_stats()

        # Group filled equity transactions by symbol
        symbol_orders: dict[str, list] = {}
        for txn in items:
            if txn.get("instrument-type") != "Equity":
                continue
            if txn.get("transaction-type") not in ("Trade",):
                continue

            symbol = txn.get("symbol", txn.get("underlying-symbol", ""))
            action = txn.get("action", "")
            qty = abs(float(txn.get("quantity", 0)))
            price = abs(float(txn.get("price", 0)))
            txn_time = txn.get("executed-at", txn.get("transaction-date", ""))

            # Client-side date filter as safety net
            if since and txn_time and txn_time[:10] < since:
                continue

            if qty <= 0 or price <= 0:
                continue

            side = "buy" if "Buy" in action else "sell"

            if symbol not in symbol_orders:
                symbol_orders[symbol] = []
            symbol_orders[symbol].append({
                "side": side,
                "qty": qty,
                "price": price,
                "time": txn_time,
            })

        # Calculate P&L for completed round trips
        completed_trades = []
        total_realized_pnl = 0.0

        for symbol, orders_list in symbol_orders.items():
            orders_list.sort(key=lambda x: x["time"])

            position_qty = 0.0
            cost_basis = 0.0

            for order in orders_list:
                if order["side"] == "buy":
                    cost_basis += order["qty"] * order["price"]
                    position_qty += order["qty"]
                elif order["side"] == "sell" and position_qty > 0:
                    sell_qty = min(order["qty"], position_qty)
                    avg_cost = cost_basis / position_qty if position_qty > 0 else 0
                    sell_proceeds = sell_qty * order["price"]
                    sell_cost = sell_qty * avg_cost

                    pnl = sell_proceeds - sell_cost
                    total_realized_pnl += pnl

                    completed_trades.append({
                        "symbol": symbol,
                        "qty": sell_qty,
                        "entry_price": avg_cost,
                        "exit_price": order["price"],
                        "pnl": pnl,
                        "pnl_pct": (pnl / sell_cost * 100) if sell_cost > 0 else 0,
                        "exit_time": order["time"],
                    })

                    cost_basis -= sell_qty * avg_cost
                    position_qty -= sell_qty

        wins = [t for t in completed_trades if t["pnl"] > 0]
        losses = [t for t in completed_trades if t["pnl"] < 0]

        return {
            "total_trades": len(completed_trades),
            "total_realized_pnl": total_realized_pnl,
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": (
                (len(wins) / len(completed_trades) * 100)
                if completed_trades
                else 0.0
            ),
            "avg_win": (
                sum(t["pnl"] for t in wins) / len(wins) if wins else 0.0
            ),
            "avg_loss": (
                sum(t["pnl"] for t in losses) / len(losses) if losses else 0.0
            ),
            "largest_win": max((t["pnl"] for t in wins), default=0.0),
            "largest_loss": min((t["pnl"] for t in losses), default=0.0),
            "trades": completed_trades[-20:],
        }

    @staticmethod
    def _empty_stats() -> dict:
        """Return empty trade stats."""
        return {
            "total_trades": 0,
            "total_realized_pnl": 0.0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "trades": [],
        }
