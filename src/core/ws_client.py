"""
Direct WebSocket client for Alpaca streaming APIs.

No alpaca-py SDK — uses the `websockets` library directly for full control
over connection management, reconnection, and message routing.

Endpoints:
- Market data: wss://stream.data.alpaca.markets/v2/{feed}
- Trade updates: wss://paper-api.alpaca.markets/stream (paper)
                 wss://api.alpaca.markets/stream (live)
- News: wss://stream.data.alpaca.markets/v1beta1/news

Protocol:
1. Connect → receive [{"T":"success","msg":"connected"}]
2. Auth    → send {"action":"auth","key":"...","secret":"..."}
             receive [{"T":"success","msg":"authenticated"}]
3. Subscribe → send {"action":"subscribe","bars":["AAPL"],...}
               receive [{"T":"subscription","bars":["AAPL"],...}]
4. Messages arrive as JSON arrays: [{"T":"b","S":"AAPL",...}, ...]
"""

import asyncio
import json
from datetime import datetime
from typing import Callable, Optional

import websockets
from loguru import logger


class AlpacaWSClient:
    """
    Direct WebSocket connection to Alpaca streaming APIs.

    Manages three independent connections:
    - data: real-time bars, quotes, trades
    - trades: order fills, cancels, rejects
    - news: real-time news articles

    Each connection auto-reconnects with exponential backoff.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        feed: str = "iex",
        paper: bool = True,
        reconnect_max_seconds: int = 30,
        heartbeat_seconds: int = 30,
    ):
        """
        Initialize WebSocket client.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            feed: Data feed ("iex" or "sip")
            paper: Use paper trading endpoint
            reconnect_max_seconds: Max backoff for reconnection
            heartbeat_seconds: Ping interval to detect dead connections
        """
        self._api_key = api_key
        self._secret_key = secret_key
        self._feed = feed
        self._paper = paper
        self._reconnect_max = reconnect_max_seconds
        self._heartbeat_interval = heartbeat_seconds

        # Endpoints
        self._data_url = f"wss://stream.data.alpaca.markets/v2/{feed}"
        if paper:
            self._trade_url = "wss://paper-api.alpaca.markets/stream"
        else:
            self._trade_url = "wss://api.alpaca.markets/stream"
        self._news_url = "wss://stream.data.alpaca.markets/v1beta1/news"

        # Connections
        self._data_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._trade_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._news_ws: Optional[websockets.WebSocketClientProtocol] = None

        # Connection state
        self._data_connected = False
        self._trade_connected = False
        self._news_connected = False
        self._data_authenticated = False
        self._trade_authenticated = False
        self._news_authenticated = False
        self._shutting_down = False

        # Current subscriptions (for resubscribe after reconnect)
        self._subscribed_bars: set[str] = set()
        self._subscribed_quotes: set[str] = set()
        self._subscribed_trades: set[str] = set()
        self._subscribed_news: set[str] = set()

        # Callbacks
        self._on_bar: Optional[Callable] = None
        self._on_quote: Optional[Callable] = None
        self._on_trade: Optional[Callable] = None
        self._on_trade_update: Optional[Callable] = None
        self._on_news: Optional[Callable] = None

    # ── Callback Registration ──────────────────────────────────────────

    def on_bar(self, callback: Callable) -> None:
        """Register callback for bar messages (T=b)."""
        self._on_bar = callback

    def on_quote(self, callback: Callable) -> None:
        """Register callback for quote messages (T=q)."""
        self._on_quote = callback

    def on_trade(self, callback: Callable) -> None:
        """Register callback for trade messages (T=t) — individual trades, not order updates."""
        self._on_trade = callback

    def on_trade_update(self, callback: Callable) -> None:
        """Register callback for trade update messages (order fills, cancels)."""
        self._on_trade_update = callback

    def on_news(self, callback: Callable) -> None:
        """Register callback for news messages (T=n)."""
        self._on_news = callback

    # ── Connection Lifecycle ───────────────────────────────────────────

    async def _connect_and_auth(
        self,
        url: str,
        name: str,
        is_trade_stream: bool = False,
    ) -> Optional[websockets.WebSocketClientProtocol]:
        """
        Connect to a WebSocket endpoint and authenticate.

        Args:
            url: WebSocket URL
            name: Connection name for logging
            is_trade_stream: Whether this is the trade updates stream
                             (uses different auth format)

        Returns:
            WebSocket connection, or None on failure
        """
        try:
            ws = await websockets.connect(
                url,
                ping_interval=self._heartbeat_interval,
                ping_timeout=self._heartbeat_interval,
                close_timeout=5,
            )

            # Authenticate
            auth_msg = {
                "action": "auth",
                "key": self._api_key,
                "secret": self._secret_key,
            }

            if is_trade_stream:
                # Trade updates stream: no initial "connected" message.
                # Send auth immediately after WebSocket handshake.
                await ws.send(json.dumps(auth_msg))
            else:
                # Data/news streams send [{"T":"success","msg":"connected"}] first
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                msgs = json.loads(raw)
                if isinstance(msgs, list):
                    for m in msgs:
                        if m.get("T") == "success" and m.get("msg") == "connected":
                            break
                    else:
                        logger.warning(f"[WS:{name}] Unexpected connect message: {msgs}")

                await ws.send(json.dumps(auth_msg))

            # Wait for auth response
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            auth_resp = json.loads(raw)

            if isinstance(auth_resp, list):
                auth_resp = auth_resp[0] if auth_resp else {}

            # Check auth success
            if is_trade_stream:
                # Trade stream: {"stream":"authorization","data":{"action":"authenticate","status":"authorized"}}
                if auth_resp.get("data", {}).get("status") == "authorized":
                    logger.info(f"[WS:{name}] Authenticated")
                elif auth_resp.get("T") == "success" and auth_resp.get("msg") == "authenticated":
                    logger.info(f"[WS:{name}] Authenticated")
                else:
                    logger.error(f"[WS:{name}] Auth failed: {auth_resp}")
                    await ws.close()
                    return None
            else:
                if auth_resp.get("T") == "success" and auth_resp.get("msg") == "authenticated":
                    logger.info(f"[WS:{name}] Authenticated")
                else:
                    logger.error(f"[WS:{name}] Auth failed: {auth_resp}")
                    await ws.close()
                    return None

            return ws

        except Exception as e:
            logger.error(f"[WS:{name}] Connection failed: {e}")
            return None

    async def connect_data(self) -> bool:
        """Connect to market data stream."""
        ws = await self._connect_and_auth(self._data_url, "DATA")
        if ws:
            self._data_ws = ws
            self._data_connected = True
            self._data_authenticated = True
            logger.info(f"[WS:DATA] Connected to {self._data_url}")
            return True
        return False

    async def connect_trades(self) -> bool:
        """Connect to trade updates stream."""
        ws = await self._connect_and_auth(
            self._trade_url, "TRADE", is_trade_stream=True
        )
        if ws:
            self._trade_ws = ws
            self._trade_connected = True
            self._trade_authenticated = True
            logger.info(f"[WS:TRADE] Connected to {self._trade_url}")

            # Subscribe to trade updates
            listen_msg = {
                "action": "listen",
                "data": {"streams": ["trade_updates"]},
            }
            await ws.send(json.dumps(listen_msg))
            logger.info("[WS:TRADE] Subscribed to trade_updates")

            return True
        return False

    async def connect_news(self) -> bool:
        """Connect to news stream."""
        ws = await self._connect_and_auth(self._news_url, "NEWS")
        if ws:
            self._news_ws = ws
            self._news_connected = True
            self._news_authenticated = True
            logger.info(f"[WS:NEWS] Connected to {self._news_url}")
            return True
        return False

    # ── Subscription Management ────────────────────────────────────────

    async def subscribe(
        self,
        bars: Optional[list[str]] = None,
        quotes: Optional[list[str]] = None,
        trades: Optional[list[str]] = None,
        news: Optional[list[str]] = None,
    ) -> None:
        """
        Subscribe to symbols on the data stream.

        Can be called multiple times to add symbols.
        """
        msg = {"action": "subscribe"}
        changed = False

        if bars:
            new_bars = set(bars) - self._subscribed_bars
            if new_bars:
                msg["bars"] = list(new_bars)
                self._subscribed_bars.update(new_bars)
                changed = True

        if quotes:
            new_quotes = set(quotes) - self._subscribed_quotes
            if new_quotes:
                msg["quotes"] = list(new_quotes)
                self._subscribed_quotes.update(new_quotes)
                changed = True

        if trades:
            new_trades = set(trades) - self._subscribed_trades
            if new_trades:
                msg["trades"] = list(new_trades)
                self._subscribed_trades.update(new_trades)
                changed = True

        if changed and self._data_ws and self._data_authenticated:
            await self._data_ws.send(json.dumps(msg))
            logger.info(
                f"[WS:DATA] Subscribed: "
                f"bars={list(self._subscribed_bars)}, "
                f"quotes={list(self._subscribed_quotes)}"
            )

        # News subscriptions go to the news connection
        if news:
            new_news = set(news) - self._subscribed_news
            if new_news and self._news_ws and self._news_authenticated:
                news_msg = {"action": "subscribe", "news": list(new_news)}
                await self._news_ws.send(json.dumps(news_msg))
                self._subscribed_news.update(new_news)
                logger.info(f"[WS:NEWS] Subscribed: {list(self._subscribed_news)}")

    async def unsubscribe(
        self,
        bars: Optional[list[str]] = None,
        quotes: Optional[list[str]] = None,
        trades: Optional[list[str]] = None,
        news: Optional[list[str]] = None,
    ) -> None:
        """Unsubscribe from symbols."""
        msg = {"action": "unsubscribe"}
        changed = False

        if bars:
            remove = set(bars) & self._subscribed_bars
            if remove:
                msg["bars"] = list(remove)
                self._subscribed_bars -= remove
                changed = True

        if quotes:
            remove = set(quotes) & self._subscribed_quotes
            if remove:
                msg["quotes"] = list(remove)
                self._subscribed_quotes -= remove
                changed = True

        if trades:
            remove = set(trades) & self._subscribed_trades
            if remove:
                msg["trades"] = list(remove)
                self._subscribed_trades -= remove
                changed = True

        if changed and self._data_ws and self._data_authenticated:
            await self._data_ws.send(json.dumps(msg))
            logger.info(f"[WS:DATA] Unsubscribed: bars={list(self._subscribed_bars)}")

        if news:
            remove = set(news) & self._subscribed_news
            if remove and self._news_ws and self._news_authenticated:
                news_msg = {"action": "unsubscribe", "news": list(remove)}
                await self._news_ws.send(json.dumps(news_msg))
                self._subscribed_news -= remove

    async def update_subscriptions(
        self,
        bars: Optional[list[str]] = None,
        quotes: Optional[list[str]] = None,
    ) -> None:
        """
        Replace current subscriptions with new set.

        Unsubscribes removed symbols and subscribes new ones.
        """
        new_bars = set(bars) if bars else set()
        new_quotes = set(quotes) if quotes else set()

        # Calculate diffs
        bars_to_add = new_bars - self._subscribed_bars
        bars_to_remove = self._subscribed_bars - new_bars
        quotes_to_add = new_quotes - self._subscribed_quotes
        quotes_to_remove = self._subscribed_quotes - new_quotes

        if bars_to_remove or quotes_to_remove:
            await self.unsubscribe(
                bars=list(bars_to_remove) if bars_to_remove else None,
                quotes=list(quotes_to_remove) if quotes_to_remove else None,
            )

        if bars_to_add or quotes_to_add:
            await self.subscribe(
                bars=list(bars_to_add) if bars_to_add else None,
                quotes=list(quotes_to_add) if quotes_to_add else None,
            )

    # ── Message Loops ──────────────────────────────────────────────────

    async def run_data_loop(self) -> None:
        """
        Run the data stream message loop with auto-reconnect.

        Routes messages to registered callbacks by type.
        """
        backoff = 1

        while not self._shutting_down:
            try:
                if not self._data_ws or not self._data_connected:
                    success = await self.connect_data()
                    if not success:
                        await asyncio.sleep(min(backoff, self._reconnect_max))
                        backoff *= 2
                        continue
                    # Resubscribe after reconnect
                    if self._subscribed_bars or self._subscribed_quotes:
                        resub = {"action": "subscribe"}
                        if self._subscribed_bars:
                            resub["bars"] = list(self._subscribed_bars)
                        if self._subscribed_quotes:
                            resub["quotes"] = list(self._subscribed_quotes)
                        if self._subscribed_trades:
                            resub["trades"] = list(self._subscribed_trades)
                        await self._data_ws.send(json.dumps(resub))
                        logger.info("[WS:DATA] Resubscribed after reconnect")
                    backoff = 1

                # Read messages
                async for raw in self._data_ws:
                    msgs = json.loads(raw)
                    if not isinstance(msgs, list):
                        msgs = [msgs]

                    for msg in msgs:
                        msg_type = msg.get("T")

                        if msg_type == "b" and self._on_bar:
                            try:
                                await self._on_bar(msg)
                            except Exception as e:
                                logger.error(f"[WS:DATA] Bar callback error: {e}")

                        elif msg_type == "q" and self._on_quote:
                            try:
                                await self._on_quote(msg)
                            except Exception as e:
                                logger.error(f"[WS:DATA] Quote callback error: {e}")

                        elif msg_type == "t" and self._on_trade:
                            try:
                                await self._on_trade(msg)
                            except Exception as e:
                                logger.error(f"[WS:DATA] Trade callback error: {e}")

                        elif msg_type == "subscription":
                            logger.debug(f"[WS:DATA] Subscription confirmed: {msg}")

                        elif msg_type == "error":
                            logger.error(f"[WS:DATA] Error: {msg}")

                        elif msg_type == "success":
                            logger.debug(f"[WS:DATA] Success: {msg}")

            except websockets.ConnectionClosed as e:
                logger.warning(f"[WS:DATA] Connection closed: {e}")
                self._data_connected = False
                self._data_authenticated = False
                self._data_ws = None
                await asyncio.sleep(min(backoff, self._reconnect_max))
                backoff *= 2

            except Exception as e:
                logger.error(f"[WS:DATA] Unexpected error: {e}")
                self._data_connected = False
                self._data_authenticated = False
                self._data_ws = None
                await asyncio.sleep(min(backoff, self._reconnect_max))
                backoff *= 2

    async def run_trade_loop(self) -> None:
        """
        Run the trade updates stream message loop with auto-reconnect.

        Handles order fill, cancel, reject notifications.
        """
        backoff = 1

        while not self._shutting_down:
            try:
                if not self._trade_ws or not self._trade_connected:
                    success = await self.connect_trades()
                    if not success:
                        await asyncio.sleep(min(backoff, self._reconnect_max))
                        backoff *= 2
                        continue
                    backoff = 1

                async for raw in self._trade_ws:
                    msg = json.loads(raw)

                    # Trade updates come as:
                    # {"stream":"trade_updates","data":{"event":"fill",...}}
                    if isinstance(msg, dict) and msg.get("stream") == "trade_updates":
                        data = msg.get("data", {})
                        if self._on_trade_update:
                            try:
                                await self._on_trade_update(data)
                            except Exception as e:
                                logger.error(f"[WS:TRADE] Callback error: {e}")
                    elif isinstance(msg, list):
                        # Some responses come as arrays
                        for m in msg:
                            if isinstance(m, dict) and m.get("T") == "success":
                                logger.debug(f"[WS:TRADE] {m}")

            except websockets.ConnectionClosed as e:
                logger.warning(f"[WS:TRADE] Connection closed: {e}")
                self._trade_connected = False
                self._trade_authenticated = False
                self._trade_ws = None
                await asyncio.sleep(min(backoff, self._reconnect_max))
                backoff *= 2

            except Exception as e:
                logger.error(f"[WS:TRADE] Unexpected error: {e}")
                self._trade_connected = False
                self._trade_authenticated = False
                self._trade_ws = None
                await asyncio.sleep(min(backoff, self._reconnect_max))
                backoff *= 2

    async def run_news_loop(self) -> None:
        """
        Run the news stream message loop with auto-reconnect.
        """
        backoff = 1

        while not self._shutting_down:
            try:
                if not self._news_ws or not self._news_connected:
                    success = await self.connect_news()
                    if not success:
                        await asyncio.sleep(min(backoff, self._reconnect_max))
                        backoff *= 2
                        continue
                    # Resubscribe
                    if self._subscribed_news:
                        resub = {
                            "action": "subscribe",
                            "news": list(self._subscribed_news),
                        }
                        await self._news_ws.send(json.dumps(resub))
                    backoff = 1

                async for raw in self._news_ws:
                    msgs = json.loads(raw)
                    if not isinstance(msgs, list):
                        msgs = [msgs]

                    for msg in msgs:
                        if msg.get("T") == "n" and self._on_news:
                            try:
                                await self._on_news(msg)
                            except Exception as e:
                                logger.error(f"[WS:NEWS] Callback error: {e}")

            except websockets.ConnectionClosed as e:
                logger.warning(f"[WS:NEWS] Connection closed: {e}")
                self._news_connected = False
                self._news_authenticated = False
                self._news_ws = None
                await asyncio.sleep(min(backoff, self._reconnect_max))
                backoff *= 2

            except Exception as e:
                logger.error(f"[WS:NEWS] Unexpected error: {e}")
                self._news_connected = False
                self._news_authenticated = False
                self._news_ws = None
                await asyncio.sleep(min(backoff, self._reconnect_max))
                backoff *= 2

    # ── Shutdown ───────────────────────────────────────────────────────

    async def disconnect(self) -> None:
        """Gracefully disconnect all WebSocket connections."""
        self._shutting_down = True

        for name, ws in [
            ("DATA", self._data_ws),
            ("TRADE", self._trade_ws),
            ("NEWS", self._news_ws),
        ]:
            if ws:
                try:
                    await ws.close()
                    logger.info(f"[WS:{name}] Disconnected")
                except Exception as e:
                    logger.debug(f"[WS:{name}] Error during disconnect: {e}")

        self._data_connected = False
        self._trade_connected = False
        self._news_connected = False

    # ── Status ─────────────────────────────────────────────────────────

    @property
    def data_connected(self) -> bool:
        """Whether market data stream is connected."""
        return self._data_connected and self._data_authenticated

    @property
    def trade_connected(self) -> bool:
        """Whether trade updates stream is connected."""
        return self._trade_connected and self._trade_authenticated

    @property
    def news_connected(self) -> bool:
        """Whether news stream is connected."""
        return self._news_connected and self._news_authenticated

    @property
    def subscribed_symbols(self) -> dict:
        """Currently subscribed symbols by type."""
        return {
            "bars": sorted(self._subscribed_bars),
            "quotes": sorted(self._subscribed_quotes),
            "trades": sorted(self._subscribed_trades),
            "news": sorted(self._subscribed_news),
        }

    def get_status(self) -> dict:
        """Get full connection status."""
        return {
            "data_connected": self.data_connected,
            "trade_connected": self.trade_connected,
            "news_connected": self.news_connected,
            "feed": self._feed,
            "paper": self._paper,
            "subscriptions": self.subscribed_symbols,
            "timestamp": datetime.now().isoformat(),
        }
