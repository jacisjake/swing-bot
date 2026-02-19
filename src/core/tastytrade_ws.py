"""
DXLink streaming client + order fill polling for tastytrade.

Replaces AlpacaWSClient with:
- DXLink (via tastytrade SDK) for real-time 5-min candles and quotes
- REST polling for order fill notifications (only when pending orders exist)
- No news stream (tastytrade has no news API)

DXLink sends native 5-min candles — no 1-min→5-min aggregation needed.
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from loguru import logger

from config.settings import get_settings


class TastytradeWSClient:
    """
    Streaming client for tastytrade market data and order updates.

    Manages two logical connections:
    - data: DXLink WebSocket for real-time bars (5-min candles) and quotes
    - trades: REST polling for order fills (every 5s when orders are pending)

    Each auto-reconnects on failure.
    """

    def __init__(
        self,
        tastytrade_client,
        reconnect_max_seconds: int = 30,
        heartbeat_seconds: int = 30,
    ):
        """
        Initialize streaming client.

        Args:
            tastytrade_client: TastytradeClient instance (for SDK session + REST)
            reconnect_max_seconds: Max backoff for reconnection
            heartbeat_seconds: Health check interval
        """
        self._client = tastytrade_client
        self._reconnect_max = reconnect_max_seconds
        self._heartbeat_interval = heartbeat_seconds

        # DXLink streamer (from tastytrade SDK)
        self._streamer = None
        self._streamer_task: Optional[asyncio.Task] = None

        # Connection state
        self._data_connected = False
        self._trade_connected = False
        self._shutting_down = False

        # Current subscriptions
        self._subscribed_bars: set[str] = set()
        self._subscribed_quotes: set[str] = set()

        # Callbacks
        self._on_bar: Optional[Callable] = None
        self._on_quote: Optional[Callable] = None
        self._on_trade_update: Optional[Callable] = None

        # Order polling state
        self._poll_orders = False
        self._last_order_ids: set[str] = set()
        self._poll_interval = 5  # seconds

        # Event loop reference for thread bridging
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── Callback Registration ──────────────────────────────────────────

    def on_bar(self, callback: Callable) -> None:
        """Register callback for bar messages."""
        self._on_bar = callback

    def on_quote(self, callback: Callable) -> None:
        """Register callback for quote messages."""
        self._on_quote = callback

    def on_trade_update(self, callback: Callable) -> None:
        """Register callback for trade update messages (order fills)."""
        self._on_trade_update = callback

    # ── Connection Lifecycle ───────────────────────────────────────────

    async def connect_data(self) -> bool:
        """Connect to DXLink market data stream."""
        try:
            from tastytrade import DXLinkStreamer
            from tastytrade.dxfeed import Candle, Quote

            self._loop = asyncio.get_running_loop()
            session = self._client._get_sdk_session()
            self._streamer = DXLinkStreamer(session)
            await self._streamer.__aenter__()
            self._data_connected = True
            logger.info("[WS:DATA] DXLink connected")
            return True
        except Exception as e:
            logger.error(f"[WS:DATA] DXLink connection failed: {e}")
            self._data_connected = False
            return False

    async def connect_trades(self) -> bool:
        """Start order fill polling."""
        self._trade_connected = True
        self._poll_orders = True
        logger.info("[WS:TRADE] Order polling enabled")
        return True

    # ── Subscription Management ────────────────────────────────────────

    async def subscribe(
        self,
        bars: Optional[list[str]] = None,
        quotes: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """
        Subscribe to symbols for bars and/or quotes.

        Bars use DXLink Candle events with 5-min period.
        Quotes use DXLink Quote events.
        """
        if not self._streamer or not self._data_connected:
            logger.warning("[WS:DATA] Cannot subscribe — not connected")
            return

        from tastytrade.dxfeed import Quote

        if bars:
            new_bars = set(bars) - self._subscribed_bars
            if new_bars:
                try:
                    # subscribe_candle requires a start_time; use 1 hour ago
                    # to get a small backfill of recent candles
                    start = datetime.now(timezone.utc) - timedelta(hours=1)
                    await self._streamer.subscribe_candle(
                        list(new_bars), interval="5m", start_time=start
                    )
                    self._subscribed_bars.update(new_bars)
                except Exception as e:
                    logger.error(f"[WS:DATA] Candle subscribe failed: {e}")

        if quotes:
            new_quotes = set(quotes) - self._subscribed_quotes
            if new_quotes:
                try:
                    await self._streamer.subscribe(Quote, list(new_quotes))
                    self._subscribed_quotes.update(new_quotes)
                except Exception as e:
                    logger.error(f"[WS:DATA] Quote subscribe failed: {e}")

        if bars or quotes:
            logger.info(
                f"[WS:DATA] Subscribed: "
                f"bars={sorted(self._subscribed_bars)}, "
                f"quotes={sorted(self._subscribed_quotes)}"
            )

    async def unsubscribe(
        self,
        bars: Optional[list[str]] = None,
        quotes: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """Unsubscribe from symbols."""
        if not self._streamer or not self._data_connected:
            return

        from tastytrade.dxfeed import Quote

        if bars:
            remove = set(bars) & self._subscribed_bars
            if remove:
                for sym in remove:
                    try:
                        await self._streamer.unsubscribe_candle(sym, interval="5m")
                    except Exception:
                        pass
                self._subscribed_bars -= remove

        if quotes:
            remove = set(quotes) & self._subscribed_quotes
            if remove:
                try:
                    await self._streamer.unsubscribe(Quote, list(remove))
                except Exception:
                    pass
                self._subscribed_quotes -= remove

    async def update_subscriptions(
        self,
        bars: Optional[list[str]] = None,
        quotes: Optional[list[str]] = None,
    ) -> None:
        """Replace current subscriptions with new set."""
        new_bars = set(bars) if bars else set()
        new_quotes = set(quotes) if quotes else set()

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
        Run the DXLink data stream loop with auto-reconnect.

        Listens for Candle and Quote events and dispatches to callbacks.
        DXLink sends native 5-min candles — no aggregation needed.
        """
        from tastytrade.dxfeed import Quote

        backoff = 1

        while not self._shutting_down:
            try:
                if not self._data_connected:
                    success = await self.connect_data()
                    if not success:
                        await asyncio.sleep(min(backoff, self._reconnect_max))
                        backoff *= 2
                        continue
                    # Resubscribe after reconnect
                    if self._subscribed_bars:
                        start = datetime.now(timezone.utc) - timedelta(hours=1)
                        await self._streamer.subscribe_candle(
                            list(self._subscribed_bars),
                            interval="5m",
                            start_time=start,
                        )
                    if self._subscribed_quotes:
                        await self._streamer.subscribe(
                            Quote, list(self._subscribed_quotes)
                        )
                    logger.info("[WS:DATA] Resubscribed after reconnect")
                    backoff = 1

                # Listen for both candle and quote events concurrently
                candle_task = asyncio.create_task(
                    self._listen_candles()
                )
                quote_task = asyncio.create_task(
                    self._listen_quotes()
                )

                # Wait for either to complete (which means error/disconnect)
                done, pending = await asyncio.wait(
                    [candle_task, quote_task],
                    return_when=asyncio.FIRST_EXCEPTION,
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

                # Check for exceptions
                for task in done:
                    if task.exception():
                        raise task.exception()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[WS:DATA] Stream error: {e}")
                self._data_connected = False
                if self._streamer:
                    try:
                        await self._streamer.__aexit__(None, None, None)
                    except Exception:
                        pass
                    self._streamer = None
                await asyncio.sleep(min(backoff, self._reconnect_max))
                backoff *= 2

    async def _listen_candles(self) -> None:
        """Listen for DXLink candle events and dispatch to callback."""
        from tastytrade.dxfeed import Candle

        async for candle in self._streamer.listen(Candle):
            if self._shutting_down:
                break
            if self._on_bar:
                try:
                    bar_msg = self._normalize_candle(candle)
                    if bar_msg:
                        await self._on_bar(bar_msg)
                except Exception as e:
                    logger.error(f"[WS:DATA] Bar callback error: {e}")

    async def _listen_quotes(self) -> None:
        """Listen for DXLink quote events and dispatch to callback."""
        from tastytrade.dxfeed import Quote

        async for quote in self._streamer.listen(Quote):
            if self._shutting_down:
                break
            if self._on_quote:
                try:
                    quote_msg = self._normalize_quote(quote)
                    if quote_msg:
                        await self._on_quote(quote_msg)
                except Exception as e:
                    logger.error(f"[WS:DATA] Quote callback error: {e}")

    async def run_trade_loop(self) -> None:
        """
        Poll for order fill updates.

        Checks /accounts/{acct}/orders/live every 5s when orders are pending.
        Dispatches fill/cancel/reject events to the trade update callback.
        """
        while not self._shutting_down:
            try:
                if not self._poll_orders:
                    await asyncio.sleep(5)
                    continue

                orders = self._client.get_orders(status="open")
                current_ids = {o["id"] for o in orders}

                # Detect newly completed orders
                removed = self._last_order_ids - current_ids
                for order_id in removed:
                    # Order was filled, cancelled, or expired
                    if self._on_trade_update:
                        # Try to get the order details
                        try:
                            all_orders = self._client.get_orders(status="all")
                            order = next(
                                (o for o in all_orders if o["id"] == order_id),
                                None,
                            )
                            if order:
                                event = "fill" if order["status"] == "filled" else order["status"]
                                update = {
                                    "event": event,
                                    "order": order,
                                    "price": str(order.get("filled_avg_price", "0")),
                                    "qty": str(order.get("filled_qty", "0")),
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                }
                                await self._on_trade_update(update)
                        except Exception as e:
                            logger.debug(f"[WS:TRADE] Could not get order details: {e}")

                self._last_order_ids = current_ids
                await asyncio.sleep(self._poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[WS:TRADE] Polling error: {e}")
                await asyncio.sleep(self._poll_interval)

    async def run_news_loop(self) -> None:
        """No-op: tastytrade has no news streaming API."""
        while not self._shutting_down:
            await asyncio.sleep(60)

    # ── Normalization ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_candle(candle) -> Optional[dict]:
        """
        Convert DXLink Candle event to Alpaca-compatible bar format.

        Returns dict matching: {"T":"b", "S":"AAPL", "o":..., "h":..., ...}
        """
        # Extract symbol from candle symbol (e.g., "AAPL{=5m}" → "AAPL")
        event_symbol = getattr(candle, "eventSymbol", "") or ""
        symbol = event_symbol.split("{")[0] if "{" in event_symbol else event_symbol

        if not symbol:
            return None

        candle_time = getattr(candle, "time", None)
        if candle_time and isinstance(candle_time, (int, float)):
            # DXLink sends time as milliseconds since epoch
            timestamp = datetime.fromtimestamp(
                candle_time / 1000, tz=timezone.utc
            ).isoformat()
        elif candle_time:
            timestamp = str(candle_time)
        else:
            timestamp = datetime.now(timezone.utc).isoformat()

        open_val = getattr(candle, "open", 0) or 0
        high_val = getattr(candle, "high", 0) or 0
        low_val = getattr(candle, "low", 0) or 0
        close_val = getattr(candle, "close", 0) or 0
        volume_val = getattr(candle, "volume", 0) or 0

        # Skip empty/zero candles
        if float(close_val) == 0:
            return None

        return {
            "T": "b",
            "S": symbol,
            "o": float(open_val),
            "h": float(high_val),
            "l": float(low_val),
            "c": float(close_val),
            "v": int(volume_val),
            "t": timestamp,
            "n": 0,
            "vw": float(close_val),  # DXLink doesn't provide VWAP; use close
        }

    @staticmethod
    def _normalize_quote(quote) -> Optional[dict]:
        """
        Convert DXLink Quote event to Alpaca-compatible quote format.

        Returns dict matching: {"T":"q", "S":"AAPL", "bp":..., "ap":..., ...}
        """
        symbol = getattr(quote, "eventSymbol", "") or ""
        if not symbol:
            return None

        bid = float(getattr(quote, "bidPrice", 0) or 0)
        ask = float(getattr(quote, "askPrice", 0) or 0)
        bid_size = int(getattr(quote, "bidSize", 0) or 0)
        ask_size = int(getattr(quote, "askSize", 0) or 0)

        if bid == 0 and ask == 0:
            return None

        quote_time = getattr(quote, "time", None)
        if quote_time and isinstance(quote_time, (int, float)):
            timestamp = datetime.fromtimestamp(
                quote_time / 1000, tz=timezone.utc
            ).isoformat()
        else:
            timestamp = datetime.now(timezone.utc).isoformat()

        return {
            "T": "q",
            "S": symbol,
            "bp": bid,
            "bs": bid_size,
            "ap": ask,
            "as": ask_size,
            "t": timestamp,
            "c": [],
            "z": "",
        }

    # ── Shutdown ───────────────────────────────────────────────────────

    async def disconnect(self) -> None:
        """Gracefully disconnect all connections."""
        self._shutting_down = True
        self._poll_orders = False

        if self._streamer:
            try:
                await self._streamer.__aexit__(None, None, None)
                logger.info("[WS:DATA] DXLink disconnected")
            except Exception as e:
                logger.debug(f"[WS:DATA] Error during disconnect: {e}")
            self._streamer = None

        self._data_connected = False
        self._trade_connected = False

    # ── Status ─────────────────────────────────────────────────────────

    @property
    def data_connected(self) -> bool:
        """Whether market data stream is connected."""
        return self._data_connected

    @property
    def trade_connected(self) -> bool:
        """Whether trade updates polling is active."""
        return self._trade_connected

    @property
    def news_connected(self) -> bool:
        """Always False — tastytrade has no news stream."""
        return False

    @property
    def subscribed_symbols(self) -> dict:
        """Currently subscribed symbols by type."""
        return {
            "bars": sorted(self._subscribed_bars),
            "quotes": sorted(self._subscribed_quotes),
        }

    def get_status(self) -> dict:
        """Get full connection status."""
        return {
            "data_connected": self.data_connected,
            "trade_connected": self.trade_connected,
            "news_connected": False,
            "feed": "dxlink",
            "paper": get_settings().is_paper,
            "subscriptions": self.subscribed_symbols,
            "timestamp": datetime.now().isoformat(),
        }

    def enable_order_polling(self) -> None:
        """Enable order polling (called when an order is submitted)."""
        self._poll_orders = True
        self._trade_connected = True
