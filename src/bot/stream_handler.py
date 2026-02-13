"""
Event-driven signal engine for WebSocket streaming.

Replaces the polling-based _run_momentum_scan() and _monitor_positions()
with real-time callbacks triggered by streaming bar/quote data.

Data flow:
1. Scanner (REST, every 2 min) finds candidates → update_watchlist()
2. WSS 1-min bars arrive → aggregate to 5-min → run strategy → execute
3. WSS quotes arrive → check position exits (stops/targets/trailing)
4. WSS trade updates → update position state on fills/cancels
5. WSS news → flag catalysts on watchlist symbols
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

import pandas as pd
from loguru import logger

from src.bot.signals.base import Signal, SignalDirection
from src.core.position_manager import PositionSide

if TYPE_CHECKING:
    from src.bot.executor import TradeExecutor
    from src.bot.monitor import PositionMonitor
    from src.bot.processor import SignalProcessor
    from src.bot.signals.base import SignalGenerator
    from src.bot.state.persistence import BotState
    from src.core.alpaca_client import AlpacaClient
    from src.core.position_manager import PositionManager
    from src.core.ws_client import AlpacaWSClient
    from src.risk.portfolio_limits import PortfolioLimits


class StreamHandler:
    """
    Processes streaming market data into trading signals.

    Replaces polling with event-driven approach:
    - on_bar: 1-min bars → aggregate to 5-min → signal generation
    - on_quote: real-time prices → instant exit checks
    - on_trade_update: order fills → position state updates
    - on_news: catalyst flagging
    """

    def __init__(
        self,
        strategy: "SignalGenerator",
        processor: "SignalProcessor",
        executor: "TradeExecutor",
        monitor: "PositionMonitor",
        position_manager: "PositionManager",
        portfolio_limits: "PortfolioLimits",
        bot_state: "BotState",
        client: "AlpacaClient",
        ws_client: "AlpacaWSClient",
        config,
    ):
        """
        Initialize stream handler.

        Args:
            strategy: MomentumPullbackStrategy for signal generation
            processor: SignalProcessor for risk checks
            executor: TradeExecutor for order execution
            monitor: PositionMonitor for exit checks
            position_manager: Position tracking
            portfolio_limits: Risk limits
            bot_state: Bot state persistence
            client: REST client for backfill and account data
            ws_client: WebSocket client for subscription management
            config: BotConfig
        """
        self.strategy = strategy
        self.processor = processor
        self.executor = executor
        self.monitor = monitor
        self.position_manager = position_manager
        self.portfolio_limits = portfolio_limits
        self.bot_state = bot_state
        self.client = client
        self.ws_client = ws_client
        self.config = config

        # 1-min bar buffer: symbol -> list of bar dicts
        self._bar_buffer: dict[str, list[dict]] = defaultdict(list)

        # Rolling 5-min bar history: symbol -> DataFrame
        # Seeded with backfill on subscribe, updated on each 5-min aggregation
        self._five_min_bars: dict[str, pd.DataFrame] = {}

        # Latest quotes: symbol -> {bid, ask, price, timestamp}
        self._latest_quotes: dict[str, dict] = {}

        # Current watchlist (symbols we're streaming)
        self._watchlist: list[str] = []

        # Catalyst tracking (from news stream)
        self._catalysts: dict[str, dict] = {}  # symbol -> {headline, source, time}

        # Day trading state (shared with TradingBot)
        self._daily_trades_today = 0
        self._max_daily_trades = config.max_daily_trades

        # Prevent concurrent signal processing
        self._processing_lock = asyncio.Lock()

        # Track which 5-min buckets we've already processed
        self._last_processed_bucket: dict[str, int] = {}

        # Exit cooldown: symbol -> timestamp of last exit attempt
        # Prevents hammering the API on every quote tick when exits fail
        self._exit_cooldown: dict[str, float] = {}
        self._exit_cooldown_seconds = 30  # Wait 30s between exit retries

    # ── Bar Handling ───────────────────────────────────────────────────

    async def on_bar(self, bar: dict) -> None:
        """
        Handle incoming 1-min bar from WebSocket.

        Aggregates 1-min bars into 5-min bars, then runs signal generation
        when a 5-min candle closes.

        Bar format from Alpaca:
        {"T": "b", "S": "AAPL", "o": 150.25, "h": 150.75, "l": 150.10,
         "c": 150.50, "v": 125000, "t": "2026-02-10T14:35:00Z", "n": 50, "vw": 150.40}
        """
        symbol = bar.get("S")
        if not symbol:
            return

        # Parse timestamp
        timestamp_str = bar.get("t", "")
        try:
            bar_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            bar_time = datetime.now(timezone.utc)

        # Store the 1-min bar
        bar_entry = {
            "open": float(bar.get("o", 0)),
            "high": float(bar.get("h", 0)),
            "low": float(bar.get("l", 0)),
            "close": float(bar.get("c", 0)),
            "volume": int(bar.get("v", 0)),
            "timestamp": bar_time,
            "vwap": float(bar.get("vw", 0)),
        }
        self._bar_buffer[symbol].append(bar_entry)

        # Keep buffer reasonable (last 30 bars = 30 minutes of 1-min data)
        if len(self._bar_buffer[symbol]) > 30:
            self._bar_buffer[symbol] = self._bar_buffer[symbol][-30:]

        # Determine 5-min bucket: minute 0-4 → bucket 0, 5-9 → bucket 5, etc.
        bucket = bar_time.minute // 5 * 5
        bucket_key = bar_time.hour * 100 + bucket  # e.g., 935 for 9:35

        # Check if this bar completes a new 5-min candle
        # A 5-min bar closes when we see minute X where X % 5 == 4
        # (minute 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59)
        if bar_time.minute % 5 == 4:
            # Check if we already processed this bucket
            last_bucket = self._last_processed_bucket.get(symbol, -1)
            if bucket_key != last_bucket:
                self._last_processed_bucket[symbol] = bucket_key
                await self._on_five_min_close(symbol, bar_time)

    async def _on_five_min_close(self, symbol: str, bar_time: datetime) -> None:
        """
        Handle 5-min candle close. Aggregate 1-min bars and run strategy.

        Args:
            symbol: Stock ticker
            bar_time: Timestamp of the closing 1-min bar
        """
        # Aggregate last 5 (or fewer) 1-min bars into one 5-min bar
        recent_bars = self._bar_buffer.get(symbol, [])
        if not recent_bars:
            return

        # Take the last 5 bars (or however many we have)
        bars_to_aggregate = recent_bars[-5:]

        five_min_bar = {
            "open": bars_to_aggregate[0]["open"],
            "high": max(b["high"] for b in bars_to_aggregate),
            "low": min(b["low"] for b in bars_to_aggregate),
            "close": bars_to_aggregate[-1]["close"],
            "volume": sum(b["volume"] for b in bars_to_aggregate),
            "vwap": bars_to_aggregate[-1]["vwap"],
        }

        # Append to rolling 5-min DataFrame
        if symbol not in self._five_min_bars:
            self._five_min_bars[symbol] = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume", "vwap"]
            )

        new_row = pd.DataFrame([five_min_bar], index=[bar_time])
        self._five_min_bars[symbol] = pd.concat(
            [self._five_min_bars[symbol], new_row]
        )

        # Keep last 100 5-min bars
        if len(self._five_min_bars[symbol]) > 100:
            self._five_min_bars[symbol] = self._five_min_bars[symbol].iloc[-100:]

        logger.debug(
            f"[STREAM] {symbol} 5-min bar: "
            f"O={five_min_bar['open']:.2f} H={five_min_bar['high']:.2f} "
            f"L={five_min_bar['low']:.2f} C={five_min_bar['close']:.2f} "
            f"V={five_min_bar['volume']:,}"
        )

        # Run signal generation
        await self._check_signal(symbol)

        # Check strategy exit for open position (MACD exhaustion, etc.)
        await self._check_strategy_exit(symbol)

    async def _check_signal(self, symbol: str) -> None:
        """
        Run signal generation for a symbol using accumulated 5-min bars.

        Only generates signals during trading window and if daily limits allow.
        """
        async with self._processing_lock:
            # Check daily trade limit
            if self._daily_trades_today >= self._max_daily_trades:
                return

            # Check if we have an open position
            open_positions = self.position_manager.get_open_positions()
            if len(open_positions) >= self.config.max_positions:
                return

            # Skip if already have position or active signal for this symbol
            if self.position_manager.has_position(symbol):
                return
            if self.bot_state.has_active_signal(symbol):
                return

            # Need enough bars for strategy
            bars_df = self._five_min_bars.get(symbol)
            if bars_df is None or len(bars_df) < 40:
                logger.debug(
                    f"[STREAM] {symbol}: insufficient 5-min bars "
                    f"({len(bars_df) if bars_df is not None else 0}/40)"
                )
                return

            # Get current price from latest quote or last bar close
            current_price = self._get_latest_price(symbol)
            if current_price is None:
                return

            # Generate signal
            try:
                gen_signal = self.strategy.generate(symbol, bars_df, current_price)
            except Exception as e:
                logger.debug(f"[STREAM] {symbol}: signal generation error: {e}")
                return

            if gen_signal is None:
                return

            # Inject catalyst metadata
            catalyst = self._catalysts.get(symbol, {})
            gen_signal.metadata["has_catalyst"] = bool(catalyst)
            gen_signal.metadata["news_headline"] = catalyst.get("headline", "")
            gen_signal.metadata["news_count"] = catalyst.get("count", 0)
            gen_signal.metadata["news_source"] = catalyst.get("source", "")
            gen_signal.metadata["source"] = "stream"  # Mark as stream-generated

            logger.info(
                f"[STREAM SIGNAL] {symbol}: {gen_signal.direction.value} "
                f"(strength={gen_signal.strength:.2f}, R:R={gen_signal.risk_reward_ratio:.1f})"
            )

            # Process through risk checks and execute
            await self._process_signal(gen_signal)

    async def _check_strategy_exit(self, symbol: str) -> None:
        """
        Check if an open position should exit based on strategy signals.

        Called on every 5-min bar close. Uses locally buffered bars
        (no REST call) to check MACD exhaustion and histogram fading.
        """
        position = self.position_manager.get_position(symbol)
        if position is None:
            return

        bars_df = self._five_min_bars.get(symbol)
        if bars_df is None or len(bars_df) < 40:
            return

        direction = (
            SignalDirection.LONG
            if position.side == PositionSide.LONG
            else SignalDirection.SHORT
        )

        try:
            should_exit, reason = self.strategy.should_exit(
                symbol=symbol,
                bars=bars_df,
                entry_price=position.entry_price,
                direction=direction,
                current_price=position.current_price,
            )
        except Exception as e:
            logger.debug(f"[STREAM] {symbol}: strategy exit check error: {e}")
            return

        if not should_exit:
            return

        exit_reason = f"Strategy exit: {reason}"
        logger.info(f"[STREAM STRATEGY EXIT] {symbol}: {exit_reason}")

        exec_result = await self.executor.execute_exit(
            symbol=symbol,
            reason=exit_reason,
        )
        if exec_result.success:
            pnl = exec_result.position.realized_pnl if exec_result.position else 0
            logger.info(f"  Closed {symbol}: P&L ${pnl:.2f}")
        else:
            logger.error(f"  Failed to close {symbol}: {exec_result.error}")

    async def _process_signal(self, signal: Signal) -> None:
        """Process signal through risk checks and execute if valid."""
        try:
            account = self.client.get_account()
            equity = float(account.get("equity", 0))
            buying_power = float(account.get("buying_power", 0))
        except Exception as e:
            logger.error(f"[STREAM] Failed to get account info: {e}")
            return

        current_positions = len(self.position_manager.get_open_positions())

        result = self.processor.process(
            signal=signal,
            account_equity=equity,
            buying_power=buying_power,
            current_positions=current_positions,
        )

        if not result.passed:
            logger.info(f"  Rejected: {result.rejection_reason}")
            self.bot_state.remove_active_signal(signal.symbol, executed=False)
            return

        for warning in result.warnings:
            logger.warning(f"  {warning}")

        self.bot_state.add_signal(signal)

        trade_params = result.trade_params
        logger.info(
            f"  Executing: {trade_params.quantity:.2f} shares of {signal.symbol} "
            f"@ ~${trade_params.entry_price:.2f} "
            f"(stop=${trade_params.stop_price:.2f}, target=${trade_params.target_price:.2f})"
        )

        exec_result = await self.executor.execute_entry(trade_params)

        if exec_result.success:
            logger.info(
                f"  FILLED: {exec_result.order_result.filled_qty:.2f} "
                f"@ ${exec_result.order_result.avg_fill_price:.2f}"
            )
            self.bot_state.remove_active_signal(signal.symbol, executed=True)
            self._daily_trades_today += 1
            self.portfolio_limits.record_entry()

            # Subscribe to quotes for position monitoring
            await self.ws_client.subscribe(quotes=[signal.symbol])

            logger.info(f"[STREAM TRADE] Trade #{self._daily_trades_today} executed for {signal.symbol}")
        else:
            logger.error(f"  FAILED: {exec_result.error}")
            self.bot_state.remove_active_signal(signal.symbol, executed=False)

    # ── Quote Handling ─────────────────────────────────────────────────

    async def on_quote(self, quote: dict) -> None:
        """
        Handle incoming quote from WebSocket.

        Checks position exits (stops, targets, trailing) in real-time.

        Quote format:
        {"T": "q", "S": "AAPL", "bp": 150.10, "bs": 5, "ap": 150.20, "as": 3,
         "t": "2026-02-10T14:35:00.123Z", "c": ["R"], "z": "C"}
        """
        symbol = quote.get("S")
        if not symbol:
            return

        bid = float(quote.get("bp", 0))
        ask = float(quote.get("ap", 0))

        # Calculate midpoint as "price"
        if bid > 0 and ask > 0:
            price = (bid + ask) / 2
        elif bid > 0:
            price = bid
        elif ask > 0:
            price = ask
        else:
            return

        self._latest_quotes[symbol] = {
            "bid": bid,
            "ask": ask,
            "price": price,
            "timestamp": quote.get("t", ""),
        }

        # Check if we have a position in this symbol
        position = self.position_manager.get_position(symbol)
        if position is None:
            return

        # Update position price
        position.update_price(price)

        # Check exit cooldown (don't hammer API if previous exit failed)
        import time as _time
        last_attempt = self._exit_cooldown.get(symbol, 0)
        if _time.time() - last_attempt < self._exit_cooldown_seconds:
            return

        # Quick exit checks (stops/targets/trailing — no expensive strategy check)
        exit_signal = await self.monitor.check_position_at_price(symbol, price)
        if exit_signal:
            self._exit_cooldown[symbol] = _time.time()
            logger.info(f"[STREAM EXIT] {symbol}: {exit_signal.reason}")
            exec_result = await self.executor.execute_exit(
                symbol=symbol,
                reason=exit_signal.reason,
            )
            if exec_result.success:
                pnl = exec_result.position.realized_pnl if exec_result.position else 0
                logger.info(f"  Closed {symbol}: P&L ${pnl:.2f}")
                self._exit_cooldown.pop(symbol, None)  # Clear cooldown on success
            else:
                logger.error(f"  Failed to close {symbol}: {exec_result.error} (retry in {self._exit_cooldown_seconds}s)")

    # ── Trade Update Handling ──────────────────────────────────────────

    async def on_trade_update(self, update: dict) -> None:
        """
        Handle order fill/cancel/reject from trade updates stream.

        Replaces broker sync polling.

        Update format:
        {"event": "fill", "order": {"id": "...", "symbol": "AAPL", ...},
         "price": "150.25", "qty": "100", "timestamp": "..."}
        """
        event = update.get("event", "")
        order = update.get("order", {})
        symbol = order.get("symbol", "")

        if event == "fill":
            qty = float(update.get("qty", order.get("filled_qty", 0)))
            price = float(update.get("price", order.get("filled_avg_price", 0)))
            side = order.get("side", "")

            logger.info(
                f"[WS FILL] {symbol}: {side} {qty} @ ${price:.2f}"
            )

        elif event == "partial_fill":
            qty = float(update.get("qty", 0))
            price = float(update.get("price", 0))
            logger.info(f"[WS PARTIAL] {symbol}: {qty} @ ${price:.2f}")

        elif event in ("canceled", "cancelled"):
            logger.info(f"[WS CANCEL] {symbol}: Order cancelled")

        elif event == "rejected":
            logger.warning(f"[WS REJECT] {symbol}: Order rejected")

        elif event == "new":
            logger.debug(f"[WS NEW] {symbol}: New order accepted")

        else:
            logger.debug(f"[WS EVENT] {symbol}: {event}")

    # ── News Handling ──────────────────────────────────────────────────

    async def on_news(self, article: dict) -> None:
        """
        Handle real-time news article from news stream.

        News format:
        {"T": "n", "id": 123, "headline": "...", "source": "...",
         "symbols": ["AAPL"], "created_at": "...", "url": "..."}
        """
        symbols = article.get("symbols", [])
        headline = article.get("headline", "")
        source = article.get("source", "")

        for symbol in symbols:
            if symbol in self._watchlist:
                existing = self._catalysts.get(symbol, {})
                count = existing.get("count", 0) + 1

                self._catalysts[symbol] = {
                    "headline": headline,
                    "source": source,
                    "count": count,
                    "time": datetime.now().isoformat(),
                }

                logger.info(
                    f"[STREAM NEWS] {symbol}: {headline[:80]}... "
                    f"(source={source}, total={count})"
                )

    # ── Watchlist Management ───────────────────────────────────────────

    async def update_watchlist(self, symbols: list[str]) -> None:
        """
        Update WebSocket subscriptions when scanner finds new candidates.

        1. Unsubscribes from removed symbols
        2. Subscribes to new symbols (1-min bars + quotes)
        3. Backfills 5-min bar history for new symbols
        4. Always subscribes to quotes for open position symbols
        """
        new_symbols = set(symbols)
        old_symbols = set(self._watchlist)

        # Always include open position symbols for quote monitoring
        pos_symbols = {p.symbol for p in self.position_manager.get_open_positions()}
        all_quote_symbols = new_symbols | pos_symbols

        # Update subscriptions
        await self.ws_client.update_subscriptions(
            bars=list(new_symbols),
            quotes=list(all_quote_symbols),
        )

        # Backfill 5-min bars for newly added symbols
        added = new_symbols - old_symbols
        for symbol in added:
            await self._backfill_bars(symbol)

        # Clean up removed symbols
        removed = old_symbols - new_symbols
        for symbol in removed:
            if symbol not in pos_symbols:
                self._bar_buffer.pop(symbol, None)
                self._five_min_bars.pop(symbol, None)
                self._last_processed_bucket.pop(symbol, None)

        self._watchlist = list(new_symbols)
        logger.info(
            f"[STREAM] Watchlist updated: {len(self._watchlist)} symbols "
            f"(+{len(added)}, -{len(removed)})"
        )

    async def _backfill_bars(self, symbol: str) -> None:
        """
        Backfill 5-min bar history for a symbol via REST.

        Seeds the rolling DataFrame so strategy has enough history
        to generate signals immediately.
        """
        try:
            bars = self.client.get_bars(
                symbol,
                timeframe="5Min",
                limit=100,
            )
            if bars is not None and not bars.empty:
                self._five_min_bars[symbol] = bars
                logger.info(
                    f"[STREAM] Backfilled {len(bars)} 5-min bars for {symbol}"
                )
            else:
                logger.debug(f"[STREAM] No backfill data for {symbol}")
        except Exception as e:
            logger.debug(f"[STREAM] Backfill failed for {symbol}: {e}")

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price from quote cache or last bar."""
        quote = self._latest_quotes.get(symbol)
        if quote and quote.get("price", 0) > 0:
            return quote["price"]

        # Fallback to last 5-min bar close
        bars = self._five_min_bars.get(symbol)
        if bars is not None and not bars.empty:
            return float(bars["close"].iloc[-1])

        return None

    def reset_daily(self) -> None:
        """Reset daily counters. Called by daily_reset."""
        self._daily_trades_today = 0
        self._catalysts.clear()
        logger.info("[STREAM] Daily reset: counters cleared")

    @property
    def daily_trades_today(self) -> int:
        """Number of trades executed today."""
        return self._daily_trades_today

    @daily_trades_today.setter
    def daily_trades_today(self, value: int) -> None:
        """Set daily trade count (for sync with TradingBot)."""
        self._daily_trades_today = value

    def get_status(self) -> dict:
        """Get stream handler status."""
        return {
            "watchlist": self._watchlist,
            "watchlist_count": len(self._watchlist),
            "buffered_symbols": list(self._bar_buffer.keys()),
            "five_min_bars_symbols": {
                s: len(df) for s, df in self._five_min_bars.items()
            },
            "latest_quotes": {
                s: {"price": q["price"], "timestamp": q["timestamp"]}
                for s, q in self._latest_quotes.items()
            },
            "catalysts": self._catalysts,
            "daily_trades": self._daily_trades_today,
        }
