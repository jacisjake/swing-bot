"""
Momentum day trading bot.

Orchestrates all components: scanner, signals, processing, execution, monitoring.
Targets low-float stocks ($1-$10, prefer $2+) with pullback entries on 5-min bars.

Architecture:
- WebSocket streaming for real-time bars, quotes, trade updates, news
- APScheduler for time-based events (scanner refresh, EOD cleanup, daily reset)
- REST API for scanner (no WSS equivalent), account info, order submission
"""

import asyncio
import signal
import sys
from datetime import datetime
from typing import Optional

from loguru import logger

from config.settings import get_settings
from src.bot.config import BotConfig, get_bot_config
from src.bot.executor import TradeExecutor
from src.bot.float_provider import FloatDataProvider
from src.bot.monitor import PositionMonitor
from src.bot.press_release_scanner import PressReleaseScanner
from src.bot.processor import SignalProcessor
from src.bot.scheduler import BotScheduler
from src.bot.screener import MomentumScreener
from src.bot.tradingview_screener import TradingViewScreener
from src.bot.signals.base import Signal
from src.bot.signals.momentum_pullback import MomentumPullbackStrategy
from src.bot.state.persistence import BotState
from src.bot.state.trade_ledger import TradeLedger
from src.bot.stream_handler import StreamHandler
from src.core.alpaca_client import AlpacaClient
from src.core.order_executor import OrderExecutor
from src.core.position_manager import PositionManager
from src.core.ws_client import AlpacaWSClient
from src.risk.portfolio_limits import PortfolioLimits
from src.risk.position_sizer import PositionSizer


class TradingBot:
    """
    Momentum day trading bot controller.

    Strategy: Ross Cameron-style momentum pullback on low-float stocks.
    Flow: Scanner -> Signal -> Risk Check -> Execute -> Monitor -> Exit
    Goal: One high-quality trade per day, 10% account growth.
    """

    def __init__(self, config: Optional[BotConfig] = None):
        """
        Initialize trading bot.

        Args:
            config: Bot configuration (uses default if None)
        """
        self.config = config or get_bot_config()

        # Core components
        self.client = AlpacaClient()
        self.trade_ledger = TradeLedger(
            path="state/trades.json",
            starting_capital=400.0,
            goal=4000.0,
        )
        self.position_manager = PositionManager(trade_ledger=self.trade_ledger)
        self.order_executor = OrderExecutor(self.client)

        # Risk components
        self.position_sizer = PositionSizer(
            max_position_risk_pct=self.config.max_position_risk_pct,
        )
        self.portfolio_limits = PortfolioLimits(
            max_drawdown_pct=self.config.max_drawdown_pct,
            max_daily_loss_pct=self.config.daily_loss_limit_pct,
            max_positions=self.config.max_positions,
            max_daily_trades=self.config.max_daily_trades,
        )

        # Float data provider
        self.float_provider = FloatDataProvider(
            fmp_api_key=self.config.fmp_api_key,
        )

        # TradingView screener (primary scanner, no API key required)
        self.tv_screener = TradingViewScreener() if self.config.use_tradingview_screener else None

        # Momentum scanner (TradingView primary, Alpaca fallback)
        self.momentum_scanner = MomentumScreener(
            float_provider=self.float_provider,
            alpaca_client=self.client,
            news_enabled=self.config.scanner_enable_news_check,
            news_lookback_hours=self.config.scanner_news_lookback_hours,
            news_max_articles=self.config.scanner_news_max_articles,
            tv_screener=self.tv_screener,
            use_tradingview=self.config.use_tradingview_screener,
        )

        # Press release scanner (pre-market catalyst detection)
        self.press_release_scanner = PressReleaseScanner(
            fmp_api_key=self.config.fmp_api_key,
            lookback_hours=self.config.press_release_lookback_hours,
        )

        # Strategy
        self.strategy = MomentumPullbackStrategy(
            macd_fast=self.config.macd_fast_period,
            macd_slow=self.config.macd_slow_period,
            macd_signal=self.config.macd_signal_period,
            atr_period=self.config.atr_period,
            atr_stop_multiplier=self.config.stock_atr_stop_multiplier,
            pullback_min_candles=self.config.pullback_min_candles,
            pullback_max_candles=self.config.pullback_max_candles,
            pullback_max_retracement=self.config.pullback_max_retracement,
            risk_reward_target=self.config.risk_reward_target,
            min_signal_strength=self.config.min_signal_strength,
        )

        # State
        self.bot_state = BotState(self.config.bot_state_file)

        # Bot components
        self.processor = SignalProcessor(
            config=self.config,
            position_sizer=self.position_sizer,
            portfolio_limits=self.portfolio_limits,
        )
        self.executor = TradeExecutor(
            order_executor=self.order_executor,
            position_manager=self.position_manager,
        )
        self.monitor = PositionMonitor(
            client=self.client,
            position_manager=self.position_manager,
            strategies={"momentum_pullback": self.strategy},
            trading_window_end=self.config.trading_window_end,
        )

        # WebSocket client (direct, no alpaca-py SDK)
        settings = get_settings()
        self.ws_client = AlpacaWSClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            feed=settings.alpaca_data_feed,
            paper=settings.is_paper,
            reconnect_max_seconds=self.config.ws_reconnect_max_seconds,
            heartbeat_seconds=self.config.ws_heartbeat_seconds,
        )

        # Stream handler (event-driven signal engine)
        self.stream_handler = StreamHandler(
            strategy=self.strategy,
            processor=self.processor,
            executor=self.executor,
            monitor=self.monitor,
            position_manager=self.position_manager,
            portfolio_limits=self.portfolio_limits,
            bot_state=self.bot_state,
            client=self.client,
            ws_client=self.ws_client,
            config=self.config,
        )

        # Register WebSocket callbacks
        self.ws_client.on_bar(self.stream_handler.on_bar)
        self.ws_client.on_quote(self.stream_handler.on_quote)
        self.ws_client.on_trade_update(self.stream_handler.on_trade_update)
        self.ws_client.on_news(self.stream_handler.on_news)

        # Scheduler (with Alpaca client for market clock API)
        # Only time-based jobs: scanner refresh, EOD cleanup, daily reset
        # Position monitor and broker sync replaced by WebSocket streaming
        self.scheduler = BotScheduler(self.config, client=self.client)
        self.scheduler.set_callbacks(
            momentum_scan=self._run_momentum_scan,
            press_release_scan=self._run_press_release_scan,
            end_of_day=self._end_of_day_cleanup,
            daily_reset=self._daily_reset,
        )

        # Day trading state
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._daily_trades_today = 0
        self._scanner_results = []  # Latest scanner hits

    async def start(self) -> None:
        """Start the trading bot with WebSocket streaming."""
        logger.info("Starting momentum day trading bot (WSS mode)...")
        logger.info(f"  Mode: {self.config.trading_mode.value}")
        logger.info(f"  Feed: {get_settings().alpaca_data_feed}")
        logger.info(f"  Window: {self.config.trading_window_start}-{self.config.trading_window_end} ET")
        logger.info(f"  Max daily trades: {self.config.max_daily_trades}")
        logger.info(
            f"  Scanner: ${self.config.scanner_min_price}-${self.config.scanner_max_price}, "
            f"{self.config.scanner_min_change_pct}%+ change, "
            f"{self.config.scanner_min_relative_volume}x+ relVol"
        )
        logger.info(
            f"  Primary screener: {'TradingView' if self.config.use_tradingview_screener else 'Alpaca'}"
        )

        # 1. Initial sync with broker (REST, one-time)
        await self._sync_with_broker()

        # 2. Initial scan (REST) to get watchlist
        await self._run_momentum_scan()

        # 3. Connect WebSocket streams
        logger.info("[WSS] Connecting to Alpaca WebSocket streams...")
        data_ok = await self.ws_client.connect_data()
        trade_ok = await self.ws_client.connect_trades()
        news_ok = await self.ws_client.connect_news()

        logger.info(
            f"[WSS] Connections: data={'OK' if data_ok else 'FAILED'}, "
            f"trade={'OK' if trade_ok else 'FAILED'}, "
            f"news={'OK' if news_ok else 'FAILED'}"
        )

        # 4. Subscribe to scanner results + open positions
        scan_symbols = [c.symbol for c in self._scanner_results]
        pos_symbols = [p.symbol for p in self.position_manager.get_open_positions()]
        all_symbols = list(set(scan_symbols + pos_symbols))

        if all_symbols:
            await self.ws_client.subscribe(
                bars=all_symbols,
                quotes=all_symbols,
            )
            # Backfill 5-min bar history for stream handler
            for symbol in all_symbols:
                await self.stream_handler._backfill_bars(symbol)
            logger.info(f"[WSS] Subscribed to {len(all_symbols)} symbols: {all_symbols}")

        # Subscribe to news for all watchlist symbols
        if scan_symbols:
            await self.ws_client.subscribe(news=["*"])  # All news

        # 5. Start scheduler (reduced: scanner refresh + EOD only)
        self.scheduler.start()
        self._running = True

        logger.info("Trading bot started (WSS mode)")
        logger.info("Scheduled jobs:")
        for job in self.scheduler.get_jobs():
            logger.info(f"  - {job['name']}: next run {job['next_run']}")

        # 6. Run WebSocket loops as concurrent tasks
        # These auto-reconnect internally, so they run forever until shutdown
        await asyncio.gather(
            self.ws_client.run_data_loop(),
            self.ws_client.run_trade_loop(),
            self.ws_client.run_news_loop(),
            self._shutdown_event.wait(),
        )

    async def stop(self) -> None:
        """Stop the trading bot gracefully."""
        logger.info("Stopping trading bot...")

        self._running = False

        # Disconnect WebSocket streams
        await self.ws_client.disconnect()
        logger.info("[WSS] Disconnected")

        self.scheduler.stop()

        # Save state
        self.bot_state.save()

        logger.info("Trading bot stopped")
        self._shutdown_event.set()

    def request_shutdown(self) -> None:
        """Request bot shutdown (called from signal handler)."""
        asyncio.create_task(self.stop())

    # -- Pre-Market: Press Release Scanning --------------------------------

    async def _run_press_release_scan(self) -> None:
        """
        Scan RSS feeds + FMP for overnight press releases with catalysts.

        Runs every 5 min from 4:00-7:00 AM ET (before momentum scanner).
        Builds a catalyst watchlist that gets merged with scanner results.
        """
        if not self._running:
            return

        if not self.config.enable_press_release_scanner:
            return

        try:
            new_hits = self.press_release_scanner.scan()

            if new_hits:
                status = self.press_release_scanner.get_status()
                logger.info(
                    f"[PR-SCAN] Status: {status['total_hits']} total hits, "
                    f"{status['positive_hits']} positive, "
                    f"{len(status['positive_symbols'])} unique symbols"
                )

                # Log positive catalyst symbols for the upcoming trading session
                positive_symbols = status["positive_symbols"]
                if positive_symbols:
                    logger.info(
                        f"[PR-SCAN] Catalyst watchlist: {positive_symbols}"
                    )

        except Exception as e:
            logger.error(f"[PR-SCAN] Press release scan error: {e}")

    # -- Core: Momentum Scan + Signal Generation --------------------------

    async def _run_momentum_scan(self) -> None:
        """
        Main momentum scanning loop.

        1. Check if we should scan (daily trade limit, position open)
        2. Run momentum scanner to find candidates
        3. For each candidate, fetch 5-min bars and generate signal
        4. On first valid signal, execute trade and stop scanning
        """
        if not self._running:
            return

        self.bot_state.update_job_timestamp("momentum_scan")

        # Check if we've hit daily trade limit
        if self._daily_trades_today >= self.config.max_daily_trades:
            logger.debug(
                f"Daily trade limit reached "
                f"({self._daily_trades_today}/{self.config.max_daily_trades})"
            )
            return

        # Check if we already have an open position
        open_positions = self.position_manager.get_open_positions()
        if len(open_positions) >= self.config.max_positions:
            logger.debug(
                f"Position limit reached "
                f"({len(open_positions)}/{self.config.max_positions})"
            )
            return

        # Check if we're in any scanning window
        if not self.scheduler.is_in_any_scan_window():
            logger.debug("Outside scanning window")
            return

        # Get account info
        try:
            account = self.client.get_account()
            equity = float(account.get("equity", 0))
            buying_power = float(account.get("buying_power", 0))
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return

        current_positions = len(open_positions)

        # Run the momentum scanner
        logger.info("[SCAN] Running momentum scanner...")
        try:
            candidates = self.momentum_scanner.scan(
                min_price=self.config.scanner_min_price,
                max_price=self.config.scanner_max_price,
                preferred_min_price=self.config.scanner_preferred_min_price,
                min_change_pct=self.config.scanner_min_change_pct,
                min_relative_volume=self.config.scanner_min_relative_volume,
                max_float_millions=self.config.scanner_max_float_millions,
                enable_float_filter=self.config.scanner_enable_float_filter,
                top_n=self.config.scanner_top_n,
            )
        except Exception as e:
            logger.error(f"Scanner error: {e}")
            candidates = []

        self._scanner_results = candidates

        # Enrich candidates with press release catalyst data
        if self.config.enable_press_release_scanner:
            for candidate in candidates:
                pr_hits = self.press_release_scanner.get_hits_for_symbol(candidate.symbol)
                if pr_hits:
                    # Use press release data to enrich the candidate
                    best_hit = pr_hits[0]  # Most recent
                    if not candidate.has_catalyst:
                        candidate.has_catalyst = True
                        candidate.news_headline = best_hit.headline
                        candidate.news_source = f"{best_hit.source} (PR)"
                        candidate.news_count = len(pr_hits)
                    else:
                        # Already has news from Alpaca — add PR count
                        candidate.news_count += len(pr_hits)

        if not candidates:
            logger.info("[SCAN] No candidates found")
            return

        logger.info(f"[SCAN] Found {len(candidates)} candidates:")
        for c in candidates:
            float_str = f", float={c.float_shares / 1e6:.1f}M" if c.float_shares else ""
            news_str = f", NEWS: {c.news_headline[:50]}..." if c.has_catalyst and c.news_headline else ""
            logger.info(
                f"  {c.symbol}: ${c.price:.2f} ({c.change_pct:+.1f}%), "
                f"relVol={c.relative_volume:.1f}x{float_str}{news_str}"
            )

        # Update WebSocket subscriptions with new candidates
        candidate_symbols = [c.symbol for c in candidates]
        await self.stream_handler.update_watchlist(candidate_symbols)

        # Only generate signals during active trading window (not premarket)
        if not self.scheduler.is_in_trading_window():
            logger.info("[SCAN] Pre-market only -- signals deferred until trading window")
            return

        # For each candidate, try to generate a signal
        for candidate in candidates:
            symbol = candidate.symbol

            # Skip if we already have a position or active signal
            if self.position_manager.has_position(symbol):
                continue
            if self.bot_state.has_active_signal(symbol):
                continue

            # Generate signal from 5-min bars (catalyst boosts signal strength)
            gen_signal = await self._generate_signal(
                symbol, has_catalyst=candidate.has_catalyst
            )
            if gen_signal is None:
                continue

            # Inject catalyst metadata from scanner into signal
            gen_signal.metadata["has_catalyst"] = candidate.has_catalyst
            gen_signal.metadata["news_headline"] = candidate.news_headline
            gen_signal.metadata["news_count"] = candidate.news_count
            gen_signal.metadata["news_source"] = candidate.news_source

            # Process through risk checks and execute
            logger.info(
                f"[SIGNAL] {symbol}: {gen_signal.direction.value} "
                f"(strength={gen_signal.strength:.2f}, R:R={gen_signal.risk_reward_ratio:.1f})"
            )

            executed = await self._process_signal(
                gen_signal, equity, buying_power, current_positions
            )

            if executed:
                # Success! Record trade and stop scanning
                self._daily_trades_today += 1
                self.portfolio_limits.record_entry()
                logger.info(f"[TRADE] Trade #{self._daily_trades_today} executed for {symbol}")
                return  # One trade per day

    async def _generate_signal(
        self, symbol: str, has_catalyst: bool = False
    ) -> Optional[Signal]:
        """
        Generate a signal for a symbol using 5-min bars.

        Args:
            symbol: Stock ticker
            has_catalyst: Whether the stock has a news catalyst (boosts signal strength)

        Returns:
            Signal if pullback pattern found, None otherwise
        """
        try:
            bars = self.client.get_bars(
                symbol,
                timeframe=self.config.stock_timeframe,
                limit=100,
            )
            if bars is None or len(bars) < 40:
                logger.debug(
                    f"[SIGNAL] {symbol}: insufficient bars "
                    f"({len(bars) if bars is not None else 0})"
                )
                return None

            current_price = self.client.get_latest_price(symbol)
            return self.strategy.generate(
                symbol, bars, current_price, has_catalyst=has_catalyst
            )

        except Exception as e:
            logger.debug(f"[SIGNAL] {symbol}: error generating signal: {e}")
            return None

    async def _process_signal(
        self,
        signal: Signal,
        equity: float,
        buying_power: float,
        current_positions: int,
    ) -> bool:
        """
        Process a signal through validation and execution.

        Returns True if trade was executed successfully.
        """
        # Process through risk checks
        result = self.processor.process(
            signal=signal,
            account_equity=equity,
            buying_power=buying_power,
            current_positions=current_positions,
        )

        if not result.passed:
            logger.info(f"  Rejected: {result.rejection_reason}")
            self.bot_state.remove_active_signal(signal.symbol, executed=False)
            return False

        for warning in result.warnings:
            logger.warning(f"  {warning}")

        # Add to active signals
        self.bot_state.add_signal(signal)

        # Execute trade
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
            return True
        else:
            logger.error(f"  FAILED: {exec_result.error}")
            self.bot_state.remove_active_signal(signal.symbol, executed=False)
            return False

    # -- Position Monitoring ----------------------------------------------

    async def _monitor_positions(self) -> None:
        """Monitor positions for exit conditions."""
        if not self._running:
            return

        self.bot_state.update_job_timestamp("position_monitor")

        exit_signals = await self.monitor.check_all_positions()

        for exit_signal in exit_signals:
            symbol = exit_signal.symbol
            logger.info(f"[EXIT] {symbol}: {exit_signal.reason}")

            exec_result = await self.executor.execute_exit(
                symbol=symbol,
                reason=exit_signal.reason,
            )

            if exec_result.success:
                pnl = exec_result.position.realized_pnl if exec_result.position else 0
                logger.info(f"  Closed {symbol}: P&L ${pnl:.2f}")
            else:
                logger.error(f"  Failed to close {symbol}: {exec_result.error}")

    # -- End of Day -------------------------------------------------------

    async def _end_of_day_cleanup(self) -> None:
        """
        End-of-day cleanup: close all positions, cancel all orders.

        Called at 10:05 AM ET (after trading window) and 3:55 PM ET (safety net).
        """
        logger.info("[EOD] Running end-of-day cleanup...")

        # Cancel all pending orders
        try:
            cancelled = await self.executor.cancel_pending_orders()
            if cancelled:
                logger.info(f"[EOD] Cancelled {cancelled} pending orders")
        except Exception as e:
            logger.error(f"[EOD] Error cancelling orders: {e}")

        # Close all open positions
        positions = self.position_manager.get_open_positions()
        for position in positions:
            logger.info(f"[EOD] Closing {position.symbol} ({position.qty} shares)")
            exec_result = await self.executor.execute_exit(
                symbol=position.symbol,
                reason="end_of_day_cleanup",
            )
            if exec_result.success:
                pnl = exec_result.position.realized_pnl if exec_result.position else 0
                logger.info(f"  Closed: P&L ${pnl:.2f}")
            else:
                logger.error(f"  Failed: {exec_result.error}")

        logger.info("[EOD] Cleanup complete")

    async def _daily_reset(self) -> None:
        """
        Daily reset: clear counters, refresh state.

        Called at 6:00 AM ET before pre-market scanning starts.
        """
        logger.info("[RESET] Daily reset...")

        self._daily_trades_today = 0
        self._scanner_results = []

        # Reset stream handler daily counters
        self.stream_handler.reset_daily()

        # Reset press release scanner
        self.press_release_scanner.reset_daily()

        # Reset portfolio limits daily counters
        self.portfolio_limits.reset_daily_limits()

        # Sync fresh state from broker
        await self._sync_with_broker()

        logger.info("[RESET] Daily reset complete. Ready for pre-market scanning.")

    # -- Broker Sync ------------------------------------------------------

    async def _sync_with_broker(self) -> None:
        """Sync positions and account with broker."""
        try:
            account = self.client.get_account()
            equity = float(account.get("equity", 0))
            buying_power = float(account.get("buying_power", 0))

            # Update risk components
            self.portfolio_limits.update_equity(equity)

            # Get broker positions
            broker_positions = self.client.get_positions()

            # Track which positions exist before sync
            existing_symbols = set(self.position_manager.get_symbols())

            # Sync with position manager
            self.position_manager.sync_with_broker(
                broker_positions=[
                    {
                        "symbol": p["symbol"],
                        "qty": float(p["qty"]),
                        "avg_entry_price": float(p["avg_entry_price"]),
                        "current_price": float(p["current_price"]),
                    }
                    for p in broker_positions
                ],
                equity=equity,
            )

            # Add default stops to newly synced positions
            for bp in broker_positions:
                symbol = bp["symbol"]
                if symbol not in existing_symbols:
                    await self._add_default_stops(symbol)

            self.bot_state.update_job_timestamp("broker_sync")

            logger.info(
                f"[SYNC] ${equity:.2f} equity, ${buying_power:.2f} BP, "
                f"{len(broker_positions)} positions"
            )

        except Exception as e:
            logger.error(f"Broker sync error: {e}")

    async def _add_default_stops(self, symbol: str) -> None:
        """Add default stop-loss and take-profit to a broker-synced position."""
        position = self.position_manager.get_position(symbol)
        if not position:
            return

        try:
            # Get 5-min bars to calculate ATR
            bars = self.client.get_bars(symbol, timeframe="5Min", limit=20)

            if bars is None or len(bars) < 14:
                # Fallback: percentage-based stops for day trading
                stop_pct = 0.05
                position.stop_loss = position.entry_price * (1 - stop_pct)
                position.take_profit = position.entry_price * (1 + stop_pct * 2)
                position.trailing_stop_pct = stop_pct
                logger.info(f"  Added default stops for {symbol} (5% fallback)")
                return

            # Calculate ATR for dynamic stops
            from src.data.indicators import atr
            atr_value = atr(bars["high"], bars["low"], bars["close"], period=14).iloc[-1]

            stop_mult = self.config.stock_atr_stop_multiplier

            # Calculate stop and target (LONG only for momentum strategy)
            from src.core.position_manager import PositionSide
            if position.side == PositionSide.LONG:
                position.stop_loss = position.entry_price - (atr_value * stop_mult)
                position.take_profit = position.entry_price + (atr_value * stop_mult * 2)
            else:
                position.stop_loss = position.entry_price + (atr_value * stop_mult)
                position.take_profit = position.entry_price - (atr_value * stop_mult * 2)

            # Trailing stop
            position.trailing_stop_pct = (atr_value * stop_mult) / position.entry_price

            logger.info(
                f"  Added ATR stops for {symbol}: "
                f"SL=${position.stop_loss:.2f}, TP=${position.take_profit:.2f}"
            )

        except Exception as e:
            logger.warning(f"  Could not add stops for {symbol}: {e}")

    # -- Health Check -----------------------------------------------------

    async def health_check(self) -> dict:
        """Get bot health status."""
        try:
            account = self.client.get_account()
            equity = float(account.get("equity", 0))
            buying_power = float(account.get("buying_power", 0))
        except Exception:
            equity = 0
            buying_power = 0

        return {
            "running": self._running,
            "scheduler_running": self.scheduler.is_running,
            "is_trading_day": self.scheduler.is_trading_day(),
            "in_trading_window": self.scheduler.is_in_trading_window(),
            "in_premarket": self.scheduler.is_in_premarket(),
            "market_open": self.scheduler.is_market_open(),
            "account": {
                "equity": equity,
                "buying_power": buying_power,
            },
            "day_trading": {
                "trades_today": self._daily_trades_today,
                "max_daily_trades": self.config.max_daily_trades,
                "scanner_hits": len(self._scanner_results),
            },
            "tradingview": {
                "enabled": self.config.use_tradingview_screener,
                "last_query": (
                    self.tv_screener.last_query_time.isoformat()
                    if self.tv_screener and self.tv_screener.last_query_time
                    else None
                ),
            },
            "websocket": self.ws_client.get_status(),
            "stream": self.stream_handler.get_status(),
            "press_releases": self.press_release_scanner.get_status(),
            "scanner_results": [
                {
                    "symbol": c.symbol,
                    "price": c.price,
                    "change_pct": c.change_pct,
                    "relative_volume": c.relative_volume,
                    "float_millions": c.float_shares / 1e6 if c.float_shares else None,
                    "passes_all": c.passes_all_filters,
                }
                for c in self._scanner_results[:10]
            ],
            "positions": self.monitor.get_positions_summary(),
            "state": self.bot_state.get_state_summary(),
            "jobs": self.scheduler.get_jobs(),
        }


def setup_signal_handlers(bot: TradingBot) -> None:
    """Setup signal handlers for graceful shutdown."""

    def handle_signal(signum, frame):
        logger.warning(f"Received signal {signum}, shutting down...")
        bot.request_shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


async def run_bot(config: Optional[BotConfig] = None) -> None:
    """Run the trading bot."""
    bot = TradingBot(config)
    setup_signal_handlers(bot)
    await bot.start()


if __name__ == "__main__":
    asyncio.run(run_bot())
