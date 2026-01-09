"""
Main trading bot.

Orchestrates all components: signals, processing, execution, monitoring.
"""

import asyncio
import signal
import sys
from datetime import datetime
from typing import Optional

from src.bot.config import BotConfig, get_bot_config
from src.bot.executor import TradeExecutor
from src.bot.monitor import PositionMonitor
from src.bot.processor import SignalProcessor
from src.bot.scheduler import BotScheduler
from src.bot.screener import StockScreener, CryptoScreener
from src.bot.signals.base import Signal
from src.bot.signals.macd import MACDStrategy
from src.bot.signals.mean_reversion import MeanReversionStrategy
from src.bot.state.persistence import BotState
from src.core.alpaca_client import AlpacaClient
from src.core.order_executor import OrderExecutor
from src.core.position_manager import PositionManager
from src.risk.portfolio_limits import PortfolioLimits
from src.risk.position_sizer import PositionSizer


class TradingBot:
    """
    Main trading bot controller.

    Coordinates:
    - Signal generation (strategies)
    - Signal processing (risk checks)
    - Trade execution
    - Position monitoring
    - State persistence
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
        self.position_manager = PositionManager()
        self.order_executor = OrderExecutor(self.client)

        # Risk components
        self.position_sizer = PositionSizer(
            max_position_risk_pct=self.config.max_position_risk_pct,
        )
        self.portfolio_limits = PortfolioLimits(
            max_drawdown_pct=self.config.max_drawdown_pct,
            max_positions=self.config.max_positions,
        )

        # Screeners for dynamic watchlist
        self.stock_screener = StockScreener()
        self.crypto_screener = CryptoScreener()

        # Dynamic watchlists (refreshed periodically)
        self._stock_watchlist: list[str] = list(self.config.stock_symbols)  # Start with config
        self._crypto_watchlist: list[str] = list(self.config.crypto_symbols)  # Start with config

        # State components
        self.bot_state = BotState(self.config.bot_state_file)

        # Strategies
        self.stock_strategy = MACDStrategy(
            fast_period=self.config.macd_fast_period,
            slow_period=self.config.macd_slow_period,
            signal_period=self.config.macd_signal_period,
            atr_stop_multiplier=self.config.stock_atr_stop_multiplier,
            min_signal_strength=self.config.min_signal_strength,
            risk_reward_target=self.config.min_risk_reward,
        )
        self.crypto_strategy = MeanReversionStrategy(
            rsi_period=self.config.crypto_rsi_period,
            rsi_oversold=self.config.crypto_rsi_oversold,
            rsi_exit=self.config.crypto_rsi_exit,
            bb_period=self.config.crypto_bb_period,
            bb_std=self.config.crypto_bb_std,
            atr_stop_multiplier=self.config.crypto_atr_stop_multiplier,
            min_signal_strength=self.config.min_signal_strength,
        )

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
            strategies={
                "macd": self.stock_strategy,
                "mean_reversion": self.crypto_strategy,
            },
        )

        # Scheduler
        self.scheduler = BotScheduler(self.config)
        self.scheduler.set_callbacks(
            stock_signal=self._check_stock_signals,
            crypto_signal=self._check_crypto_signals if self.config.enable_crypto_trading else None,
            position_monitor=self._monitor_positions,
            broker_sync=self._sync_with_broker,
            watchlist_refresh=self._refresh_watchlist,
        )

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the trading bot."""
        print(f"[{datetime.now()}] Starting trading bot...")
        print(f"  Mode: {self.config.trading_mode.value}")
        print(f"  Crypto trading: {'enabled' if self.config.enable_crypto_trading else 'PAUSED'}")

        # Initial watchlist refresh from screeners
        await self._refresh_watchlist()
        print(f"  Stocks: {self._stock_watchlist}")
        print(f"  Crypto: {self._crypto_watchlist}")

        # Initial sync with broker
        await self._sync_with_broker()

        # Start scheduler
        self.scheduler.start()
        self._running = True

        print(f"[{datetime.now()}] Trading bot started")
        print("  Jobs scheduled:")
        for job in self.scheduler.get_jobs():
            print(f"    - {job['name']}: next run {job['next_run']}")

        # Wait for shutdown signal
        await self._shutdown_event.wait()

    async def stop(self) -> None:
        """Stop the trading bot gracefully."""
        print(f"[{datetime.now()}] Stopping trading bot...")

        self._running = False
        self.scheduler.stop()

        # Save state
        self.bot_state.save()

        print(f"[{datetime.now()}] Trading bot stopped")
        self._shutdown_event.set()

    def request_shutdown(self) -> None:
        """Request bot shutdown (called from signal handler)."""
        asyncio.create_task(self.stop())

    async def _sync_with_broker(self) -> None:
        """Sync positions and account with broker."""
        try:
            # Get account info
            account = self.client.get_account()
            equity = float(account.get("equity", 0))
            buying_power = float(account.get("buying_power", 0))

            # Update components
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

            print(f"[{datetime.now()}] Synced: ${equity:.2f} equity, {len(broker_positions)} positions")

        except Exception as e:
            print(f"[{datetime.now()}] Broker sync error: {e}")

    async def _add_default_stops(self, symbol: str) -> None:
        """Add default stop-loss and take-profit to a broker-synced position."""
        position = self.position_manager.get_position(symbol)
        if not position:
            return

        try:
            # Get bars to calculate ATR
            is_crypto = "/" in symbol or symbol.endswith("USD")
            timeframe = "1Hour" if is_crypto else "1Day"
            bars = self.client.get_bars(symbol, timeframe=timeframe, limit=20)

            if bars is None or len(bars) < 14:
                # Fallback: use percentage-based stops
                stop_pct = 0.05 if is_crypto else 0.03  # 5% crypto, 3% stocks
                position.stop_loss = position.entry_price * (1 - stop_pct)
                position.take_profit = position.entry_price * (1 + stop_pct * 2)
                position.trailing_stop_pct = stop_pct
                print(f"  Added default stops for {symbol} (percentage-based)")
                return

            # Calculate ATR for dynamic stops
            from src.data.indicators import atr
            atr_value = atr(bars["high"], bars["low"], bars["close"], period=14).iloc[-1]

            # Use config multipliers
            if is_crypto:
                stop_mult = self.config.crypto_atr_stop_multiplier
            else:
                stop_mult = self.config.stock_atr_stop_multiplier

            # Calculate stop and target
            from src.core.position_manager import PositionSide
            if position.side == PositionSide.LONG:
                position.stop_loss = position.entry_price - (atr_value * stop_mult)
                position.take_profit = position.entry_price + (atr_value * stop_mult * 2)
            else:
                position.stop_loss = position.entry_price + (atr_value * stop_mult)
                position.take_profit = position.entry_price - (atr_value * stop_mult * 2)

            # Add trailing stop
            position.trailing_stop_pct = (atr_value * stop_mult) / position.entry_price

            print(f"  Added ATR-based stops for {symbol}: SL=${position.stop_loss:.2f}, TP=${position.take_profit:.2f}")

        except Exception as e:
            print(f"  Could not add stops for {symbol}: {e}")

    async def _refresh_watchlist(self) -> None:
        """Refresh watchlists from screeners."""
        try:
            print(f"[{datetime.now()}] Refreshing watchlists from screeners...")

            # Get dynamic stock watchlist from screeners
            new_stocks = set(self.config.stock_symbols)  # Start with config symbols

            # Add top gainers and most active
            screener_symbols = self.stock_screener.get_combined_watchlist(
                include_gainers=True,
                include_active=True,
                include_losers=False,  # Skip losers for momentum strategy
                top_n=5,
            )
            new_stocks.update(screener_symbols)

            # Check for volume breakouts on existing watchlist
            if self._stock_watchlist:
                breakouts = self.stock_screener.get_volume_breakouts(
                    symbols=list(new_stocks),
                    volume_threshold=1.5,
                    lookback_days=20,
                )
                for result in breakouts[:5]:  # Top 5 volume breakouts
                    new_stocks.add(result.symbol)

            self._stock_watchlist = list(new_stocks)

            # Update crypto watchlist with movers
            new_crypto = set(self.config.crypto_symbols)  # Start with config symbols
            momentum = self.crypto_screener.get_momentum_crypto()
            new_crypto.update(momentum)

            # Add symbols from open positions to ensure we monitor them
            for pos in self.position_manager.get_open_positions():
                symbol = pos.symbol
                # Normalize symbol format (ETHUSD -> ETH/USD)
                if symbol.endswith("USD") and "/" not in symbol:
                    normalized = symbol[:-3] + "/USD"
                    new_crypto.add(normalized)
                elif "/" in symbol:
                    new_crypto.add(symbol)
                else:
                    new_stocks.add(symbol)

            self._stock_watchlist = list(new_stocks)
            self._crypto_watchlist = list(new_crypto)

            self.bot_state.update_job_timestamp("watchlist_refresh")
            print(f"  Stocks: {len(self._stock_watchlist)} symbols")
            print(f"  Crypto: {len(self._crypto_watchlist)} symbols")

        except Exception as e:
            print(f"[{datetime.now()}] Watchlist refresh error: {e}")
            # Keep existing watchlists on error

    async def _check_stock_signals(self) -> None:
        """Check stock watchlist for signals."""
        if not self._running:
            return

        self.bot_state.update_job_timestamp("stock_signals")
        print(f"[{datetime.now()}] Checking stock signals...")

        account = self.client.get_account()
        equity = float(account.get("equity", 0))
        buying_power = float(account.get("buying_power", 0))
        current_positions = len(self.position_manager.get_open_positions())

        for symbol in self._stock_watchlist:
            # Skip if already have position
            if self.position_manager.has_position(symbol):
                continue

            # Skip if already have active signal
            if self.bot_state.has_active_signal(symbol):
                continue

            try:
                signal = await self._generate_stock_signal(symbol)
                if signal:
                    await self._process_signal(signal, equity, buying_power, current_positions)
            except Exception as e:
                print(f"  Error checking {symbol}: {e}")

    async def _check_crypto_signals(self) -> None:
        """Check crypto watchlist for signals."""
        if not self._running:
            return

        self.bot_state.update_job_timestamp("crypto_signals")
        print(f"[{datetime.now()}] Checking crypto signals...")

        account = self.client.get_account()
        equity = float(account.get("equity", 0))
        buying_power = float(account.get("buying_power", 0))
        current_positions = len(self.position_manager.get_open_positions())

        for symbol in self._crypto_watchlist:
            # Skip if already have position
            if self.position_manager.has_position(symbol):
                continue

            # Skip if already have active signal
            if self.bot_state.has_active_signal(symbol):
                continue

            try:
                signal = await self._generate_crypto_signal(symbol)
                if signal:
                    await self._process_signal(signal, equity, buying_power, current_positions)
            except Exception as e:
                print(f"  Error checking {symbol}: {e}")

    async def _generate_stock_signal(self, symbol: str) -> Optional[Signal]:
        """Generate signal for a stock symbol."""
        try:
            bars = self.client.get_bars(
                symbol, timeframe=self.config.stock_timeframe, limit=100
            )
            if bars is None or len(bars) < 30:
                return None

            current_price = self.client.get_latest_price(symbol)
            return self.stock_strategy.generate(symbol, bars, current_price)
        except Exception:
            return None

    async def _generate_crypto_signal(self, symbol: str) -> Optional[Signal]:
        """Generate signal for a crypto symbol."""
        try:
            bars = self.client.get_bars(symbol, timeframe="1Hour", limit=50)
            if bars is None or len(bars) < 25:
                return None

            current_price = self.client.get_latest_price(symbol)
            return self.crypto_strategy.generate(symbol, bars, current_price)
        except Exception:
            return None

    async def _process_signal(
        self,
        signal: Signal,
        equity: float,
        buying_power: float,
        current_positions: int,
    ) -> None:
        """Process a signal through validation and execution."""
        print(f"  Signal: {signal.symbol} {signal.direction.value} (strength: {signal.strength:.2f})")

        # Process through risk checks
        result = self.processor.process(
            signal=signal,
            account_equity=equity,
            buying_power=buying_power,
            current_positions=current_positions,
        )

        if not result.passed:
            print(f"    Rejected: {result.rejection_reason}")
            self.bot_state.remove_active_signal(signal.symbol, executed=False)
            return

        for warning in result.warnings:
            print(f"    Warning: {warning}")

        # Add to active signals
        self.bot_state.add_signal(signal)

        # Execute trade
        trade_params = result.trade_params
        print(f"    Executing: {trade_params.quantity:.4f} shares @ ~${trade_params.entry_price:.2f}")

        exec_result = await self.executor.execute_entry(trade_params)

        if exec_result.success:
            print(f"    Filled: {exec_result.order_result.filled_qty:.4f} @ ${exec_result.order_result.avg_fill_price:.2f}")
            self.bot_state.remove_active_signal(signal.symbol, executed=True)
        else:
            print(f"    Failed: {exec_result.error}")
            self.bot_state.remove_active_signal(signal.symbol, executed=False)

    async def _monitor_positions(self) -> None:
        """Monitor positions for exit conditions."""
        if not self._running:
            return

        self.bot_state.update_job_timestamp("position_monitor")

        exit_signals = await self.monitor.check_all_positions()

        for exit_signal in exit_signals:
            print(f"[{datetime.now()}] Exit signal: {exit_signal.symbol} - {exit_signal.reason}")

            # Execute exit
            exec_result = await self.executor.execute_exit(
                symbol=exit_signal.symbol,
                reason=exit_signal.reason,
            )

            if exec_result.success:
                pnl = exec_result.position.realized_pnl if exec_result.position else 0
                print(f"  Closed: P&L ${pnl:.2f}")
            else:
                print(f"  Failed: {exec_result.error}")

    async def health_check(self) -> dict:
        """Get bot health status."""
        account = self.client.get_account()

        return {
            "running": self._running,
            "scheduler_running": self.scheduler.is_running,
            "market_open": self.scheduler.is_market_open(),
            "account": {
                "equity": float(account.get("equity", 0)),
                "buying_power": float(account.get("buying_power", 0)),
            },
            "watchlists": {
                "stocks": self._stock_watchlist,
                "crypto": self._crypto_watchlist,
            },
            "positions": self.monitor.get_positions_summary(),
            "state": self.bot_state.get_state_summary(),
            "jobs": self.scheduler.get_jobs(),
        }


def setup_signal_handlers(bot: TradingBot) -> None:
    """Setup signal handlers for graceful shutdown."""

    def handle_signal(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
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
