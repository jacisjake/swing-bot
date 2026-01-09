"""
Bot scheduler.

Manages scheduled jobs for signal checking and position monitoring.
Uses APScheduler for reliable scheduling.
"""

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from src.bot.config import BotConfig

if TYPE_CHECKING:
    from src.bot.main import TradingBot


class BotScheduler:
    """
    Scheduler for trading bot jobs.

    Jobs:
    - Stock signal check: Every N minutes during market hours (M-F 9:30-16:00 ET)
    - Crypto signal check: Every N minutes 24/7
    - Position monitor: Every minute always
    - Broker sync: Every N minutes always
    """

    def __init__(self, config: BotConfig):
        """
        Initialize scheduler.

        Args:
            config: Bot configuration
        """
        self.config = config
        self.scheduler = AsyncIOScheduler(timezone="America/New_York")

        # Job callbacks (set by TradingBot)
        self._stock_signal_callback: Optional[Callable] = None
        self._crypto_signal_callback: Optional[Callable] = None
        self._position_monitor_callback: Optional[Callable] = None
        self._broker_sync_callback: Optional[Callable] = None
        self._watchlist_refresh_callback: Optional[Callable] = None

        # Track job status
        self._is_running = False

    def set_callbacks(
        self,
        stock_signal: Optional[Callable] = None,
        crypto_signal: Optional[Callable] = None,
        position_monitor: Optional[Callable] = None,
        broker_sync: Optional[Callable] = None,
        watchlist_refresh: Optional[Callable] = None,
    ) -> None:
        """
        Set job callbacks.

        Args:
            stock_signal: Callback for stock signal checking
            crypto_signal: Callback for crypto signal checking
            position_monitor: Callback for position monitoring
            broker_sync: Callback for broker synchronization
            watchlist_refresh: Callback for refreshing watchlist from screeners
        """
        self._stock_signal_callback = stock_signal
        self._crypto_signal_callback = crypto_signal
        self._position_monitor_callback = position_monitor
        self._broker_sync_callback = broker_sync
        self._watchlist_refresh_callback = watchlist_refresh

    def setup_jobs(self) -> None:
        """Configure all scheduled jobs."""

        # Stock signals: Every N minutes, M-F 9:30-16:00 ET
        if self._stock_signal_callback:
            self.scheduler.add_job(
                self._run_stock_signals,
                CronTrigger(
                    day_of_week="mon-fri",
                    hour="9-15",
                    minute=f"*/{self.config.stock_check_interval_minutes}",
                    timezone="America/New_York",
                ),
                id="stock_signals",
                name="Stock Signal Check",
                replace_existing=True,
            )
            # Also run at market open (9:30)
            self.scheduler.add_job(
                self._run_stock_signals,
                CronTrigger(
                    day_of_week="mon-fri",
                    hour="9",
                    minute="30",
                    timezone="America/New_York",
                ),
                id="stock_signals_open",
                name="Stock Signal Check (Open)",
                replace_existing=True,
            )

        # Crypto signals: Every N minutes, 24/7
        if self._crypto_signal_callback:
            self.scheduler.add_job(
                self._run_crypto_signals,
                IntervalTrigger(minutes=self.config.crypto_check_interval_minutes),
                id="crypto_signals",
                name="Crypto Signal Check",
                replace_existing=True,
            )

        # Position monitor: Every minute, always
        if self._position_monitor_callback:
            self.scheduler.add_job(
                self._run_position_monitor,
                IntervalTrigger(minutes=self.config.position_monitor_interval_minutes),
                id="position_monitor",
                name="Position Monitor",
                replace_existing=True,
            )

        # Broker sync: Every N minutes, always
        if self._broker_sync_callback:
            self.scheduler.add_job(
                self._run_broker_sync,
                IntervalTrigger(minutes=self.config.broker_sync_interval_minutes),
                id="broker_sync",
                name="Broker Sync",
                replace_existing=True,
            )

        # Watchlist refresh: Every N minutes during market hours
        if self._watchlist_refresh_callback:
            self.scheduler.add_job(
                self._run_watchlist_refresh,
                CronTrigger(
                    day_of_week="mon-fri",
                    hour="9-15",
                    minute=f"*/{self.config.watchlist_refresh_interval_minutes}",
                    timezone="America/New_York",
                ),
                id="watchlist_refresh",
                name="Watchlist Refresh",
                replace_existing=True,
            )

    async def _run_stock_signals(self) -> None:
        """Run stock signal callback with error handling."""
        if self._stock_signal_callback:
            try:
                await self._stock_signal_callback()
            except Exception as e:
                print(f"[{datetime.now()}] Stock signal error: {e}")

    async def _run_crypto_signals(self) -> None:
        """Run crypto signal callback with error handling."""
        if self._crypto_signal_callback:
            try:
                await self._crypto_signal_callback()
            except Exception as e:
                print(f"[{datetime.now()}] Crypto signal error: {e}")

    async def _run_position_monitor(self) -> None:
        """Run position monitor callback with error handling."""
        if self._position_monitor_callback:
            try:
                await self._position_monitor_callback()
            except Exception as e:
                print(f"[{datetime.now()}] Position monitor error: {e}")

    async def _run_broker_sync(self) -> None:
        """Run broker sync callback with error handling."""
        if self._broker_sync_callback:
            try:
                await self._broker_sync_callback()
            except Exception as e:
                print(f"[{datetime.now()}] Broker sync error: {e}")

    async def _run_watchlist_refresh(self) -> None:
        """Run watchlist refresh callback with error handling."""
        if self._watchlist_refresh_callback:
            try:
                await self._watchlist_refresh_callback()
            except Exception as e:
                print(f"[{datetime.now()}] Watchlist refresh error: {e}")

    def start(self) -> None:
        """Start the scheduler."""
        if not self._is_running:
            self.setup_jobs()
            self.scheduler.start()
            self._is_running = True
            print(f"[{datetime.now()}] Scheduler started")

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._is_running:
            self.scheduler.shutdown(wait=True)
            self._is_running = False
            print(f"[{datetime.now()}] Scheduler stopped")

    def pause(self) -> None:
        """Pause all jobs."""
        self.scheduler.pause()
        print(f"[{datetime.now()}] Scheduler paused")

    def resume(self) -> None:
        """Resume all jobs."""
        self.scheduler.resume()
        print(f"[{datetime.now()}] Scheduler resumed")

    def get_jobs(self) -> list[dict]:
        """Get list of scheduled jobs."""
        return [
            {
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
            }
            for job in self.scheduler.get_jobs()
        ]

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._is_running

    def is_market_open(self) -> bool:
        """
        Check if US stock market is currently open.

        Simple check based on time - doesn't account for holidays.
        """
        now = datetime.now()

        # Weekend check
        if now.weekday() >= 5:
            return False

        # TODO: Use Alpaca market clock for accurate check
        # For now, approximate ET hours
        hour = now.hour
        minute = now.minute

        # Roughly 9:30 ET to 16:00 ET
        # This is a rough check - proper implementation would use Alpaca API
        if hour < 9 or hour >= 16:
            return False

        if hour == 9 and minute < 30:
            return False

        return True
