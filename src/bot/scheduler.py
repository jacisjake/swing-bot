"""
Momentum day trading scheduler.

Manages scheduled jobs for the trading day:
- Pre-market scanning (6:00-7:00 AM ET)
- Active momentum scanning + signal generation (7:00-10:00 AM ET)
- End-of-day cleanup (close positions, cancel orders)
- Safety net close-all (3:55 PM ET)
- Daily reset (6:00 AM ET)

Uses APScheduler with Eastern Time for all trading jobs.
Uses schedule-based NYSE holiday list for market status.
"""

from datetime import datetime, time
from typing import Callable, Optional

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from src.bot.config import BotConfig
from src.core.tastytrade_client import NYSE_HOLIDAYS

ET = pytz.timezone("America/New_York")


class BotScheduler:
    """
    Day trading scheduler.

    Schedule (all times Eastern):
    - 06:00 AM: Daily reset + start pre-market scanning (every 5 min)
    - 07:00 AM: Active scanning + signals (every 1 min)
    - 10:00 AM: Stop scanning for new entries
    - 10:05 AM: End-of-day cleanup (close positions)
    - 3:55 PM: Safety net close-all
    - Position monitor: every 30s whenever running
    - Broker sync: every 1 min whenever running
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
        self._momentum_scan_callback: Optional[Callable] = None
        self._press_release_scan_callback: Optional[Callable] = None
        self._position_monitor_callback: Optional[Callable] = None
        self._broker_sync_callback: Optional[Callable] = None
        self._end_of_day_callback: Optional[Callable] = None
        self._daily_reset_callback: Optional[Callable] = None

        # Track state
        self._is_running = False

        # Parse trading window times from config
        self._premarket_start = self._parse_time(config.premarket_scan_start)
        if config.full_day_trading:
            # Full day: scan from premarket start, trade 9:30 AM - 3:55 PM ET
            self._window_start = time(9, 30)
            self._window_end = time(15, 55)
            logger.info("[SCHEDULER] Full day trading mode: 9:30 AM - 3:55 PM ET")
        else:
            self._window_start = self._parse_time(config.trading_window_start)
            self._window_end = self._parse_time(config.trading_window_end)

    @staticmethod
    def _parse_time(time_str: str) -> time:
        """Parse HH:MM string to time object."""
        parts = time_str.split(":")
        return time(int(parts[0]), int(parts[1]))

    def set_callbacks(
        self,
        momentum_scan: Optional[Callable] = None,
        press_release_scan: Optional[Callable] = None,
        end_of_day: Optional[Callable] = None,
        daily_reset: Optional[Callable] = None,
    ) -> None:
        """
        Set job callbacks.

        Note: position_monitor and broker_sync removed — replaced by WebSocket streaming.

        Args:
            momentum_scan: Callback for momentum scanner + signal generation
            press_release_scan: Callback for pre-market press release RSS scanning
            end_of_day: Callback for end-of-day cleanup (close positions, cancel orders)
            daily_reset: Callback for daily reset (clear counters, refresh state)
        """
        self._momentum_scan_callback = momentum_scan
        self._press_release_scan_callback = press_release_scan
        self._end_of_day_callback = end_of_day
        self._daily_reset_callback = daily_reset

    def setup_jobs(self) -> None:
        """Configure all scheduled jobs for momentum day trading."""

        pre_h = self._premarket_start.hour
        pre_m = self._premarket_start.minute
        win_start_h = self._window_start.hour
        win_end_h = self._window_end.hour
        win_end_m = self._window_end.minute

        # ── 0. Press release scan: 4 AM + 9:15 AM ET ─────────────────────
        # Two scans: overnight PRs at 4 AM, last-minute earnings at 9:15 AM
        if self._press_release_scan_callback:
            pr_start = self._parse_time(self.config.press_release_scan_start)

            self.scheduler.add_job(
                self._run_press_release_scan,
                CronTrigger(
                    day_of_week="mon-fri",
                    hour=str(pr_start.hour),
                    minute=str(pr_start.minute),
                    timezone="America/New_York",
                ),
                id="press_release_scan_early",
                name="Press Release Scan (4 AM)",
                replace_existing=True,
            )

            self.scheduler.add_job(
                self._run_press_release_scan,
                CronTrigger(
                    day_of_week="mon-fri",
                    hour="9",
                    minute="15",
                    timezone="America/New_York",
                ),
                id="press_release_scan_preopen",
                name="Press Release Scan (9:15 AM)",
                replace_existing=True,
            )

        # ── 1. Pre-market scan: 6:00-6:59 AM ET, every 5 minutes ────────
        if self._momentum_scan_callback:
            self.scheduler.add_job(
                self._run_momentum_scan,
                CronTrigger(
                    day_of_week="mon-fri",
                    hour=str(pre_h),
                    minute=f"{pre_m}-59/5",
                    timezone="America/New_York",
                ),
                id="premarket_scan",
                name="Pre-Market Momentum Scan",
                replace_existing=True,
            )

            # ── 2. Active scan: 7:00-9:59 AM ET, every 1 minute ─────────
            # Build hour range for active window
            active_hours = f"{win_start_h}-{win_end_h - 1}" if win_end_h > win_start_h else str(win_start_h)

            self.scheduler.add_job(
                self._run_momentum_scan,
                CronTrigger(
                    day_of_week="mon-fri",
                    hour=active_hours,
                    minute=f"*/{self.config.stock_check_interval_minutes}",
                    timezone="America/New_York",
                ),
                id="active_scan",
                name="Active Momentum Scan",
                replace_existing=True,
            )

        # NOTE: Position monitor and broker sync removed — replaced by WebSocket streaming
        # Position exits now handled by real-time quote callbacks (StreamHandler.on_quote)
        # Broker state now handled by trade update stream (StreamHandler.on_trade_update)

        # ── 3. End-of-day cleanup: 10:05 AM ET ──────────────────────────
        if self._end_of_day_callback:
            eod_minute = win_end_m + 5  # 5 minutes after window close
            eod_hour = win_end_h
            if eod_minute >= 60:
                eod_minute -= 60
                eod_hour += 1

            self.scheduler.add_job(
                self._run_end_of_day,
                CronTrigger(
                    day_of_week="mon-fri",
                    hour=str(eod_hour),
                    minute=str(eod_minute),
                    timezone="America/New_York",
                ),
                id="end_of_day",
                name="End-of-Day Cleanup",
                replace_existing=True,
            )

            # ── 6. Safety net close-all: 3:55 PM ET ─────────────────────
            self.scheduler.add_job(
                self._run_end_of_day,
                CronTrigger(
                    day_of_week="mon-fri",
                    hour="15",
                    minute="55",
                    timezone="America/New_York",
                ),
                id="safety_net_close",
                name="Safety Net Close-All (3:55 PM)",
                replace_existing=True,
            )

        # ── 7. Daily reset: 6:00 AM ET ──────────────────────────────────
        if self._daily_reset_callback:
            self.scheduler.add_job(
                self._run_daily_reset,
                CronTrigger(
                    day_of_week="mon-fri",
                    hour=str(pre_h),
                    minute=str(pre_m),
                    timezone="America/New_York",
                ),
                id="daily_reset",
                name="Daily Reset",
                replace_existing=True,
            )

    # ── Market Clock (Schedule-Based) ────────────────────────────────────

    def is_trading_day(self) -> bool:
        """
        Check if today is a trading day (not weekend, not holiday).

        Uses static NYSE holiday list for accurate holiday detection.
        """
        now_et = datetime.now(ET)

        # Weekend check (fast path)
        if now_et.weekday() >= 5:
            return False

        # Holiday check
        if now_et.date() in NYSE_HOLIDAYS:
            return False

        return True

    # ── Trading Window Helpers ───────────────────────────────────────────

    def is_in_trading_window(self) -> bool:
        """
        Check if we're inside the active trading window.

        Returns True between trading_window_start and trading_window_end ET,
        but only on valid trading days. This is the window where new entries are allowed.
        """
        if not self.is_trading_day():
            return False

        now_et = datetime.now(ET)
        current_time = now_et.time()
        return self._window_start <= current_time < self._window_end

    def is_in_premarket(self) -> bool:
        """
        Check if we're in the pre-market scanning window.

        Returns True between premarket_scan_start and trading_window_start ET,
        but only on valid trading days.
        """
        if not self.is_trading_day():
            return False

        now_et = datetime.now(ET)
        current_time = now_et.time()
        return self._premarket_start <= current_time < self._window_start

    def is_in_any_scan_window(self) -> bool:
        """Check if we're in any scanning window (premarket or active)."""
        return self.is_in_premarket() or self.is_in_trading_window()

    def is_market_open(self) -> bool:
        """
        Check if US stock market is currently open.

        Uses schedule-based check with NYSE holiday list.
        """
        now_et = datetime.now(ET)

        if now_et.weekday() >= 5:
            return False

        if now_et.date() in NYSE_HOLIDAYS:
            return False

        current_time = now_et.time()
        market_open = time(9, 30)
        market_close = time(16, 0)
        return market_open <= current_time < market_close

    def time_until_window_open(self) -> Optional[float]:
        """
        Get seconds until trading window opens.

        Returns None if window is already open or it's not a trading day.
        """
        if not self.is_trading_day():
            return None

        now_et = datetime.now(ET)
        current_time = now_et.time()
        if current_time >= self._window_start:
            return None

        # Calculate seconds until window start
        window_dt = now_et.replace(
            hour=self._window_start.hour,
            minute=self._window_start.minute,
            second=0,
            microsecond=0,
        )
        return (window_dt - now_et).total_seconds()

    # ── Job Runners (with error handling) ────────────────────────────────

    async def _run_press_release_scan(self) -> None:
        """Run press release catalyst scanner with error handling."""
        if self._press_release_scan_callback:
            try:
                await self._press_release_scan_callback()
            except Exception as e:
                logger.error(f"Press release scan error: {e}")

    async def _run_momentum_scan(self) -> None:
        """Run momentum scanner with error handling."""
        if self._momentum_scan_callback:
            try:
                await self._momentum_scan_callback()
            except Exception as e:
                logger.error(f"Momentum scan error: {e}")

    async def _run_position_monitor(self) -> None:
        """Run position monitor with error handling."""
        if self._position_monitor_callback:
            try:
                await self._position_monitor_callback()
            except Exception as e:
                logger.error(f"Position monitor error: {e}")

    async def _run_broker_sync(self) -> None:
        """Run broker sync with error handling."""
        if self._broker_sync_callback:
            try:
                await self._broker_sync_callback()
            except Exception as e:
                logger.error(f"Broker sync error: {e}")

    async def _run_end_of_day(self) -> None:
        """Run end-of-day cleanup with error handling."""
        if self._end_of_day_callback:
            try:
                logger.info("Running end-of-day cleanup...")
                await self._end_of_day_callback()
            except Exception as e:
                logger.error(f"End-of-day cleanup error: {e}")

    async def _run_daily_reset(self) -> None:
        """Run daily reset with error handling."""
        if self._daily_reset_callback:
            try:
                logger.info("Running daily reset...")
                await self._daily_reset_callback()
            except Exception as e:
                logger.error(f"Daily reset error: {e}")

    def set_full_day_trading(self, enabled: bool) -> None:
        """Switch between early window and full day trading at runtime."""
        self.config.full_day_trading = enabled
        if enabled:
            self._window_start = time(9, 30)
            self._window_end = time(15, 55)
            logger.info("[SCHEDULER] Switched to full day trading: 9:30 AM - 3:55 PM ET")
        else:
            self._window_start = self._parse_time(self.config.trading_window_start)
            self._window_end = self._parse_time(self.config.trading_window_end)
            logger.info(
                f"[SCHEDULER] Switched to early window: "
                f"{self.config.trading_window_start}-{self.config.trading_window_end} ET"
            )
        # Reschedule jobs with new window
        if self._is_running:
            self.setup_jobs()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the scheduler."""
        if not self._is_running:
            self.setup_jobs()
            self.scheduler.start()
            self._is_running = True
            logger.info(
                f"Scheduler started | "
                f"Premarket: {self.config.premarket_scan_start} ET | "
                f"Trading: {self.config.trading_window_start}-{self.config.trading_window_end} ET | "
                f"Monitor: every {self.config.position_monitor_interval_seconds}s"
            )

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._is_running:
            self.scheduler.shutdown(wait=True)
            self._is_running = False
            logger.info("Scheduler stopped")

    def pause(self) -> None:
        """Pause all jobs."""
        self.scheduler.pause()
        logger.info("Scheduler paused")

    def resume(self) -> None:
        """Resume all jobs."""
        self.scheduler.resume()
        logger.info("Scheduler resumed")

    def get_jobs(self) -> list[dict]:
        """Get list of scheduled jobs with next run times."""
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
