"""
Momentum day trading scheduler.

Manages scheduled jobs for the trading day:
- Pre-market scanning (6:00-7:00 AM ET)
- Active momentum scanning + signal generation (7:00-10:00 AM ET)
- Position monitoring (30-second intervals when positions open)
- End-of-day cleanup (close positions, cancel orders)
- Safety net close-all (3:55 PM ET)
- Daily reset (6:00 AM ET)

Uses APScheduler with Eastern Time for all trading jobs.
Uses Alpaca market clock API for accurate market status (holidays, early closes).
"""

from datetime import datetime, time
from typing import Callable, Optional, TYPE_CHECKING

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from src.bot.config import BotConfig

if TYPE_CHECKING:
    from src.core.alpaca_client import AlpacaClient

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

    def __init__(self, config: BotConfig, client: Optional["AlpacaClient"] = None):
        """
        Initialize scheduler.

        Args:
            config: Bot configuration
            client: Alpaca client for market clock API (accurate holidays/early closes)
        """
        self.config = config
        self.client = client
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

        # Market clock cache (refreshed every 5 minutes to avoid excessive API calls)
        self._market_clock_cache: Optional[dict] = None
        self._market_clock_cache_time: Optional[datetime] = None
        self._CLOCK_CACHE_TTL_SECONDS = 300  # 5 minutes

        # Parse trading window times from config
        self._premarket_start = self._parse_time(config.premarket_scan_start)
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

        # ── 0. Press release scan: 4:00-6:59 AM ET, every 5 minutes ──────
        if self._press_release_scan_callback:
            pr_start = self._parse_time(self.config.press_release_scan_start)
            pr_interval = self.config.press_release_scan_interval_minutes

            # Run from press_release_scan_start until trading window starts
            # e.g., 4:00-6:59 AM ET
            pr_hours = f"{pr_start.hour}-{win_start_h - 1}" if win_start_h > pr_start.hour else str(pr_start.hour)

            self.scheduler.add_job(
                self._run_press_release_scan,
                CronTrigger(
                    day_of_week="mon-fri",
                    hour=pr_hours,
                    minute=f"*/{pr_interval}",
                    timezone="America/New_York",
                ),
                id="press_release_scan",
                name="Press Release Catalyst Scan",
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

    # ── Market Clock (Alpaca API) ────────────────────────────────────────

    def _get_market_clock(self) -> Optional[dict]:
        """
        Get market clock from Alpaca API with caching.

        Returns dict with:
            is_open: bool - whether market is currently open
            next_open: datetime - next market open time
            next_close: datetime - next market close time

        Uses a 5-minute cache to avoid excessive API calls.
        Falls back to simple time-based checks if API is unavailable.
        """
        if self.client is None:
            return None

        now = datetime.now(ET)

        # Return cached value if fresh
        if (
            self._market_clock_cache is not None
            and self._market_clock_cache_time is not None
            and (now - self._market_clock_cache_time).total_seconds() < self._CLOCK_CACHE_TTL_SECONDS
        ):
            return self._market_clock_cache

        try:
            clock = self.client.trading.get_clock()
            self._market_clock_cache = {
                "is_open": clock.is_open,
                "next_open": clock.next_open,
                "next_close": clock.next_close,
            }
            self._market_clock_cache_time = now
            return self._market_clock_cache
        except Exception as e:
            logger.warning(f"Market clock API error (using time fallback): {e}")
            return None

    def is_trading_day(self) -> bool:
        """
        Check if today is a trading day (not weekend, not holiday).

        Uses Alpaca market clock to accurately detect holidays.
        Falls back to simple weekday check if API unavailable.
        """
        now_et = datetime.now(ET)

        # Weekend check (fast path, no API needed)
        if now_et.weekday() >= 5:
            return False

        # Use Alpaca market clock for holiday detection
        clock = self._get_market_clock()
        if clock is not None:
            # If the market is open, it's definitely a trading day
            if clock["is_open"]:
                return True

            # If market is closed, check if next_open is today
            # (market might just be in pre/post-market hours on a valid trading day)
            next_open = clock["next_open"]
            if next_open is not None:
                # Convert to ET for comparison
                if hasattr(next_open, "astimezone"):
                    next_open_et = next_open.astimezone(ET)
                else:
                    next_open_et = next_open

                # If next open is today, it's a trading day (just not market hours yet)
                if next_open_et.date() == now_et.date():
                    return True

                # If next open is tomorrow (or later), today might be a holiday
                # But only if we're past when the market should have opened (9:30 AM)
                if now_et.time() >= time(9, 30) and next_open_et.date() > now_et.date():
                    logger.info(
                        f"Market holiday detected - next open: {next_open_et.strftime('%Y-%m-%d %H:%M')} ET"
                    )
                    return False

        # Fallback: assume weekdays are trading days
        return True

    # ── Trading Window Helpers ───────────────────────────────────────────

    def is_in_trading_window(self) -> bool:
        """
        Check if we're inside the active trading window.

        Returns True between trading_window_start and trading_window_end ET,
        but only on valid trading days (uses Alpaca market clock for holidays).
        This is the window where new entries are allowed.
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

        Uses Alpaca market clock API for accurate status
        (handles holidays, early closes, etc.).
        Falls back to simple time check if API unavailable.
        """
        clock = self._get_market_clock()
        if clock is not None:
            return clock["is_open"]

        # Fallback: simple time-based check
        now_et = datetime.now(ET)
        if now_et.weekday() >= 5:
            return False

        current_time = now_et.time()
        market_open = time(9, 30)
        market_close = time(16, 0)
        return market_open <= current_time < market_close

    def get_next_market_close(self) -> Optional[datetime]:
        """
        Get the next market close time from Alpaca API.

        Useful for early close days (e.g., 1:00 PM on day before holidays).
        Returns None if API unavailable.
        """
        clock = self._get_market_clock()
        if clock is not None:
            return clock.get("next_close")
        return None

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
