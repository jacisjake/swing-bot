"""
Bot-specific configuration for momentum day trading.

Extends base Settings with scanner, strategy, and scheduler parameters.
Targeting $1-$10 low-float stocks (prefer $2+) on 5-min bars with pullback entries.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from config.settings import Settings


class BotConfig(Settings):
    """
    Momentum day trading bot configuration.

    Strategy: Ross Cameron-style pullback entries on low-float momentum stocks.
    Timeframe: 5-minute bars during 7:00-10:00 AM ET window.
    Goal: One high-quality trade per day, 10% account growth.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Scheduler Settings ──────────────────────────────────────────────

    stock_check_interval_minutes: int = Field(
        default=1,
        ge=1,
        le=60,
        description="How often to run momentum scan during trading window",
    )
    position_monitor_interval_seconds: int = Field(
        default=30,
        ge=10,
        le=120,
        description="How often to check position exits (seconds)",
    )
    broker_sync_interval_minutes: int = Field(
        default=1,
        ge=1,
        le=30,
        description="How often to sync with broker positions",
    )
    scanner_refresh_interval_minutes: int = Field(
        default=2,
        ge=1,
        le=30,
        description="How often to refresh scanner results during trading window",
    )

    # ── Trading Window (Eastern Time) ───────────────────────────────────

    premarket_scan_start: str = Field(
        default="06:00",
        description="When to start pre-market scanning (ET, HH:MM)",
    )
    trading_window_start: str = Field(
        default="07:00",
        description="Start of active trading window (ET, HH:MM)",
    )
    trading_window_end: str = Field(
        default="10:00",
        description="End of active trading window (ET, HH:MM)",
    )

    # ── Momentum Scanner Settings ───────────────────────────────────────
    # Ross Cameron's 5 pillars: price, float, relative volume, change%, catalyst

    scanner_min_price: float = Field(
        default=1.0,
        ge=0.50,
        le=20.0,
        description="Minimum stock price for scanner ($1 floor, prefer $2+)",
    )
    scanner_preferred_min_price: float = Field(
        default=2.0,
        ge=1.0,
        le=20.0,
        description="Preferred minimum price — stocks above this get priority weighting",
    )
    scanner_max_price: float = Field(
        default=10.0,
        ge=2.0,
        le=50.0,
        description="Maximum stock price for scanner ($10 sweet spot)",
    )
    scanner_min_change_pct: float = Field(
        default=10.0,
        ge=5.0,
        le=50.0,
        description="Minimum % gain today to qualify (already moving)",
    )
    scanner_min_relative_volume: float = Field(
        default=5.0,
        ge=1.5,
        le=20.0,
        description="Minimum relative volume (today vs 20-day avg, 5x = strong interest)",
    )
    scanner_max_float_millions: float = Field(
        default=20.0,
        ge=1.0,
        le=100.0,
        description="Maximum float in millions (low supply = bigger moves)",
    )
    scanner_enable_float_filter: bool = Field(
        default=True,
        description="Enable float filtering (requires FMP API key or yfinance)",
    )
    scanner_top_n: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Number of gainers to fetch from Alpaca screener API",
    )

    # ── Catalyst / News Settings (5th Pillar) ────────────────────────────

    scanner_enable_news_check: bool = Field(
        default=True,
        description="Check for news catalysts during enrichment (uses Alpaca News API)",
    )
    scanner_news_lookback_hours: int = Field(
        default=12,
        ge=1,
        le=48,
        description="Hours to look back for news (12h covers overnight + pre-market)",
    )
    scanner_news_max_articles: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Max articles to fetch per symbol (minimize API usage)",
    )

    # ── TradingView Screener (Primary Scanner) ──────────────────────────

    use_tradingview_screener: bool = Field(
        default=True,
        description="Use TradingView as primary screener (falls back to Alpaca on failure)",
    )

    # ── Press Release Scanner (Pre-Market Catalyst Detection) ────────────

    enable_press_release_scanner: bool = Field(
        default=True,
        description="Enable pre-market press release scanning via RSS feeds + FMP",
    )
    press_release_scan_start: str = Field(
        default="04:00",
        description="When to start scanning press releases (ET, HH:MM). Earlier than premarket scan.",
    )
    press_release_scan_interval_minutes: int = Field(
        default=5,
        ge=2,
        le=30,
        description="How often to poll RSS feeds during pre-market (minutes)",
    )
    press_release_lookback_hours: int = Field(
        default=16,
        ge=4,
        le=48,
        description="Hours to look back for press releases (16h covers overnight + previous evening)",
    )

    # ── MACD Strategy Parameters ────────────────────────────────────────
    # Standard 12/26/9 MACD on 5-min bars

    macd_fast_period: int = Field(
        default=8,
        ge=3,
        le=20,
        description="MACD fast EMA period (8 for faster response on volatile stocks)",
    )
    macd_slow_period: int = Field(
        default=21,
        ge=10,
        le=50,
        description="MACD slow EMA period (21 converges faster than 26)",
    )
    macd_signal_period: int = Field(
        default=5,
        ge=1,
        le=20,
        description="MACD signal line EMA period (5 for quicker crossovers)",
    )
    stock_timeframe: str = Field(
        default="5Min",
        description="Entry timeframe for signals (5-min bars for day trading)",
    )
    stock_atr_stop_multiplier: float = Field(
        default=1.5,
        ge=0.5,
        le=4.0,
        description="ATR multiplier for stop-loss (tighter for day trading)",
    )
    atr_period: int = Field(
        default=14,
        ge=5,
        le=30,
        description="ATR calculation period",
    )

    # ── Pullback Pattern Parameters ─────────────────────────────────────

    pullback_min_candles: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Minimum candles in pullback before entry",
    )
    pullback_max_candles: int = Field(
        default=15,
        ge=3,
        le=25,
        description="Maximum candles in pullback (volatile stocks consolidate 10-15 candles)",
    )
    pullback_max_retracement: float = Field(
        default=0.65,
        ge=0.20,
        le=0.80,
        description="Maximum pullback retracement of surge (65% for volatile low-float stocks)",
    )
    risk_reward_target: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Risk/reward ratio for take-profit target",
    )

    # ── Signal Filtering ────────────────────────────────────────────────

    min_signal_strength: float = Field(
        default=0.5,
        ge=0.3,
        le=0.9,
        description="Minimum signal strength to act on (0-1)",
    )
    min_risk_reward: float = Field(
        default=1.0,
        ge=0.5,
        le=5.0,
        description="Minimum risk/reward ratio to accept trade",
    )
    allow_short_selling: bool = Field(
        default=False,
        description="Allow short selling (not used in momentum strategy)",
    )

    # ── Day Trading Risk Management ─────────────────────────────────────

    max_daily_trades: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Maximum trades per day (1 = Ross Cameron's cash account approach)",
    )
    daily_profit_target_pct: float = Field(
        default=0.10,
        ge=0.02,
        le=0.30,
        description="Daily profit target (10% of account)",
    )
    daily_loss_limit_pct: float = Field(
        default=0.10,
        ge=0.02,
        le=0.20,
        description="Maximum daily loss before halt (10% of account)",
    )
    max_position_pct_of_buying_power: float = Field(
        default=0.90,
        ge=0.25,
        le=1.0,
        description="Max % of buying power to use per trade (cash account style)",
    )

    # ── Crypto (disabled for day trading focus) ─────────────────────────

    enable_crypto_trading: bool = Field(
        default=False,
        description="Enable crypto trading (disabled for momentum day trading)",
    )
    crypto_watchlist: str = Field(
        default="BTC/USD,ETH/USD,SOL/USD",
        description="Crypto symbols (not used when disabled)",
    )

    # ── Watchlist (scanner-driven, no static list) ──────────────────────

    stock_watchlist: str = Field(
        default="",
        description="Static stock watchlist (empty = fully scanner-driven)",
    )

    # ── WebSocket Settings ────────────────────────────────────────────

    ws_reconnect_max_seconds: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Max reconnect backoff in seconds for WebSocket",
    )
    ws_heartbeat_seconds: int = Field(
        default=30,
        ge=10,
        le=60,
        description="WebSocket ping interval in seconds",
    )

    # ── State Files ─────────────────────────────────────────────────────

    state_dir: str = Field(
        default="state",
        description="Directory for state files",
    )

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def stock_symbols(self) -> list[str]:
        """Parse stock watchlist into list (may be empty if scanner-driven)."""
        return [s.strip().upper() for s in self.stock_watchlist.split(",") if s.strip()]

    @property
    def crypto_symbols(self) -> list[str]:
        """Parse crypto watchlist into list."""
        return [s.strip().upper() for s in self.crypto_watchlist.split(",") if s.strip()]

    @property
    def state_path(self) -> Path:
        """Get state directory path."""
        return Path(self.state_dir)

    @property
    def bot_state_file(self) -> Path:
        """Get bot state file path."""
        return self.state_path / "bot_state.json"


def get_bot_config() -> BotConfig:
    """Get bot configuration instance."""
    return BotConfig()
