"""
Bot-specific configuration.

Extends base Settings with bot scheduler and strategy parameters.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from config.settings import Settings


class BotConfig(Settings):
    """
    Trading bot configuration.

    Extends base Settings with:
    - Scheduler settings
    - Strategy parameters
    - State file paths
    - Watchlist symbols
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Scheduler settings
    stock_check_interval_minutes: int = Field(
        default=5,
        ge=1,
        le=60,
        description="How often to check stock signals (during market hours)",
    )
    crypto_check_interval_minutes: int = Field(
        default=10,
        ge=1,
        le=60,
        description="How often to check crypto signals (24/7)",
    )
    enable_crypto_trading: bool = Field(
        default=False,
        description="Enable/disable crypto signal checking and trading",
    )
    position_monitor_interval_minutes: int = Field(
        default=1,
        ge=1,
        le=10,
        description="How often to check position exits",
    )
    broker_sync_interval_minutes: int = Field(
        default=5,
        ge=1,
        le=30,
        description="How often to sync with broker positions",
    )
    watchlist_refresh_interval_minutes: int = Field(
        default=10,
        ge=5,
        le=120,
        description="How often to refresh watchlist from screeners (during market hours)",
    )

    # Stock strategy settings (MACD)
    macd_fast_period: int = Field(
        default=8,
        ge=3,
        le=20,
        description="MACD fast EMA period",
    )
    macd_slow_period: int = Field(
        default=17,
        ge=10,
        le=50,
        description="MACD slow EMA period",
    )
    macd_signal_period: int = Field(
        default=9,
        ge=3,
        le=20,
        description="MACD signal line EMA period",
    )
    stock_timeframe: str = Field(
        default="5Min",
        description="Timeframe for stock signal checking",
    )
    stock_atr_stop_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="ATR multiplier for stock stop-loss",
    )

    # Crypto strategy settings
    crypto_rsi_period: int = Field(
        default=14,
        ge=7,
        le=21,
        description="RSI period for crypto mean reversion",
    )
    crypto_rsi_oversold: int = Field(
        default=30,
        ge=20,
        le=40,
        description="RSI oversold threshold for entry",
    )
    crypto_rsi_exit: int = Field(
        default=50,
        ge=40,
        le=60,
        description="RSI threshold for exit",
    )
    crypto_bb_period: int = Field(
        default=20,
        ge=10,
        le=30,
        description="Bollinger Band period",
    )
    crypto_bb_std: float = Field(
        default=2.0,
        ge=1.5,
        le=3.0,
        description="Bollinger Band standard deviation",
    )
    crypto_atr_stop_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="ATR multiplier for crypto stop-loss",
    )

    # Signal filtering
    min_signal_strength: float = Field(
        default=0.5,
        ge=0.3,
        le=0.9,
        description="Minimum signal strength to act on (0-1)",
    )
    min_risk_reward: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="Minimum risk/reward ratio to accept trade",
    )

    # Watchlist
    stock_watchlist: str = Field(
        default="AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META,AMD,NFLX,SPY",
        description="Comma-separated stock symbols to monitor",
    )
    crypto_watchlist: str = Field(
        default="BTC/USD,ETH/USD,SOL/USD,LINK/USD,AVAX/USD,DOT/USD,XTZ/USD,LTC/USD",
        description="Comma-separated crypto symbols to monitor",
    )

    # State files
    state_dir: str = Field(
        default="state",
        description="Directory for state files",
    )

    @property
    def stock_symbols(self) -> list[str]:
        """Parse stock watchlist into list."""
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
