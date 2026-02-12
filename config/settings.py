"""
Configuration management using Pydantic Settings.
All settings loaded from environment variables with validation.
"""

from enum import Enum
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Alpaca credentials
    alpaca_api_key: str = Field(..., min_length=10)
    alpaca_secret_key: str = Field(..., min_length=10)
    alpaca_data_feed: Literal["iex", "sip"] = Field(default="iex")

    # Trading mode
    trading_mode: TradingMode = Field(default=TradingMode.PAPER)
    enable_extended_hours: bool = Field(default=True)

    # Float data provider (Financial Modeling Prep)
    fmp_api_key: Optional[str] = Field(
        default=None,
        description="Financial Modeling Prep API key for float data (free tier: 250 req/day)",
    )

    # Risk management
    max_position_risk_pct: float = Field(default=0.02, ge=0.001, le=0.10)
    max_portfolio_risk_pct: float = Field(default=0.10, ge=0.01, le=0.50)
    max_positions: int = Field(default=1, ge=1, le=20)
    max_drawdown_pct: float = Field(default=0.15, ge=0.05, le=0.50)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    @field_validator("trading_mode", mode="before")
    @classmethod
    def validate_trading_mode(cls, v: str) -> TradingMode:
        if isinstance(v, str):
            return TradingMode(v.lower())
        return v

    @property
    def is_paper(self) -> bool:
        return self.trading_mode == TradingMode.PAPER

    @property
    def is_live(self) -> bool:
        return self.trading_mode == TradingMode.LIVE


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance. Validates on first call."""
    return Settings()


# Convenience export
settings = get_settings()
