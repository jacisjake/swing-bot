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

    # tastytrade credentials (legacy username/password OR OAuth)
    tt_username: Optional[str] = Field(default=None)
    tt_password: Optional[str] = Field(default=None)
    tt_account_number: str = Field(..., min_length=1)

    # tastytrade OAuth2 credentials
    tt_client_id: Optional[str] = Field(default=None)
    tt_client_secret: Optional[str] = Field(default=None)
    tt_refresh_token: Optional[str] = Field(default=None)
    tt_oauth_redirect_uri: Optional[str] = Field(
        default=None,
        description="OAuth redirect URI (must match developer portal). "
        "Set this when behind a reverse proxy.",
    )

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

    @property
    def has_oauth(self) -> bool:
        """True if OAuth credentials are configured (client_secret + refresh_token)."""
        return bool(self.tt_client_secret and self.tt_refresh_token)

    @property
    def has_legacy_auth(self) -> bool:
        """True if legacy username/password credentials are configured."""
        return bool(self.tt_username and self.tt_password)

    @property
    def can_authenticate(self) -> bool:
        """True if any auth method is available."""
        return self.has_oauth or self.has_legacy_auth


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance. Validates on first call."""
    return Settings()


# Convenience export
settings = get_settings()
