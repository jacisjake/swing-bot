"""
Float data provider for momentum scanner.

Fetches share float data to identify low-supply stocks that can make big moves.
Low float (< 5-20M shares) = less supply = bigger price impact from demand.

Sources:
- Primary: Financial Modeling Prep (FMP) free API (250 req/day)
- Fallback: yfinance package
- Cache: In-memory with 24h TTL to minimize API calls
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class FloatData:
    """Share float information for a stock."""

    float_shares: Optional[float] = None  # Shares available to trade
    shares_outstanding: Optional[float] = None  # Total shares issued
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

    @property
    def float_millions(self) -> Optional[float]:
        """Float in millions of shares."""
        if self.float_shares is None:
            return None
        return self.float_shares / 1_000_000

    @property
    def is_low_float(self) -> bool:
        """Check if stock has low float (< 20M shares)."""
        if self.float_shares is None:
            return False
        return self.float_shares < 20_000_000

    @property
    def is_very_low_float(self) -> bool:
        """Check if stock has very low float (< 5M shares, best for cold markets)."""
        if self.float_shares is None:
            return False
        return self.float_shares < 5_000_000


class FloatDataProvider:
    """
    Fetch and cache share float data from free APIs.

    Usage:
        provider = FloatDataProvider(fmp_api_key="your_key")
        data = provider.get_float("AAPL")
        if data and data.is_low_float:
            print(f"Low float: {data.float_millions:.1f}M shares")
    """

    CACHE_TTL_HOURS = 24  # Float doesn't change frequently

    def __init__(self, fmp_api_key: Optional[str] = None):
        """
        Initialize float data provider.

        Args:
            fmp_api_key: Financial Modeling Prep API key (free tier: 250 req/day)
        """
        self.fmp_api_key = fmp_api_key
        self._cache: dict[str, tuple[FloatData, datetime]] = {}  # symbol -> (data, expire_time)
        self._daily_request_count = 0
        self._request_count_date = datetime.now().date()

    def get_float(self, symbol: str) -> Optional[FloatData]:
        """
        Get float data for a symbol, checking cache first.

        Args:
            symbol: Stock ticker (e.g., "AAPL")

        Returns:
            FloatData if available, None if all sources fail
        """
        # Check cache
        cached = self._get_cached(symbol)
        if cached is not None:
            return cached

        # Try FMP first (more reliable)
        data = self._fetch_from_fmp(symbol)
        if data is not None:
            self._set_cached(symbol, data)
            return data

        # Fallback to yfinance
        data = self._fetch_from_yfinance(symbol)
        if data is not None:
            self._set_cached(symbol, data)
            return data

        logger.warning(f"Could not fetch float data for {symbol} from any source")
        return None

    def set_float_hint(self, symbol: str, float_shares: float) -> None:
        """
        Pre-populate cache with float data from an external source (e.g., TradingView).

        Only used if the symbol is not already cached. This avoids
        wasting FMP API calls when TradingView already provided float data.

        Args:
            symbol: Stock ticker
            float_shares: Float shares from TradingView's float_shares_outstanding
        """
        if self._get_cached(symbol) is None and float_shares and float_shares > 0:
            data = FloatData(float_shares=float_shares)
            self._set_cached(symbol, data)
            logger.debug(
                f"[FLOAT] Pre-cached {symbol} from TradingView: "
                f"{data.float_millions:.1f}M"
            )

    def get_float_batch(self, symbols: list[str]) -> dict[str, Optional[FloatData]]:
        """
        Get float data for multiple symbols.

        Args:
            symbols: List of stock tickers

        Returns:
            Dict mapping symbol -> FloatData (or None if unavailable)
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.get_float(symbol)
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        return results

    def _get_cached(self, symbol: str) -> Optional[FloatData]:
        """Get cached float data if not expired."""
        if symbol in self._cache:
            data, expire_time = self._cache[symbol]
            if datetime.now() < expire_time:
                return data
            else:
                del self._cache[symbol]
        return None

    def _set_cached(self, symbol: str, data: FloatData) -> None:
        """Cache float data with TTL."""
        expire_time = datetime.now() + timedelta(hours=self.CACHE_TTL_HOURS)
        self._cache[symbol] = (data, expire_time)

    def _check_daily_limit(self) -> bool:
        """Check if we're within FMP daily request limit."""
        today = datetime.now().date()
        if today != self._request_count_date:
            self._daily_request_count = 0
            self._request_count_date = today

        # FMP free tier: 250 requests/day, leave buffer
        return self._daily_request_count < 200

    def _fetch_from_fmp(self, symbol: str) -> Optional[FloatData]:
        """
        Fetch float data from Financial Modeling Prep API.

        Free tier: 250 requests/day
        Endpoint: GET /api/v3/profile/{symbol}
        Returns: floatShares, mktCap, sharesOutstanding, etc.
        """
        if not self.fmp_api_key:
            return None

        if not self._check_daily_limit():
            logger.warning("FMP daily request limit approaching, skipping")
            return None

        try:
            url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
            params = {"apikey": self.fmp_api_key}

            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()

            self._daily_request_count += 1

            data = response.json()
            if not data or not isinstance(data, list) or len(data) == 0:
                return None

            profile = data[0]
            float_shares = profile.get("floatShares")
            shares_outstanding = profile.get("sharesOutstanding")

            if float_shares is None and shares_outstanding is None:
                return None

            result = FloatData(
                float_shares=float(float_shares) if float_shares else None,
                shares_outstanding=float(shares_outstanding) if shares_outstanding else None,
            )

            logger.debug(
                f"[FMP] {symbol}: float={result.float_millions:.1f}M"
                if result.float_millions
                else f"[FMP] {symbol}: no float data"
            )
            return result

        except requests.exceptions.Timeout:
            logger.debug(f"FMP timeout for {symbol}")
            return None
        except Exception as e:
            logger.debug(f"FMP error for {symbol}: {e}")
            return None

    def _fetch_from_yfinance(self, symbol: str) -> Optional[FloatData]:
        """
        Fallback: fetch float data from yfinance.

        Free but can be slow and rate-limited.
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info

            float_shares = info.get("floatShares")
            shares_outstanding = info.get("sharesOutstanding")

            if float_shares is None and shares_outstanding is None:
                return None

            result = FloatData(
                float_shares=float(float_shares) if float_shares else None,
                shares_outstanding=float(shares_outstanding) if shares_outstanding else None,
            )

            logger.debug(
                f"[yfinance] {symbol}: float={result.float_millions:.1f}M"
                if result.float_millions
                else f"[yfinance] {symbol}: no float data"
            )
            return result

        except ImportError:
            logger.debug("yfinance not installed, skipping fallback")
            return None
        except Exception as e:
            logger.debug(f"yfinance error for {symbol}: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear all cached float data."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Number of cached entries."""
        return len(self._cache)

    @property
    def daily_requests_used(self) -> int:
        """FMP API requests used today."""
        return self._daily_request_count
