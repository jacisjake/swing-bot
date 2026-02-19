"""
TradingView-based stock screener for pre-market and active trading sessions.

Uses the tradingview-screener package (no API key required) to find:
- Pre-market gappers with high change%, volume, and low float
- Active session movers with relative volume breakouts

Primary data source for the momentum screener. TradingView provides
real pre-market data starting at 4:00 AM ET.

Data has ~15-minute delay without auth cookies, which is acceptable
for scanner/watchlist building. Actual trade entries are validated against
real-time DXLink bar/quote data before execution.
"""

from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger
from tradingview_screener import Query, col

from src.bot.screener import ScreenerResult


class TradingViewScreener:
    """
    Screen for stocks using TradingView's screener API.

    Two scanning modes:
    - Pre-market (before 7:00 AM ET): Uses premarket_change, premarket_volume,
      premarket_gap fields. These update during extended hours starting ~4 AM.
    - Active session (7:00+ AM ET): Uses regular change, volume fields.
      These update once the market opens.

    Only returns stocks listed on NYSE, NASDAQ, or AMEX — filters out OTC/pink
    sheets that are not tradable.

    Returns list[ScreenerResult] for the downstream MomentumScreener pipeline.
    """

    # Only include stocks on major US exchanges
    US_EXCHANGES = ["NYSE", "NASDAQ", "AMEX"]

    def __init__(self):
        """Initialize TradingView screener (no credentials needed)."""
        self._last_query_time: Optional[datetime] = None
        self._last_results: list[ScreenerResult] = []
        # Cache float data from TradingView results — avoids wasting FMP API calls
        self._float_cache: dict[str, float] = {}

    def get_premarket_gainers(
        self,
        top_n: int = 25,
        min_price: float = 1.0,
        max_price: float = 10.0,
        min_change_pct: float = 5.0,
    ) -> list[ScreenerResult]:
        """
        Get pre-market gappers from TradingView.

        Uses premarket_change, premarket_volume, premarket_gap fields
        which are available during extended hours (4:00 AM+).

        Args:
            top_n: Maximum results to return
            min_price: Minimum stock price (previous close)
            max_price: Maximum stock price
            min_change_pct: Minimum pre-market change %

        Returns:
            List of ScreenerResult sorted by pre-market change %
        """
        try:
            total_count, df = (
                Query()
                .select(
                    "name", "close", "premarket_change", "premarket_volume",
                    "premarket_gap", "float_shares_outstanding", "volume",
                    "relative_volume_10d_calc", "exchange",
                )
                .where(
                    col("premarket_change") > min_change_pct,
                    col("close").between(min_price, max_price),
                    col("exchange").isin(self.US_EXCHANGES),
                )
                .order_by("premarket_change", ascending=False)
                .limit(top_n)
                .set_markets("america")
                .get_scanner_data()
            )

            results = self._dataframe_to_results(
                df,
                change_col="premarket_change",
                volume_col="premarket_volume",
                source="tv_premarket",
            )

            logger.info(
                f"[TV-SCREEN] Pre-market gainers: {len(results)} results "
                f"(total matching: {total_count})"
            )
            for r in results[:5]:
                float_str = ""
                if r.symbol in self._float_cache:
                    float_m = self._float_cache[r.symbol] / 1_000_000
                    float_str = f", float={float_m:.1f}M"
                logger.info(
                    f"  [TV] {r.symbol}: {r.change_pct:+.1f}% @ ${r.price:.2f} "
                    f"vol={r.volume or 0:,}{float_str}"
                )

            self._last_query_time = datetime.now()
            self._last_results = results
            return results

        except Exception as e:
            logger.error(f"[TV-SCREEN] Pre-market query failed: {e}")
            return []

    def get_active_gainers(
        self,
        top_n: int = 25,
        min_price: float = 1.0,
        max_price: float = 10.0,
        min_change_pct: float = 5.0,
    ) -> list[ScreenerResult]:
        """
        Get active session gainers from TradingView.

        Uses regular change, volume fields which update after market open.

        Args:
            top_n: Maximum results to return
            min_price: Minimum stock price
            max_price: Maximum stock price
            min_change_pct: Minimum intraday change %

        Returns:
            List of ScreenerResult sorted by change %
        """
        try:
            total_count, df = (
                Query()
                .select(
                    "name", "close", "change", "volume",
                    "float_shares_outstanding",
                    "relative_volume_10d_calc", "exchange",
                )
                .where(
                    col("change") > min_change_pct,
                    col("close").between(min_price, max_price),
                    col("exchange").isin(self.US_EXCHANGES),
                )
                .order_by("change", ascending=False)
                .limit(top_n)
                .set_markets("america")
                .get_scanner_data()
            )

            results = self._dataframe_to_results(
                df,
                change_col="change",
                volume_col="volume",
                source="tv_active",
            )

            logger.info(
                f"[TV-SCREEN] Active gainers: {len(results)} results "
                f"(total matching: {total_count})"
            )
            for r in results[:5]:
                logger.info(
                    f"  [TV] {r.symbol}: {r.change_pct:+.1f}% @ ${r.price:.2f} "
                    f"vol={r.volume or 0:,}"
                )

            self._last_query_time = datetime.now()
            self._last_results = results
            return results

        except Exception as e:
            logger.error(f"[TV-SCREEN] Active query failed: {e}")
            return []

    def get_float_cache(self) -> dict[str, float]:
        """
        Get cached float data extracted from TradingView results.

        Returns:
            Dict mapping symbol -> float_shares_outstanding
        """
        return self._float_cache.copy()

    def _dataframe_to_results(
        self,
        df: pd.DataFrame,
        change_col: str,
        volume_col: str,
        source: str,
    ) -> list[ScreenerResult]:
        """
        Convert TradingView DataFrame to list of ScreenerResult.

        Handles:
        - Ticker format conversion (NASDAQ:AAPL -> AAPL)
        - NaN/None handling for optional fields
        - Float data extraction (cached for FloatDataProvider)

        Args:
            df: TradingView result DataFrame
            change_col: Column name for change % (premarket_change or change)
            volume_col: Column name for volume (premarket_volume or volume)
            source: Source tag for ScreenerResult

        Returns:
            List of ScreenerResult
        """
        if df is None or df.empty:
            return []

        results = []
        for _, row in df.iterrows():
            # Extract symbol from "EXCHANGE:SYMBOL" format
            ticker_raw = str(row.get("ticker", ""))
            symbol = ticker_raw.split(":")[-1] if ":" in ticker_raw else ticker_raw

            if not symbol or len(symbol) > 5:
                continue

            # Skip warrants and rights
            if symbol.endswith("W") or ".WS" in symbol:
                continue

            # Extract values with NaN handling
            price = _safe_float(row.get("close"))
            change_pct = _safe_float(row.get(change_col))
            volume = _safe_int(row.get(volume_col))
            rel_vol = _safe_float(row.get("relative_volume_10d_calc"))
            float_shares = _safe_float(row.get("float_shares_outstanding"))

            if price is None or price <= 0:
                continue

            # Cache float data for later use by FloatDataProvider
            if float_shares is not None and float_shares > 0:
                self._float_cache[symbol] = float_shares

            results.append(ScreenerResult(
                symbol=symbol,
                price=price,
                change_pct=change_pct or 0.0,
                volume=volume,
                volume_ratio=rel_vol,
                source=source,
            ))

        return results

    @property
    def last_query_time(self) -> Optional[datetime]:
        """When the last query was performed."""
        return self._last_query_time


def _safe_float(val) -> Optional[float]:
    """Convert a value to float, returning None for NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        if pd.isna(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> Optional[int]:
    """Convert a value to int, returning None for NaN/None."""
    f = _safe_float(val)
    if f is None:
        return None
    return int(f)
