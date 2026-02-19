"""
Stock screener for momentum day trading.

Uses TradingView screener API to find:
- Market movers (top gainers/losers)
- Volume breakout candidates
- Momentum candidates (low-float, high relative volume, big % gainers)

TradingView provides real pre-market data starting at 4:00 AM ET.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pytz
from loguru import logger


@dataclass
class ScreenerResult:
    """Result from stock screening."""
    symbol: str
    price: float
    change_pct: float
    volume: Optional[int] = None
    avg_volume: Optional[float] = None
    volume_ratio: Optional[float] = None
    source: str = "unknown"
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "change_pct": self.change_pct,
            "volume": self.volume,
            "avg_volume": self.avg_volume,
            "volume_ratio": self.volume_ratio,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class MomentumCandidate(ScreenerResult):
    """
    Enhanced screener result for momentum day trading.

    Extends ScreenerResult with float data, relative volume, and filter status.
    Used by the MomentumScreener to find Ross Cameron-style setups.
    """
    float_shares: Optional[float] = None        # Shares in float
    relative_volume: Optional[float] = None      # Today vol / 20-day avg vol
    gap_pct: Optional[float] = None              # Gap % from previous close
    high_of_day: Optional[float] = None
    low_of_day: Optional[float] = None
    prev_close: Optional[float] = None
    # Catalyst / news data (5th pillar)
    has_catalyst: bool = False                    # Whether recent news found
    news_headline: Optional[str] = None           # Top headline (if any)
    news_count: int = 0                           # Number of recent articles
    news_source: Optional[str] = None             # Source of top headline
    passes_all_filters: bool = False
    filter_failures: list = field(default_factory=list)

    @property
    def float_millions(self) -> Optional[float]:
        """Float in millions of shares."""
        if self.float_shares is None:
            return None
        return self.float_shares / 1_000_000

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "float_shares": self.float_shares,
            "float_millions": self.float_millions,
            "relative_volume": self.relative_volume,
            "gap_pct": self.gap_pct,
            "high_of_day": self.high_of_day,
            "low_of_day": self.low_of_day,
            "prev_close": self.prev_close,
            "has_catalyst": self.has_catalyst,
            "news_headline": self.news_headline,
            "news_count": self.news_count,
            "news_source": self.news_source,
            "passes_all_filters": self.passes_all_filters,
            "filter_failures": self.filter_failures,
        })
        return base


class MomentumScreener:
    """
    Ross Cameron-style momentum scanner for day trading.

    Scans for stocks meeting the 5 pillars of stock selection:
    1. Price: $1-$10 (low-priced, big % move potential; prefer $2+)
    2. Float: Under 5-20M shares (low supply)
    3. Relative Volume: 5x+ above 20-day average (high demand)
    4. Change: Already up 10%+ today (momentum confirmed)
    5. Catalyst: Breaking news (detected by the move itself)

    Pipeline:
    1. Fetch top gainers from TradingView screener (initial universe)
    2. Enrich each with snapshot data and relative volume calculation
    3. Fetch float data from FMP
    4. Apply filter chain
    5. Return top candidates sorted by relative volume
    """

    def __init__(self, float_provider=None, client=None,
                 news_enabled: bool = True, news_lookback_hours: int = 12,
                 news_max_articles: int = 5,
                 tv_screener=None, use_tradingview: bool = True):
        """
        Initialize momentum screener.

        Args:
            float_provider: FloatDataProvider instance for float data
            client: TastytradeClient instance for bar data and news
            news_enabled: Whether to check for news catalysts
            news_lookback_hours: How far back to look for news
            news_max_articles: Max articles to fetch per symbol
            tv_screener: TradingViewScreener instance (optional)
            use_tradingview: Whether to use TradingView as primary source
        """
        self.tv_screener = tv_screener
        self.use_tradingview = use_tradingview
        self.float_provider = float_provider
        self.client = client
        self.news_enabled = news_enabled
        self.news_lookback_hours = news_lookback_hours
        self.news_max_articles = news_max_articles
        self._last_scan_results: list[MomentumCandidate] = []
        self._last_scan_time: Optional[datetime] = None

    def scan(
        self,
        min_price: float = 1.0,
        max_price: float = 10.0,
        preferred_min_price: float = 2.0,
        min_change_pct: float = 10.0,
        min_relative_volume: float = 5.0,
        min_float_millions: float = 0.5,
        enable_float_filter: bool = True,
        top_n: int = 20,
        max_results: int = 5,
    ) -> list[MomentumCandidate]:
        """
        Run the full momentum scan pipeline.

        Args:
            min_price: Hard minimum stock price ($1 floor)
            max_price: Maximum stock price ($10)
            preferred_min_price: Preferred minimum — stocks above this get priority
                in sorting. $1-$2 stocks are included but ranked lower.
            min_change_pct: Minimum % gain today (10%)
            min_relative_volume: Minimum relative volume vs 20-day avg (5x)
            min_float_millions: Minimum float in millions (0.5M floor for liquidity)
            enable_float_filter: Whether to filter by float
            top_n: Number of gainers to fetch
            max_results: Maximum candidates to return

        Returns:
            List of MomentumCandidate sorted by relative volume (best first),
            with $2+ stocks weighted higher than $1-$2 stocks.
        """
        logger.info(f"[SCANNER] Running momentum scan...")

        # Step 1: Get initial universe from TradingView
        raw_gainers = self._get_initial_universe(
            top_n=top_n,
            min_price=min_price,
            max_price=max_price,
            min_change_pct=min_change_pct,
        )
        logger.info(f"[SCANNER] Raw gainers: {len(raw_gainers)}")

        if not raw_gainers:
            logger.warning("[SCANNER] No gainers returned from any source")
            self._last_scan_results = []
            self._last_scan_time = datetime.now()
            return []

        # Step 2: Quick price filter first (cheapest filter)
        price_filtered = [
            g for g in raw_gainers
            if min_price <= g.price <= max_price and g.price > 0
        ]
        logger.info(
            f"[SCANNER] After price filter (${min_price}-${max_price}): "
            f"{len(price_filtered)} / {len(raw_gainers)}"
        )

        # Step 3: Change % filter
        change_filtered = [
            g for g in price_filtered
            if g.change_pct >= min_change_pct
        ]
        logger.info(
            f"[SCANNER] After change filter (>={min_change_pct}%): "
            f"{len(change_filtered)} / {len(price_filtered)}"
        )

        if not change_filtered:
            logger.info("[SCANNER] No candidates after price + change filters")
            self._last_scan_results = []
            self._last_scan_time = datetime.now()
            return []

        # Step 4: Enrich each candidate with relative volume and float data
        candidates = []
        for gainer in change_filtered:
            candidate = self._enrich_candidate(
                gainer, enable_float_filter=enable_float_filter
            )
            if candidate is not None:
                candidates.append(candidate)

        logger.info(f"[SCANNER] Enriched candidates: {len(candidates)}")

        # Step 5: Apply remaining filters
        final = self._apply_filters(
            candidates,
            min_relative_volume=min_relative_volume,
            min_float_millions=min_float_millions,
            enable_float_filter=enable_float_filter,
        )

        # Step 6: Sort by relative volume with price preference weighting
        # Stocks >= preferred_min_price get full relVol score
        # Stocks below preferred_min_price get 50% weight (still included, just ranked lower)
        def _sort_key(c):
            rv = c.relative_volume or 0
            if c.price < preferred_min_price:
                rv *= 0.5  # Discount sub-$2 stocks in ranking
            return rv

        final.sort(key=_sort_key, reverse=True)
        final = final[:max_results]

        # Log results
        for c in final:
            float_str = f"{c.float_millions:.1f}M" if c.float_millions else "N/A"
            news_str = f" NEWS: {c.news_headline[:50]}..." if c.has_catalyst and c.news_headline else ""
            logger.info(
                f"  [SCANNER] {c.symbol}: ${c.price:.2f} "
                f"+{c.change_pct:.1f}% "
                f"relVol={c.relative_volume:.1f}x "
                f"float={float_str} "
                f"{'PASS' if c.passes_all_filters else 'PARTIAL'}"
                f"{news_str}"
            )

        self._last_scan_results = final
        self._last_scan_time = datetime.now()
        return final

    def _get_initial_universe(
        self,
        top_n: int = 20,
        min_price: float = 1.0,
        max_price: float = 10.0,
        min_change_pct: float = 10.0,
    ) -> list[ScreenerResult]:
        """
        Get initial universe of stocks from TradingView (sole source).

        Also pre-populates FloatDataProvider cache from TradingView's
        float_shares_outstanding field to save FMP API calls.

        Args:
            top_n: Maximum results to return
            min_price: Minimum stock price
            max_price: Maximum stock price
            min_change_pct: Minimum % change

        Returns:
            List of ScreenerResult from TradingView
        """
        if not self.use_tradingview or not self.tv_screener:
            logger.warning("[SCANNER] TradingView screener not configured")
            return []

        is_premarket = self._is_premarket()

        try:
            if is_premarket:
                results = self.tv_screener.get_premarket_gainers(
                    top_n=top_n,
                    min_price=min_price,
                    max_price=max_price,
                    min_change_pct=min_change_pct,
                )
                logger.info(
                    f"[SCANNER] TradingView pre-market: {len(results)} results"
                )
            else:
                results = self.tv_screener.get_active_gainers(
                    top_n=top_n,
                    min_price=min_price,
                    max_price=max_price,
                    min_change_pct=min_change_pct,
                )
                logger.info(
                    f"[SCANNER] TradingView active: {len(results)} results"
                )

            # Pre-populate float cache from TradingView data
            if self.float_provider and results:
                tv_float_cache = self.tv_screener.get_float_cache()
                for symbol, float_shares in tv_float_cache.items():
                    self.float_provider.set_float_hint(symbol, float_shares)
                if tv_float_cache:
                    logger.debug(
                        f"[SCANNER] Pre-cached float for "
                        f"{len(tv_float_cache)} symbols from TradingView"
                    )

            return results

        except Exception as e:
            logger.error(f"[SCANNER] TradingView failed: {e}")
            return []

    def _is_premarket(self) -> bool:
        """
        Check if we're in pre-market hours (before 7:00 AM ET).

        During pre-market, screener data may be stale,
        so we rely on TradingView exclusively.

        Returns:
            True if before 7:00 AM ET (trading_window_start)
        """
        et = pytz.timezone("US/Eastern")
        now_et = datetime.now(et)
        # TradingView's `change` field doesn't reset until market open (~9:30 AM).
        # Before that, get_active_gainers() returns yesterday's movers (stale).
        # Use premarket query (premarket_change field) until 9:30 AM.
        return now_et.hour < 9 or (now_et.hour == 9 and now_et.minute < 30)

    def _enrich_candidate(
        self,
        gainer: ScreenerResult,
        enable_float_filter: bool = True,
    ) -> Optional[MomentumCandidate]:
        """
        Enrich a gainer with relative volume and float data.

        Args:
            gainer: Raw screener result
            enable_float_filter: Whether to fetch float data

        Returns:
            MomentumCandidate with enriched data, or None on error
        """
        symbol = gainer.symbol

        try:
            # Fetch 20-day daily bars via DXLink
            if not self.client:
                logger.debug(f"[SCANNER] {symbol}: no client for bar data")
                return None

            bars_df = self.client.get_bars(symbol, timeframe="1Day", limit=21)
            bar_count = len(bars_df)
            if bar_count > 0:
                logger.debug(
                    f"[SCANNER] {symbol}: got {bar_count} daily bars "
                    f"(first={bars_df.index[0].date()}, last={bars_df.index[-1].date()})"
                )
            if bar_count < 5:
                logger.debug(f"[SCANNER] {symbol}: insufficient daily bars ({bar_count} < 5)")
                return None

            # Today's data (most recent bar)
            today_volume = int(bars_df["volume"].iloc[-1])
            today_close = float(bars_df["close"].iloc[-1])
            today_high = float(bars_df["high"].iloc[-1])
            today_low = float(bars_df["low"].iloc[-1])

            # Previous close
            prev_close = float(bars_df["close"].iloc[-2]) if bar_count > 1 else today_close

            # Calculate gap % from previous close
            gap_pct = ((today_close - prev_close) / prev_close * 100) if prev_close > 0 else 0

            # Calculate relative volume
            # Use historical bars excluding today
            historical_volumes = bars_df["volume"].iloc[:-1]
            if len(historical_volumes) >= 3:
                avg_volume = float(historical_volumes.mean())
            else:
                avg_volume = today_volume  # Fallback: no ratio

            relative_volume = (today_volume / avg_volume) if avg_volume > 0 else 1.0

            # Fetch float data
            float_shares = None
            if enable_float_filter and self.float_provider:
                float_data = self.float_provider.get_float(symbol)
                if float_data:
                    float_shares = float_data.float_shares

            # Fetch news/catalyst data (5th pillar)
            has_catalyst = False
            news_headline = None
            news_count = 0
            news_source = None

            if self.news_enabled and self.client:
                try:
                    articles = self.client.get_news(
                        symbol=symbol,
                        limit=self.news_max_articles,
                        hours_back=self.news_lookback_hours,
                    )
                    news_count = len(articles)
                    has_catalyst = news_count > 0
                    if articles:
                        news_headline = articles[0].get("headline", "")
                        news_source = articles[0].get("source", "")
                except Exception as e:
                    logger.debug(f"[SCANNER] News fetch failed for {symbol}: {e}")

            candidate = MomentumCandidate(
                symbol=symbol,
                price=gainer.price if gainer.price > 0 else today_close,
                change_pct=gainer.change_pct if gainer.change_pct != 0 else gap_pct,
                volume=today_volume,
                avg_volume=avg_volume,
                volume_ratio=relative_volume,
                relative_volume=relative_volume,
                float_shares=float_shares,
                gap_pct=gap_pct,
                high_of_day=today_high,
                low_of_day=today_low,
                prev_close=prev_close,
                has_catalyst=has_catalyst,
                news_headline=news_headline,
                news_count=news_count,
                news_source=news_source,
                source="momentum_scanner",
            )

            return candidate

        except Exception as e:
            logger.debug(f"[SCANNER] Error enriching {symbol}: {e}")
            return None

    def _apply_filters(
        self,
        candidates: list[MomentumCandidate],
        min_relative_volume: float = 5.0,
        min_float_millions: float = 0.5,
        enable_float_filter: bool = True,
    ) -> list[MomentumCandidate]:
        """
        Apply the remaining filters (relative volume, float floor).

        Candidates that partially pass are still included but marked.

        Args:
            candidates: Enriched candidates
            min_relative_volume: Minimum relative volume threshold
            min_float_millions: Minimum float in millions (liquidity floor)
            enable_float_filter: Whether to apply float filter

        Returns:
            Filtered list of candidates
        """
        results = []

        for c in candidates:
            failures = []

            # Relative volume filter
            if c.relative_volume is not None and c.relative_volume < min_relative_volume:
                failures.append(f"relVol={c.relative_volume:.1f}x < {min_relative_volume}x")

            # Float floor filter (reject micro-floats with too little liquidity)
            if enable_float_filter and c.float_shares is not None:
                float_m = c.float_shares / 1_000_000
                if float_m < min_float_millions:
                    failures.append(f"float={float_m:.2f}M < {min_float_millions}M min")

            c.filter_failures = failures
            c.passes_all_filters = len(failures) == 0

            # Include candidates that pass relative volume at minimum
            # Float data might be missing, so we're lenient there
            rv_passes = c.relative_volume is None or c.relative_volume >= min_relative_volume
            float_passes = (
                not enable_float_filter
                or c.float_shares is None  # Missing data = lenient
                or (c.float_shares / 1_000_000) >= min_float_millions
            )

            if rv_passes and float_passes:
                results.append(c)
            elif rv_passes:
                # Passes volume but not float — still include as partial match
                results.append(c)

        return results

    @property
    def last_scan_results(self) -> list[MomentumCandidate]:
        """Get results from the most recent scan."""
        return self._last_scan_results

    @property
    def last_scan_time(self) -> Optional[datetime]:
        """When the last scan was performed."""
        return self._last_scan_time
