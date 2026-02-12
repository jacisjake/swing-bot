"""
Stock and crypto screener for momentum day trading.

Uses TradingView (primary) and Alpaca (fallback) screener APIs to find:
- Market movers (top gainers/losers)
- Most active by volume
- Volume breakout candidates
- Momentum candidates (low-float, high relative volume, big % gainers)

TradingView is the primary scanner because Alpaca's StockScreener.get_market_movers()
returns PREVIOUS-DAY movers during pre-market — it doesn't update until market open.
TradingView provides real pre-market data starting at 4:00 AM ET.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pytz
from loguru import logger

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import (
    MostActivesRequest,
    StockBarsRequest,
)
from alpaca.data.timeframe import TimeFrame

from config.settings import get_settings


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


class StockScreener:
    """
    Screen for tradeable stocks using Alpaca data.

    Methods:
    - get_top_gainers: Stocks with biggest gains today
    - get_top_losers: Stocks with biggest losses today
    - get_most_active: Stocks with highest volume
    - get_volume_breakouts: Stocks with unusual volume
    """

    def __init__(self):
        """Initialize screener with Alpaca credentials."""
        settings = get_settings()
        self.data_client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )

        # Default screening parameters
        self.default_top_n = 10
        self.volume_lookback_days = 20
        self.volume_breakout_threshold = 1.5  # 1.5x average volume

    def get_most_active(self, top_n: int = 10) -> list[ScreenerResult]:
        """
        Get most active stocks by volume.

        Args:
            top_n: Number of stocks to return

        Returns:
            List of ScreenerResult sorted by volume
        """
        try:
            from alpaca.data.historical.screener import ScreenerClient

            settings = get_settings()
            screener = ScreenerClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
            )

            request = MostActivesRequest(top=top_n)
            response = screener.get_most_actives(request)

            results = []
            for stock in response.most_actives:
                # ActiveStock only has symbol, volume, trade_count
                results.append(ScreenerResult(
                    symbol=stock.symbol,
                    price=0.0,  # Not available from this endpoint
                    change_pct=0.0,  # Not available from this endpoint
                    volume=int(stock.volume) if stock.volume else 0,
                    source="most_active",
                ))

            logger.info(f"[SCREENER] Most Active from Alpaca ({len(results)} results):")
            for r in results:
                logger.info(f"  {r.symbol}: volume={r.volume:,}")

            return results

        except ImportError:
            return self._get_most_actives_rest(top_n)
        except Exception as e:
            print(f"Error fetching most actives: {e}")
            return self._get_most_actives_rest(top_n)

    def get_top_gainers(self, top_n: int = 10) -> list[ScreenerResult]:
        """
        Get top gaining stocks.

        Uses Alpaca's market movers endpoint.

        Args:
            top_n: Number of stocks to return

        Returns:
            List of ScreenerResult sorted by gain %
        """
        try:
            # Alpaca's movers endpoint via REST API
            # Note: ScreenerClient may not be available in all alpaca-py versions
            # Fallback to manual implementation
            from alpaca.data.historical.screener import ScreenerClient
            from alpaca.data.requests import MarketMoversRequest
            from alpaca.data.enums import MarketType

            settings = get_settings()
            screener = ScreenerClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
            )

            request = MarketMoversRequest(top=top_n, market_type=MarketType.STOCKS)
            response = screener.get_market_movers(request)

            results = []
            for mover in response.gainers:
                results.append(ScreenerResult(
                    symbol=mover.symbol,
                    price=float(mover.price) if mover.price else 0.0,
                    change_pct=float(mover.percent_change) if mover.percent_change else 0.0,
                    source="top_gainer",
                ))

            logger.info(f"[SCREENER] Top Gainers from Alpaca ({len(results)} results):")
            for r in results:
                logger.info(f"  {r.symbol}: {r.change_pct:+.2f}% @ ${r.price:.2f}")

            return results

        except ImportError:
            # ScreenerClient not available, use REST fallback
            return self._get_movers_rest("gainers", top_n)
        except Exception as e:
            print(f"Error fetching top gainers: {e}")
            return []

    def get_top_losers(self, top_n: int = 10) -> list[ScreenerResult]:
        """
        Get top losing stocks.

        Args:
            top_n: Number of stocks to return

        Returns:
            List of ScreenerResult sorted by loss %
        """
        try:
            from alpaca.data.historical.screener import ScreenerClient
            from alpaca.data.requests import MarketMoversRequest
            from alpaca.data.enums import MarketType

            settings = get_settings()
            screener = ScreenerClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
            )

            request = MarketMoversRequest(top=top_n, market_type=MarketType.STOCKS)
            response = screener.get_market_movers(request)

            results = []
            for mover in response.losers:
                results.append(ScreenerResult(
                    symbol=mover.symbol,
                    price=float(mover.price) if mover.price else 0.0,
                    change_pct=float(mover.percent_change) if mover.percent_change else 0.0,
                    source="top_loser",
                ))

            return results

        except ImportError:
            return self._get_movers_rest("losers", top_n)
        except Exception as e:
            print(f"Error fetching top losers: {e}")
            return []

    def _get_movers_rest(self, mover_type: str, top_n: int) -> list[ScreenerResult]:
        """Fallback REST API call for movers if SDK doesn't have ScreenerClient."""
        import requests

        settings = get_settings()
        url = f"https://data.alpaca.markets/v1beta1/screener/stocks/movers"

        headers = {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        }

        params = {"top": top_n}

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            movers = data.get(mover_type, [])
            for mover in movers:
                results.append(ScreenerResult(
                    symbol=mover.get("symbol", ""),
                    price=float(mover.get("price", 0)),
                    change_pct=float(mover.get("percent_change", 0)),
                    source=f"top_{mover_type[:-1]}",  # "gainers" -> "top_gainer"
                ))

            logger.info(f"[SCREENER] {mover_type.title()} from Alpaca REST ({len(results)} results):")
            for r in results:
                logger.info(f"  {r.symbol}: {r.change_pct:+.2f}% @ ${r.price:.2f}")

            return results

        except Exception as e:
            print(f"Error in REST movers fallback: {e}")
            return []

    def _get_most_actives_rest(self, top_n: int) -> list[ScreenerResult]:
        """Fallback REST API call for most actives."""
        import requests

        settings = get_settings()
        url = "https://data.alpaca.markets/v1beta1/screener/stocks/most-actives"

        headers = {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        }

        params = {"top": top_n, "by": "volume"}

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            results = []
            for stock in data.get("most_actives", []):
                results.append(ScreenerResult(
                    symbol=stock.get("symbol", ""),
                    price=float(stock.get("price", 0)),
                    change_pct=float(stock.get("percent_change", 0)),
                    volume=int(stock.get("volume", 0)),
                    source="most_active",
                ))

            logger.info(f"[SCREENER] Most Active from Alpaca REST ({len(results)} results):")
            for r in results:
                logger.info(f"  {r.symbol}: {r.change_pct:+.2f}% @ ${r.price:.2f}, volume={r.volume:,}")

            return results

        except Exception as e:
            logger.error(f"Error in REST most actives fallback: {e}")
            return []

    def get_volume_breakouts(
        self,
        symbols: list[str],
        volume_threshold: float = 1.5,
        lookback_days: int = 20,
    ) -> list[ScreenerResult]:
        """
        Find stocks with unusual volume (potential breakouts).

        Args:
            symbols: List of symbols to check
            volume_threshold: Multiple of avg volume to qualify (e.g., 1.5 = 50% above avg)
            lookback_days: Days to calculate average volume

        Returns:
            List of ScreenerResult for stocks with volume > threshold * avg
        """
        results = []

        for symbol in symbols:
            try:
                # Get recent bars for volume analysis
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    limit=lookback_days + 1,  # +1 for today
                )
                bars = self.data_client.get_stock_bars(request)

                if symbol not in bars or len(bars[symbol]) < lookback_days:
                    continue

                symbol_bars = list(bars[symbol])

                # Today's volume (most recent bar)
                today_volume = int(symbol_bars[-1].volume)
                today_close = float(symbol_bars[-1].close)
                yesterday_close = float(symbol_bars[-2].close) if len(symbol_bars) > 1 else today_close

                # Average volume (excluding today)
                historical_volumes = [int(bar.volume) for bar in symbol_bars[:-1]]
                avg_volume = sum(historical_volumes) / len(historical_volumes) if historical_volumes else 0

                if avg_volume == 0:
                    continue

                volume_ratio = today_volume / avg_volume
                change_pct = ((today_close - yesterday_close) / yesterday_close) * 100 if yesterday_close else 0

                # Check if volume exceeds threshold
                if volume_ratio >= volume_threshold:
                    results.append(ScreenerResult(
                        symbol=symbol,
                        price=today_close,
                        change_pct=change_pct,
                        volume=today_volume,
                        avg_volume=avg_volume,
                        volume_ratio=volume_ratio,
                        source="volume_breakout",
                    ))

            except Exception as e:
                print(f"Error checking volume for {symbol}: {e}")
                continue

        # Sort by volume ratio descending
        results.sort(key=lambda x: x.volume_ratio or 0, reverse=True)
        return results

    def get_combined_watchlist(
        self,
        include_gainers: bool = True,
        include_active: bool = True,
        include_losers: bool = False,
        top_n: int = 5,
    ) -> list[str]:
        """
        Get combined watchlist from multiple screeners.

        Args:
            include_gainers: Include top gainers
            include_active: Include most active by volume
            include_losers: Include top losers (for short or mean reversion)
            top_n: Number from each category

        Returns:
            Deduplicated list of symbols
        """
        symbols = set()

        if include_gainers:
            for result in self.get_top_gainers(top_n):
                symbols.add(result.symbol)

        if include_active:
            for result in self.get_most_active(top_n):
                symbols.add(result.symbol)

        if include_losers:
            for result in self.get_top_losers(top_n):
                symbols.add(result.symbol)

        logger.info(f"[SCREENER] Combined watchlist: {sorted(symbols)}")
        return list(symbols)


class CryptoScreener:
    """
    Screen for crypto trading opportunities.

    Crypto is simpler - fewer options, so we focus on:
    - Price momentum
    - Volume changes
    """

    # Major crypto pairs on Alpaca
    CRYPTO_SYMBOLS = [
        "BTC/USD",
        "ETH/USD",
        "SOL/USD",
        "AVAX/USD",
        "LINK/USD",
        "DOT/USD",
        "MATIC/USD",
        "UNI/USD",
        "AAVE/USD",
        "LTC/USD",
    ]

    def __init__(self):
        """Initialize crypto screener."""
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient

        settings = get_settings()
        self.data_client = CryptoHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )

    def get_crypto_movers(self, top_n: int = 5) -> list[ScreenerResult]:
        """
        Get crypto assets with biggest moves.

        Args:
            top_n: Number of results

        Returns:
            List of ScreenerResult sorted by absolute change
        """
        from alpaca.data.requests import CryptoBarsRequest

        results = []

        try:
            # Get recent bars for all crypto symbols
            request = CryptoBarsRequest(
                symbol_or_symbols=self.CRYPTO_SYMBOLS,
                timeframe=TimeFrame.Hour,
                limit=25,  # ~1 day of hourly bars
            )
            bars = self.data_client.get_crypto_bars(request)

            for symbol in self.CRYPTO_SYMBOLS:
                if symbol not in bars or len(bars[symbol]) < 2:
                    continue

                symbol_bars = list(bars[symbol])
                current_price = float(symbol_bars[-1].close)
                price_24h_ago = float(symbol_bars[0].close)

                change_pct = ((current_price - price_24h_ago) / price_24h_ago) * 100 if price_24h_ago else 0

                # Calculate volume
                total_volume = sum(float(bar.volume) for bar in symbol_bars)

                results.append(ScreenerResult(
                    symbol=symbol,
                    price=current_price,
                    change_pct=change_pct,
                    volume=int(total_volume),
                    source="crypto_mover",
                ))

        except Exception as e:
            print(f"Error fetching crypto movers: {e}")

        # Sort by absolute change (biggest movers first)
        results.sort(key=lambda x: abs(x.change_pct), reverse=True)
        return results[:top_n]

    def get_oversold_crypto(self) -> list[str]:
        """
        Get crypto assets that are down significantly (oversold candidates).

        Returns:
            List of symbols that are down > 5% in 24h
        """
        movers = self.get_crypto_movers(top_n=10)
        return [r.symbol for r in movers if r.change_pct < -5.0]

    def get_momentum_crypto(self) -> list[str]:
        """
        Get crypto assets with positive momentum.

        Returns:
            List of symbols that are up > 2% in 24h
        """
        movers = self.get_crypto_movers(top_n=10)
        return [r.symbol for r in movers if r.change_pct > 2.0]


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
    1. Fetch top gainers from Alpaca screener API (initial universe)
    2. Enrich each with snapshot data and relative volume calculation
    3. Fetch float data from FMP/yfinance
    4. Apply filter chain
    5. Return top candidates sorted by relative volume
    """

    def __init__(self, float_provider=None, alpaca_client=None,
                 news_enabled: bool = True, news_lookback_hours: int = 12,
                 news_max_articles: int = 5,
                 tv_screener=None, use_tradingview: bool = True):
        """
        Initialize momentum screener.

        Args:
            float_provider: FloatDataProvider instance for float data
            alpaca_client: AlpacaClient instance for news data
            news_enabled: Whether to check for news catalysts
            news_lookback_hours: How far back to look for news
            news_max_articles: Max articles to fetch per symbol
            tv_screener: TradingViewScreener instance (optional)
            use_tradingview: Whether to use TradingView as primary source
        """
        settings = get_settings()
        self.data_client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
        self.stock_screener = StockScreener()
        self.tv_screener = tv_screener
        self.use_tradingview = use_tradingview
        self.float_provider = float_provider
        self.alpaca_client = alpaca_client
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
            top_n: Number of gainers to fetch from Alpaca
            max_results: Maximum candidates to return

        Returns:
            List of MomentumCandidate sorted by relative volume (best first),
            with $2+ stocks weighted higher than $1-$2 stocks.
        """
        logger.info(f"[SCANNER] Running momentum scan...")

        # Step 1: Get initial universe (TradingView primary, Alpaca fallback)
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
        Get initial universe of stocks from the best available source.

        Source selection by time of day:
        - Pre-market (before 7 AM ET): TradingView only (Alpaca is stale)
        - Active session (7 AM+): TradingView + Alpaca merged & deduped
        - TradingView failure: falls back to Alpaca silently

        Also pre-populates FloatDataProvider cache from TradingView's
        float_shares_outstanding field to save FMP API calls.

        Args:
            top_n: Maximum results to return per source
            min_price: Minimum stock price
            max_price: Maximum stock price
            min_change_pct: Minimum % change

        Returns:
            List of ScreenerResult from best available source(s)
        """
        tv_results = []
        alpaca_results = []
        is_premarket = self._is_premarket()

        # Try TradingView first (primary source)
        if self.use_tradingview and self.tv_screener:
            try:
                if is_premarket:
                    tv_results = self.tv_screener.get_premarket_gainers(
                        top_n=top_n,
                        min_price=min_price,
                        max_price=max_price,
                        min_change_pct=min_change_pct,
                    )
                    logger.info(
                        f"[SCANNER] TradingView pre-market: {len(tv_results)} results"
                    )
                else:
                    tv_results = self.tv_screener.get_active_gainers(
                        top_n=top_n,
                        min_price=min_price,
                        max_price=max_price,
                        min_change_pct=min_change_pct,
                    )
                    logger.info(
                        f"[SCANNER] TradingView active: {len(tv_results)} results"
                    )

                # Pre-populate float cache from TradingView data
                if self.float_provider and tv_results:
                    tv_float_cache = self.tv_screener.get_float_cache()
                    for symbol, float_shares in tv_float_cache.items():
                        self.float_provider.set_float_hint(symbol, float_shares)
                    if tv_float_cache:
                        logger.debug(
                            f"[SCANNER] Pre-cached float for "
                            f"{len(tv_float_cache)} symbols from TradingView"
                        )

            except Exception as e:
                logger.error(f"[SCANNER] TradingView failed: {e}")
                tv_results = []

        # During pre-market, only use TradingView (Alpaca shows stale data)
        if is_premarket:
            if tv_results:
                return tv_results
            # TradingView failed during premarket — try Alpaca as last resort
            logger.warning(
                "[SCANNER] TradingView failed in pre-market, "
                "falling back to Alpaca (data may be stale)"
            )

        # During active session or if TradingView failed: also get Alpaca data
        try:
            alpaca_results = self.stock_screener.get_top_gainers(top_n=top_n)
            logger.info(
                f"[SCANNER] Alpaca gainers: {len(alpaca_results)} results"
            )
        except Exception as e:
            logger.error(f"[SCANNER] Alpaca screener failed: {e}")

        # Merge and deduplicate (TradingView takes priority)
        if tv_results and alpaca_results:
            merged = self._merge_screener_results(tv_results, alpaca_results)
            logger.info(
                f"[SCANNER] Merged: {len(merged)} unique symbols "
                f"(TV={len(tv_results)}, Alpaca={len(alpaca_results)})"
            )
            return merged
        elif tv_results:
            return tv_results
        else:
            return alpaca_results

    def _merge_screener_results(
        self,
        primary: list[ScreenerResult],
        secondary: list[ScreenerResult],
    ) -> list[ScreenerResult]:
        """
        Merge and deduplicate screener results from two sources.

        Primary source (TradingView) takes priority for duplicate symbols.
        Secondary source (Alpaca) fills in any symbols not found in primary.

        Args:
            primary: Primary source results (TradingView)
            secondary: Secondary source results (Alpaca)

        Returns:
            Merged and deduplicated list, primary results first
        """
        seen = set()
        merged = []

        # Add all primary results first
        for r in primary:
            if r.symbol not in seen:
                seen.add(r.symbol)
                merged.append(r)

        # Add secondary results not already in primary
        for r in secondary:
            if r.symbol not in seen:
                seen.add(r.symbol)
                merged.append(r)

        return merged

    def _is_premarket(self) -> bool:
        """
        Check if we're in pre-market hours (before 7:00 AM ET).

        During pre-market, Alpaca's screener returns stale previous-day data,
        so we should rely on TradingView exclusively.

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
            # Fetch 20-day daily bars using the working AlpacaClient
            if not self.alpaca_client:
                logger.debug(f"[SCANNER] {symbol}: no alpaca_client for bar data")
                return None

            bars_df = self.alpaca_client.get_bars(symbol, timeframe="1Day", limit=21)
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

            if self.news_enabled and self.alpaca_client:
                try:
                    articles = self.alpaca_client.get_news(
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
