"""
Stock and crypto screener.

Uses Alpaca's screener API to find:
- Market movers (top gainers/losers)
- Most active by volume
- Volume breakout candidates
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

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

            return results

        except Exception as e:
            print(f"Error in REST most actives fallback: {e}")
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
