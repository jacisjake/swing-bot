"""
Unified Alpaca API client.

Single client handling stocks, crypto, and options with consistent interface.
Mode (paper/live) controlled by configuration, not code changes.
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import (
    CryptoHistoricalDataClient,
    StockHistoricalDataClient,
)
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import (
    CryptoBarsRequest,
    NewsRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import (
    GetAssetsRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    TrailingStopOrderRequest,
)
from loguru import logger

from config import settings


class AlpacaClient:
    """
    Unified Alpaca client for all trading operations.

    Combines trading, stock data, and crypto data clients into one interface.
    Paper/live mode determined by settings, not constructor args.
    """

    def __init__(self):
        """Initialize all Alpaca clients based on settings."""
        self._validate_credentials()

        # Trading client (paper or live based on settings)
        self.trading = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=settings.is_paper,
        )

        # Data clients (same credentials for both modes)
        self.stock_data = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )

        self.crypto_data = CryptoHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )

        self.news = NewsClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )

        logger.info(
            f"AlpacaClient initialized in {'PAPER' if settings.is_paper else 'LIVE'} mode"
        )

    def _validate_credentials(self) -> None:
        """Validate API credentials exist and look valid."""
        if not settings.alpaca_api_key or len(settings.alpaca_api_key) < 10:
            raise ValueError("Invalid ALPACA_API_KEY - check .env file")
        if not settings.alpaca_secret_key or len(settings.alpaca_secret_key) < 10:
            raise ValueError("Invalid ALPACA_SECRET_KEY - check .env file")

    # =========================================================================
    # Account Methods
    # =========================================================================

    def get_account(self) -> dict:
        """Get account information including equity and buying power."""
        account = self.trading.get_account()
        return {
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "account_blocked": account.account_blocked,
        }

    def get_buying_power(self) -> float:
        """Get current buying power."""
        return float(self.trading.get_account().buying_power)

    def get_equity(self) -> float:
        """Get current account equity."""
        return float(self.trading.get_account().equity)

    # =========================================================================
    # Position Methods
    # =========================================================================

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        positions = self.trading.get_all_positions()
        return [
            {
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "side": "long" if float(pos.qty) > 0 else "short",
                "market_value": float(pos.market_value),
                "cost_basis": float(pos.cost_basis),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "current_price": float(pos.current_price),
                "avg_entry_price": float(pos.avg_entry_price),
                "asset_class": pos.asset_class.value,
            }
            for pos in positions
        ]

    def get_position(self, symbol: str) -> Optional[dict]:
        """Get position for a specific symbol, or None if not held."""
        try:
            pos = self.trading.get_open_position(symbol)
            return {
                "symbol": pos.symbol,
                "qty": float(pos.qty),
                "side": "long" if float(pos.qty) > 0 else "short",
                "market_value": float(pos.market_value),
                "cost_basis": float(pos.cost_basis),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "current_price": float(pos.current_price),
                "avg_entry_price": float(pos.avg_entry_price),
            }
        except Exception:
            return None

    def has_position(self, symbol: str) -> bool:
        """Check if we have an open position in a symbol."""
        return self.get_position(symbol) is not None

    # =========================================================================
    # Market Data Methods
    # =========================================================================

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        limit: int = 100,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get historical bars for a symbol.

        Args:
            symbol: Stock or crypto symbol (e.g., "AAPL" or "BTC/USD")
            timeframe: "1Min", "5Min", "15Min", "1Hour", "1Day"
            limit: Number of bars to retrieve
            end: End datetime (defaults to now)

        Returns:
            DataFrame with columns: open, high, low, close, volume, vwap
        """
        tf = self._parse_timeframe(timeframe)
        end = end or datetime.now()
        start = end - timedelta(days=limit * 2)  # Fetch extra to ensure we get enough

        is_crypto = "/" in symbol

        if is_crypto:
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
            )
            bars = self.crypto_data.get_crypto_bars(request)
        else:
            # Map string setting to DataFeed enum
            feed = DataFeed.SIP if settings.alpaca_data_feed == "sip" else DataFeed.IEX
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
                feed=feed,
            )
            bars = self.stock_data.get_stock_bars(request)

        # Convert to DataFrame
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")
        
        return df.tail(limit)

    def get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        is_crypto = "/" in symbol

        if is_crypto:
            bars = self.get_bars(symbol, timeframe="1Min", limit=1)
            if bars.empty:
                raise ValueError(f"No price data for {symbol}")
            return float(bars["close"].iloc[-1])
        else:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.stock_data.get_stock_latest_quote(request)
            quote = quotes[symbol]
            # Use midpoint of bid/ask, or last trade if available
            if quote.bid_price and quote.ask_price:
                return (float(quote.bid_price) + float(quote.ask_price)) / 2
            return float(quote.ask_price or quote.bid_price)

    def get_latest_quotes_with_change(self, symbols: list[str]) -> dict:
        """Get the latest price and daily change for a list of symbols."""
        if not symbols:
            return {}

        results = {}
        stock_symbols = [s for s in symbols if "/" not in s]
        crypto_symbols = [s for s in symbols if "/" in s]

        # Process stocks
        if stock_symbols:
            try:
                request = StockLatestQuoteRequest(symbol_or_symbols=stock_symbols)
                quotes = self.stock_data.get_stock_latest_quote(request)

                for symbol in stock_symbols:
                    price = 0.0
                    change = 0.0
                    quote = quotes.get(symbol)

                    if quote:
                        if quote.bid_price and quote.ask_price:
                            price = (float(quote.bid_price) + float(quote.ask_price)) / 2
                        elif quote.bid_price:
                            price = float(quote.bid_price)
                        elif quote.ask_price:
                            price = float(quote.ask_price)

                    # Get daily bars for change calculation
                    bars = self.get_bars(symbol, timeframe="1Day", limit=2)
                    if len(bars) >= 2:
                        latest_close = bars["close"].iloc[-1]
                        prev_close = bars["close"].iloc[-2]
                        change = latest_close - prev_close
                        # If live price is more recent, use it for price
                        if price == 0.0:
                            price = latest_close
                    elif len(bars) == 1:
                        # If only one bar, use its close as price, change is 0
                        if price == 0.0:
                            price = bars["close"].iloc[-1]

                    results[symbol] = {
                        "price": price,
                        "change": change,
                    }
            except Exception as e:
                logger.error(f"Error fetching latest stock quotes with change: {e}")

        # Process crypto (similar logic, but getting quotes directly from bars)
        if crypto_symbols:
            for symbol in crypto_symbols:
                price = 0.0
                change = 0.0
                
                bars = self.get_bars(symbol, timeframe="1Day", limit=2)
                if len(bars) >= 2:
                    latest_close = bars["close"].iloc[-1]
                    prev_close = bars["close"].iloc[-2]
                    price = latest_close
                    change = latest_close - prev_close
                elif len(bars) == 1:
                    price = bars["close"].iloc[-1]
                    change = 0.0 # No previous day to calculate change

                results[symbol] = {
                    "price": price,
                    "change": change,
                }
            
        return results

    def get_news(
        self,
        symbol: str,
        limit: int = 5,
        hours_back: int = 12,
    ) -> list[dict]:
        """
        Get recent news articles for a symbol.

        Args:
            symbol: Stock ticker (e.g., "AAPL")
            limit: Maximum articles to return
            hours_back: How far back to look for news

        Returns:
            List of news article dicts with headline, summary,
            created_at, source, url. Empty list on failure.
        """
        try:
            end = datetime.now()
            start = end - timedelta(hours=hours_back)

            request = NewsRequest(
                symbols=symbol,
                start=start,
                end=end,
                limit=limit,
                include_content=False,
                exclude_contentless=False,
            )

            response = self.news.get_news(request)

            articles = []
            # response.data is a dict with key "news" containing News objects
            news_list = response.data.get("news", []) if isinstance(response.data, dict) else []
            for article in news_list:
                created = getattr(article, "created_at", None)
                articles.append({
                    "headline": getattr(article, "headline", ""),
                    "summary": (getattr(article, "summary", "") or "")[:200],
                    "source": getattr(article, "source", ""),
                    "created_at": created.isoformat() if hasattr(created, "isoformat") else str(created),
                    "url": getattr(article, "url", ""),
                    "symbols": getattr(article, "symbols", []),
                })

            return articles[:limit]

        except Exception as e:
            logger.debug(f"News fetch failed for {symbol}: {e}")
            return []

    def _parse_timeframe(self, timeframe: str) -> TimeFrame:
        """Parse string timeframe to Alpaca TimeFrame."""
        mapping = {
            "1min": TimeFrame(1, TimeFrameUnit.Minute),
            "5min": TimeFrame(5, TimeFrameUnit.Minute),
            "15min": TimeFrame(15, TimeFrameUnit.Minute),
            "30min": TimeFrame(30, TimeFrameUnit.Minute),
            "1hour": TimeFrame(1, TimeFrameUnit.Hour),
            "4hour": TimeFrame(4, TimeFrameUnit.Hour),
            "1day": TimeFrame(1, TimeFrameUnit.Day),
            "1week": TimeFrame(1, TimeFrameUnit.Week),
        }
        tf_lower = timeframe.lower().replace(" ", "")
        if tf_lower not in mapping:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        return mapping[tf_lower]

    def get_multi_timeframe_bars(
        self,
        symbol: str,
        timeframes: list[str],
        limit: int = 100,
        end: Optional[datetime] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Get historical bars for multiple timeframes at once.

        Used for multi-timeframe analysis (System 3 strategy).

        Args:
            symbol: Stock or crypto symbol
            timeframes: List of timeframes (e.g., ["1Day", "4Hour", "1Hour"])
            limit: Number of bars to retrieve per timeframe
            end: End datetime (defaults to now)

        Returns:
            Dict mapping timeframe -> DataFrame with OHLCV data
        """
        results = {}
        for tf in timeframes:
            try:
                bars = self.get_bars(symbol, timeframe=tf, limit=limit, end=end)
                results[tf] = bars
            except Exception as e:
                logger.warning(f"Failed to fetch {tf} bars for {symbol}: {e}")
                results[tf] = pd.DataFrame()
        return results

    # =========================================================================
    # Order Methods
    # =========================================================================

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
    ) -> dict:
        """
        Submit a market order.

        Args:
            symbol: Stock or crypto symbol
            qty: Quantity to buy/sell (can be fractional)
            side: "buy" or "sell"

        Returns:
            Order details dict
        """
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )

        order = self.trading.submit_order(request)
        logger.info(f"Market order submitted: {side} {qty} {symbol}")

        return self._order_to_dict(order)

    def submit_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        extended_hours: bool = False,
    ) -> dict:
        """Submit a limit order."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            limit_price=round(limit_price, 2),
            time_in_force=TimeInForce.DAY,
            extended_hours=extended_hours,
        )

        order = self.trading.submit_order(request)
        logger.info(f"Limit order submitted: {side} {qty} {symbol} @ ${limit_price}")

        return self._order_to_dict(order)

    def submit_stop_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_price: float,
        limit_price: float,
    ) -> dict:
        """Submit a stop-limit order (typically for stop-losses)."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        request = StopLimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            stop_price=round(stop_price, 2),
            limit_price=round(limit_price, 2),
            time_in_force=TimeInForce.DAY,
        )

        order = self.trading.submit_order(request)
        logger.info(
            f"Stop-limit order submitted: {side} {qty} {symbol} "
            f"stop=${stop_price} limit=${limit_price}"
        )

        return self._order_to_dict(order)

    def submit_trailing_stop_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        trail_percent: float,
    ) -> dict:
        """Submit a trailing stop order."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        request = TrailingStopOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            trail_percent=trail_percent,
            time_in_force=TimeInForce.DAY,
        )

        order = self.trading.submit_order(request)
        logger.info(
            f"Trailing stop order submitted: {side} {qty} {symbol} trail={trail_percent}%"
        )

        return self._order_to_dict(order)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order by ID."""
        try:
            self.trading.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        cancelled = self.trading.cancel_orders()
        count = len(cancelled) if cancelled else 0
        logger.info(f"Cancelled {count} orders")
        return count

    def get_orders(self, status: str = "open") -> list[dict]:
        """Get orders by status (open, closed, all)."""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        status_map = {
            "open": QueryOrderStatus.OPEN,
            "closed": QueryOrderStatus.CLOSED,
            "all": QueryOrderStatus.ALL,
        }
        request = GetOrdersRequest(status=status_map.get(status, QueryOrderStatus.OPEN))
        orders = self.trading.get_orders(request)
        return [self._order_to_dict(o) for o in orders]

    def _order_to_dict(self, order) -> dict:
        """Convert order object to dictionary."""
        return {
            "id": str(order.id),
            "symbol": order.symbol,
            "qty": float(order.qty) if order.qty else None,
            "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
            "side": order.side.value,
            "type": order.type.value,
            "status": order.status.value,
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "stop_price": float(order.stop_price) if order.stop_price else None,
            "filled_avg_price": (
                float(order.filled_avg_price) if order.filled_avg_price else None
            ),
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
        }

    # =========================================================================
    # Asset Methods
    # =========================================================================

    def get_tradeable_stocks(self, min_price: float = 5.0) -> list[str]:
        """Get list of tradeable stock symbols."""
        request = GetAssetsRequest(asset_class=AssetClass.US_EQUITY)
        assets = self.trading.get_all_assets(request)

        return [
            a.symbol
            for a in assets
            if a.tradable and a.fractionable and a.status == "active"
        ]

    def get_asset(self, symbol: str) -> dict:
        """Get asset information including name, exchange, etc."""
        try:
            asset = self.trading.get_asset(symbol)
            return {
                "symbol": asset.symbol,
                "name": asset.name,
                "exchange": asset.exchange.value if asset.exchange else "",
                "class": asset.asset_class.value if asset.asset_class else "",
                "tradable": asset.tradable,
                "fractionable": asset.fractionable,
                "status": asset.status.value if asset.status else "",
            }
        except Exception as e:
            return {"symbol": symbol, "name": symbol, "error": str(e)}

    def is_fractionable(self, symbol: str) -> bool:
        """Check if an asset supports fractional shares."""
        try:
            asset = self.trading.get_asset(symbol)
            return asset.fractionable
        except Exception:
            # Assume not fractionable if we can't check
            return False

    def is_market_open(self) -> bool:
        """Check if the stock market is currently open."""
        clock = self.trading.get_clock()
        return clock.is_open

    def get_trade_stats(self) -> dict:
        """
        Calculate trading stats from Alpaca order history.

        Matches buy/sell orders by symbol to calculate realized P&L.
        Returns stats including total trades, win rate, realized P&L.
        """
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=500,  # Get recent history
        )
        orders = self.trading.get_orders(request)

        # Group filled orders by symbol
        symbol_orders: dict[str, list] = {}
        for order in orders:
            if order.filled_qty and float(order.filled_qty) > 0:
                symbol = order.symbol
                if symbol not in symbol_orders:
                    symbol_orders[symbol] = []
                symbol_orders[symbol].append({
                    "side": order.side.value,
                    "qty": float(order.filled_qty),
                    "price": float(order.filled_avg_price) if order.filled_avg_price else 0,
                    "time": order.filled_at or order.created_at,
                })

        # Calculate P&L for completed round trips (buy then sell)
        completed_trades = []
        total_realized_pnl = 0.0

        for symbol, orders_list in symbol_orders.items():
            # Sort by time
            orders_list.sort(key=lambda x: x["time"] if x["time"] else datetime.min)

            # Track position and cost basis
            position_qty = 0.0
            cost_basis = 0.0

            for order in orders_list:
                if order["side"] == "buy":
                    # Add to position
                    cost_basis += order["qty"] * order["price"]
                    position_qty += order["qty"]
                elif order["side"] == "sell" and position_qty > 0:
                    # Calculate P&L for this sale
                    sell_qty = min(order["qty"], position_qty)
                    avg_cost = cost_basis / position_qty if position_qty > 0 else 0
                    sell_proceeds = sell_qty * order["price"]
                    sell_cost = sell_qty * avg_cost

                    pnl = sell_proceeds - sell_cost
                    total_realized_pnl += pnl

                    completed_trades.append({
                        "symbol": symbol,
                        "qty": sell_qty,
                        "entry_price": avg_cost,
                        "exit_price": order["price"],
                        "pnl": pnl,
                        "pnl_pct": (pnl / sell_cost * 100) if sell_cost > 0 else 0,
                    })

                    # Reduce position
                    cost_basis -= sell_qty * avg_cost
                    position_qty -= sell_qty

        # Calculate stats
        wins = [t for t in completed_trades if t["pnl"] > 0]
        losses = [t for t in completed_trades if t["pnl"] < 0]

        return {
            "total_trades": len(completed_trades),
            "total_realized_pnl": total_realized_pnl,
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": (len(wins) / len(completed_trades) * 100) if completed_trades else 0.0,
            "avg_win": sum(t["pnl"] for t in wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(t["pnl"] for t in losses) / len(losses) if losses else 0.0,
            "largest_win": max((t["pnl"] for t in wins), default=0.0),
            "largest_loss": min((t["pnl"] for t in losses), default=0.0),
            "trades": completed_trades[-20:],  # Return last 20 trades for display
        }
