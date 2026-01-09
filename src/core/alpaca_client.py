"""
Unified Alpaca API client.

Single client handling stocks, crypto, and options with consistent interface.
Mode (paper/live) controlled by configuration, not code changes.
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from alpaca.data.historical import (
    CryptoHistoricalDataClient,
    StockHistoricalDataClient,
)
from alpaca.data.requests import (
    CryptoBarsRequest,
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
                limit=limit,
            )
            bars = self.crypto_data.get_crypto_bars(request)
        else:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
                limit=limit,
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
    ) -> dict:
        """Submit a limit order."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            limit_price=round(limit_price, 2),
            time_in_force=TimeInForce.DAY,
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
        orders = self.trading.get_orders(status=status)
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

    def is_market_open(self) -> bool:
        """Check if the stock market is currently open."""
        clock = self.trading.get_clock()
        return clock.is_open
