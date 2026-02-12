"""
Order Executor - Handle order placement with retries and verification.

Responsible for:
- Submitting orders with proper error handling
- Retrying on transient failures
- Verifying fills
- Managing order lifecycle
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from loguru import logger

from .alpaca_client import AlpacaClient


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class OrderResult:
    """Result of an order execution attempt."""

    success: bool
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    filled_price: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "order_id": self.order_id,
            "status": self.status.value,
            "filled_qty": self.filled_qty,
            "filled_price": self.filled_price,
            "error": self.error,
        }


class OrderExecutor:
    """
    Handles order execution with retries and verification.

    Features:
    - Automatic retries on transient failures
    - Order fill verification
    - Proper error classification
    - Logging of all order activity
    """

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 1.0
    FILL_CHECK_INTERVAL = 0.5
    FILL_TIMEOUT_SECONDS = 30.0

    # Error messages that indicate transient failures (retry-able)
    TRANSIENT_ERRORS = [
        "connection",
        "timeout",
        "rate limit",
        "temporarily",
        "try again",
        "503",
        "504",
    ]

    def __init__(self, client: AlpacaClient):
        self.client = client

    def execute_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        wait_for_fill: bool = True,
    ) -> OrderResult:
        """
        Execute a market order with retries.

        Args:
            symbol: Stock or crypto symbol
            qty: Quantity to trade
            side: "buy" or "sell"
            wait_for_fill: Whether to wait for order to fill

        Returns:
            OrderResult with execution details
        """
        return self._execute_with_retry(
            order_type="market",
            symbol=symbol,
            qty=qty,
            side=side,
            wait_for_fill=wait_for_fill,
        )

    def execute_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        wait_for_fill: bool = False,
        extended_hours: bool = False,
    ) -> OrderResult:
        """
        Execute a limit order with retries.

        Args:
            symbol: Stock or crypto symbol
            qty: Quantity to trade
            side: "buy" or "sell"
            limit_price: Limit price
            wait_for_fill: Whether to wait for order to fill

        Returns:
            OrderResult with execution details
        """
        return self._execute_with_retry(
            order_type="limit",
            symbol=symbol,
            qty=qty,
            side=side,
            limit_price=limit_price,
            wait_for_fill=wait_for_fill,
            extended_hours=extended_hours,
        )

    def execute_stop_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        stop_price: float,
        limit_price: float,
    ) -> OrderResult:
        """
        Execute a stop-limit order (for stop-losses).

        These are submitted but not expected to fill immediately.
        """
        return self._execute_with_retry(
            order_type="stop_limit",
            symbol=symbol,
            qty=qty,
            side=side,
            stop_price=stop_price,
            limit_price=limit_price,
            wait_for_fill=False,  # Stop orders wait for trigger
        )

    def execute_trailing_stop_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        trail_percent: float,
    ) -> OrderResult:
        """Execute a trailing stop order."""
        return self._execute_with_retry(
            order_type="trailing_stop",
            symbol=symbol,
            qty=qty,
            side=side,
            trail_percent=trail_percent,
            wait_for_fill=False,
        )

    def _execute_with_retry(
        self,
        order_type: str,
        symbol: str,
        qty: float,
        side: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trail_percent: Optional[float] = None,
        wait_for_fill: bool = True,
        extended_hours: bool = False,
    ) -> OrderResult:
        """
        Execute order with retry logic.

        Retries on transient failures, fails fast on permanent errors.
        """
        last_error = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                # Submit order based on type
                order = self._submit_order(
                    order_type=order_type,
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    limit_price=limit_price,
                    stop_price=stop_price,
                    trail_percent=trail_percent,
                    extended_hours=extended_hours,
                )

                if order is None:
                    raise ValueError("Order submission returned None")

                logger.debug(f"Order submitted: {order['id']} (attempt {attempt})")

                # Wait for fill if requested
                if wait_for_fill:
                    return self._wait_for_fill(order["id"])
                else:
                    return OrderResult(
                        success=True,
                        order_id=order["id"],
                        status=OrderStatus.SUBMITTED,
                    )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Order attempt {attempt}/{self.MAX_RETRIES} failed: {last_error}"
                )

                # Check if error is transient (retry-able)
                if self._is_transient_error(last_error):
                    if attempt < self.MAX_RETRIES:
                        time.sleep(self.RETRY_DELAY_SECONDS * attempt)
                        continue
                else:
                    # Permanent error, don't retry
                    break

        # All retries exhausted
        logger.error(f"Order failed after {self.MAX_RETRIES} attempts: {last_error}")
        return OrderResult(
            success=False,
            status=OrderStatus.FAILED,
            error=last_error,
        )

    def _submit_order(
        self,
        order_type: str,
        symbol: str,
        qty: float,
        side: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trail_percent: Optional[float] = None,
        extended_hours: bool = False,
    ) -> dict:
        """Submit order to Alpaca based on type."""
        # Check if asset supports fractional shares
        if not self.client.is_fractionable(symbol):
            # Round to whole shares for non-fractionable assets
            original_qty = qty
            qty = int(qty)
            if qty < 1:
                raise ValueError(f"Cannot buy less than 1 share of {symbol} (non-fractionable)")
            if qty != original_qty:
                logger.info(f"Rounded {symbol} qty from {original_qty:.6f} to {qty} (non-fractionable)")

        if order_type == "market":
            return self.client.submit_market_order(symbol, qty, side)
        elif order_type == "limit":
            if limit_price is None:
                raise ValueError("limit_price required for limit orders")
            return self.client.submit_limit_order(
                symbol, qty, side, limit_price, extended_hours=extended_hours
            )
        elif order_type == "stop_limit":
            if stop_price is None or limit_price is None:
                raise ValueError("stop_price and limit_price required for stop-limit")
            return self.client.submit_stop_limit_order(
                symbol, qty, side, stop_price, limit_price
            )
        elif order_type == "trailing_stop":
            if trail_percent is None:
                raise ValueError("trail_percent required for trailing stop")
            return self.client.submit_trailing_stop_order(
                symbol, qty, side, trail_percent
            )
        else:
            raise ValueError(f"Unknown order type: {order_type}")

    def _wait_for_fill(self, order_id: str) -> OrderResult:
        """Wait for order to fill or timeout."""
        start_time = time.time()

        while time.time() - start_time < self.FILL_TIMEOUT_SECONDS:
            orders = self.client.get_orders(status="all")
            order = next((o for o in orders if o["id"] == order_id), None)

            if order is None:
                return OrderResult(
                    success=False,
                    order_id=order_id,
                    status=OrderStatus.FAILED,
                    error="Order not found",
                )

            status = order["status"]

            if status == "filled":
                return OrderResult(
                    success=True,
                    order_id=order_id,
                    status=OrderStatus.FILLED,
                    filled_qty=order["filled_qty"],
                    filled_price=order["filled_avg_price"],
                )
            elif status == "partially_filled":
                # Continue waiting
                pass
            elif status in ["cancelled", "expired", "rejected"]:
                return OrderResult(
                    success=False,
                    order_id=order_id,
                    status=OrderStatus(status),
                    filled_qty=order["filled_qty"],
                    error=f"Order {status}",
                )

            time.sleep(self.FILL_CHECK_INTERVAL)

        # Timeout - check final status
        orders = self.client.get_orders(status="all")
        order = next((o for o in orders if o["id"] == order_id), None)

        if order and order["filled_qty"] > 0:
            # Partial fill
            return OrderResult(
                success=True,  # Partial success
                order_id=order_id,
                status=OrderStatus.PARTIALLY_FILLED,
                filled_qty=order["filled_qty"],
                filled_price=order["filled_avg_price"],
            )

        # Cancel the order
        self.client.cancel_order(order_id)

        return OrderResult(
            success=False,
            order_id=order_id,
            status=OrderStatus.EXPIRED,
            error="Order fill timeout",
        )

    def _is_transient_error(self, error_message: str) -> bool:
        """Check if error is transient (should retry)."""
        error_lower = error_message.lower()
        return any(term in error_lower for term in self.TRANSIENT_ERRORS)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        return self.client.cancel_order(order_id)

    def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        return self.client.cancel_all_orders()

    def get_open_orders(self) -> list[dict]:
        """Get all open orders."""
        return self.client.get_orders(status="open")

    def get_order_status(self, order_id: str) -> Optional[OrderResult]:
        """Get current status of an order."""
        orders = self.client.get_orders(status="all")
        order = next((o for o in orders if o["id"] == order_id), None)

        if order is None:
            return None

        status_map = {
            "new": OrderStatus.SUBMITTED,
            "accepted": OrderStatus.SUBMITTED,
            "pending_new": OrderStatus.PENDING,
            "filled": OrderStatus.FILLED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "expired": OrderStatus.EXPIRED,
            "rejected": OrderStatus.REJECTED,
        }

        return OrderResult(
            success=order["status"] == "filled",
            order_id=order_id,
            status=status_map.get(order["status"], OrderStatus.PENDING),
            filled_qty=order["filled_qty"],
            filled_price=order["filled_avg_price"],
        )
