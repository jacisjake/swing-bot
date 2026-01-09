"""
Trade executor.

Coordinates order placement using existing OrderExecutor and PositionManager.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.bot.processor import TradeParams
from src.core.order_executor import OrderExecutor, OrderResult
from src.core.position_manager import Position, PositionManager, PositionSide


@dataclass
class ExecutionResult:
    """Result of trade execution."""
    success: bool
    order_result: Optional[OrderResult]
    position: Optional[Position]
    error: Optional[str]
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "order_id": self.order_result.order_id if self.order_result else None,
            "filled_qty": self.order_result.filled_qty if self.order_result else None,
            "filled_price": self.order_result.avg_fill_price if self.order_result else None,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


class TradeExecutor:
    """
    Execute trades from processed signals.

    Handles:
    - Order placement via OrderExecutor
    - Position tracking via PositionManager
    - Error handling and logging
    """

    def __init__(
        self,
        order_executor: OrderExecutor,
        position_manager: PositionManager,
    ):
        """
        Initialize trade executor.

        Args:
            order_executor: Order execution handler
            position_manager: Position tracking
        """
        self.order_executor = order_executor
        self.position_manager = position_manager

    async def execute_entry(self, trade_params: TradeParams) -> ExecutionResult:
        """
        Execute a trade entry.

        Args:
            trade_params: Trade parameters from SignalProcessor

        Returns:
            ExecutionResult with order and position info
        """
        symbol = trade_params.symbol
        timestamp = datetime.now()

        try:
            # Check if we already have a position
            if self.position_manager.has_position(symbol):
                return ExecutionResult(
                    success=False,
                    order_result=None,
                    position=None,
                    error=f"Position already exists for {symbol}",
                    timestamp=timestamp,
                )

            # Execute order
            if trade_params.order_type == "market":
                order_result = await self.order_executor.execute_market_order(
                    symbol=symbol,
                    qty=trade_params.quantity,
                    side=trade_params.side,
                    wait_for_fill=True,
                )
            else:
                order_result = await self.order_executor.execute_limit_order(
                    symbol=symbol,
                    qty=trade_params.quantity,
                    side=trade_params.side,
                    limit_price=trade_params.entry_price,
                    wait_for_fill=True,
                )

            # Check if filled
            if not order_result.is_filled:
                return ExecutionResult(
                    success=False,
                    order_result=order_result,
                    position=None,
                    error=f"Order not filled: {order_result.status}",
                    timestamp=timestamp,
                )

            # Record in position manager
            position_side = (
                PositionSide.LONG
                if trade_params.side == "buy"
                else PositionSide.SHORT
            )

            position = self.position_manager.open_position(
                symbol=symbol,
                side=position_side,
                qty=order_result.filled_qty,
                entry_price=order_result.avg_fill_price,
                stop_loss=trade_params.stop_price,
                take_profit=trade_params.target_price,
            )

            return ExecutionResult(
                success=True,
                order_result=order_result,
                position=position,
                error=None,
                timestamp=timestamp,
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                order_result=None,
                position=None,
                error=str(e),
                timestamp=timestamp,
            )

    async def execute_exit(
        self,
        symbol: str,
        reason: str,
        exit_price: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Execute a position exit.

        Args:
            symbol: Position to exit
            reason: Exit reason (stop_loss, take_profit, signal, etc.)
            exit_price: Optional limit price (market order if None)

        Returns:
            ExecutionResult with order info
        """
        timestamp = datetime.now()

        try:
            # Get current position
            position = self.position_manager.get_position(symbol)
            if not position:
                return ExecutionResult(
                    success=False,
                    order_result=None,
                    position=None,
                    error=f"No position found for {symbol}",
                    timestamp=timestamp,
                )

            # Determine exit side
            exit_side = "sell" if position.side == PositionSide.LONG else "buy"

            # Execute exit order
            if exit_price:
                order_result = await self.order_executor.execute_limit_order(
                    symbol=symbol,
                    qty=position.qty,
                    side=exit_side,
                    limit_price=exit_price,
                    wait_for_fill=True,
                )
            else:
                order_result = await self.order_executor.execute_market_order(
                    symbol=symbol,
                    qty=position.qty,
                    side=exit_side,
                    wait_for_fill=True,
                )

            # Check if filled
            if not order_result.is_filled:
                return ExecutionResult(
                    success=False,
                    order_result=order_result,
                    position=position,
                    error=f"Exit order not filled: {order_result.status}",
                    timestamp=timestamp,
                )

            # Close position in manager
            closed_position = self.position_manager.close_position(
                symbol=symbol,
                exit_price=order_result.avg_fill_price,
                reason=reason,
            )

            return ExecutionResult(
                success=True,
                order_result=order_result,
                position=closed_position,
                error=None,
                timestamp=timestamp,
            )

        except Exception as e:
            return ExecutionResult(
                success=False,
                order_result=None,
                position=None,
                error=str(e),
                timestamp=timestamp,
            )

    async def cancel_pending_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel pending orders.

        Args:
            symbol: Cancel for specific symbol, or all if None

        Returns:
            Number of orders cancelled
        """
        if symbol:
            orders = await self.order_executor.get_open_orders(symbol)
            cancelled = 0
            for order in orders:
                try:
                    await self.order_executor.cancel_order(order["id"])
                    cancelled += 1
                except Exception:
                    pass
            return cancelled
        else:
            return await self.order_executor.cancel_all_orders()
