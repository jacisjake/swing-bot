"""
Position Manager - Track positions, P&L, and trade history.

Maintains local state synchronized with Alpaca, calculates metrics,
and provides position lifecycle management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from loguru import logger


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"


class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    PENDING = "pending"  # Order submitted but not filled


@dataclass
class Position:
    """Represents an open or closed position."""

    symbol: str
    side: PositionSide
    qty: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_pct: Optional[float] = None

    # Updated as position evolves
    current_price: float = 0.0
    highest_price: float = 0.0  # For trailing stop
    lowest_price: float = float("inf")

    # Set when position closes
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None

    # Calculated fields
    status: PositionStatus = PositionStatus.OPEN

    @property
    def cost_basis(self) -> float:
        """Total cost of position."""
        return self.qty * self.entry_price

    @property
    def market_value(self) -> float:
        """Current market value."""
        return self.qty * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss in dollars."""
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.qty
        else:
            return (self.entry_price - self.current_price) * self.qty

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.entry_price == 0:
            return 0.0
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price

    @property
    def realized_pnl(self) -> Optional[float]:
        """Realized P&L (only for closed positions)."""
        if self.status != PositionStatus.CLOSED or self.exit_price is None:
            return None
        if self.side == PositionSide.LONG:
            return (self.exit_price - self.entry_price) * self.qty
        else:
            return (self.entry_price - self.exit_price) * self.qty

    @property
    def realized_pnl_pct(self) -> Optional[float]:
        """Realized P&L as percentage."""
        if self.status != PositionStatus.CLOSED or self.exit_price is None:
            return None
        if self.side == PositionSide.LONG:
            return (self.exit_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.exit_price) / self.entry_price

    @property
    def hold_duration(self) -> Optional[float]:
        """How long position was/has been held in hours."""
        end = self.exit_time or datetime.now()
        delta = end - self.entry_time
        return delta.total_seconds() / 3600

    def update_price(self, price: float) -> None:
        """Update current price and track high/low."""
        self.current_price = price
        self.highest_price = max(self.highest_price, price)
        self.lowest_price = min(self.lowest_price, price)

    def close(self, exit_price: float, reason: str) -> None:
        """Mark position as closed."""
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        self.exit_reason = reason
        self.status = PositionStatus.CLOSED
        self.current_price = exit_price

    def should_stop_loss(self) -> bool:
        """Check if stop-loss should trigger."""
        if self.stop_loss is None:
            return False

        if self.side == PositionSide.LONG:
            return self.current_price <= self.stop_loss
        else:
            return self.current_price >= self.stop_loss

    def should_take_profit(self) -> bool:
        """Check if take-profit should trigger."""
        if self.take_profit is None:
            return False

        if self.side == PositionSide.LONG:
            return self.current_price >= self.take_profit
        else:
            return self.current_price <= self.take_profit

    def get_trailing_stop_price(self) -> Optional[float]:
        """Calculate current trailing stop price."""
        if self.trailing_stop_pct is None:
            return None

        if self.side == PositionSide.LONG:
            return self.highest_price * (1 - self.trailing_stop_pct)
        else:
            return self.lowest_price * (1 + self.trailing_stop_pct)

    def should_trailing_stop(self) -> bool:
        """Check if trailing stop should trigger."""
        trailing_price = self.get_trailing_stop_price()
        if trailing_price is None:
            return False

        if self.side == PositionSide.LONG:
            return self.current_price <= trailing_price
        else:
            return self.current_price >= trailing_price

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop_pct": self.trailing_stop_pct,
            "current_price": self.current_price,
            "highest_price": self.highest_price,
            "lowest_price": self.lowest_price,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_reason": self.exit_reason,
            "status": self.status.value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "realized_pnl": self.realized_pnl,
            "hold_duration_hours": self.hold_duration,
        }


@dataclass
class PortfolioMetrics:
    """Aggregate portfolio metrics."""

    total_equity: float = 0.0
    cash: float = 0.0
    positions_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    open_positions: int = 0
    win_count: int = 0
    loss_count: int = 0
    total_trades: int = 0

    @property
    def win_rate(self) -> float:
        """Win rate as percentage."""
        if self.total_trades == 0:
            return 0.0
        return self.win_count / self.total_trades

    @property
    def avg_win(self) -> float:
        """Average winning trade P&L."""
        # Calculated from trade history
        return 0.0

    @property
    def avg_loss(self) -> float:
        """Average losing trade P&L."""
        return 0.0


class PositionManager:
    """
    Manages all positions and trade history.

    Responsibilities:
    - Track open positions with P&L
    - Maintain trade history
    - Calculate portfolio metrics
    - Sync with Alpaca positions
    """

    def __init__(self, trade_ledger=None):
        """
        Initialize position manager.

        Args:
            trade_ledger: Optional TradeLedger for persistent trade recording
        """
        self.positions: dict[str, Position] = {}
        self.closed_positions: list[Position] = []
        self._peak_equity: float = 0.0
        self._starting_equity: float = 0.0
        self.trade_ledger = trade_ledger

    def sync_with_broker(self, broker_positions: list[dict], equity: float) -> None:
        """
        Sync local state with broker positions.

        Called on startup and periodically to ensure consistency.
        """
        if self._starting_equity == 0:
            self._starting_equity = equity
        self._peak_equity = max(self._peak_equity, equity)

        broker_symbols = {p["symbol"] for p in broker_positions}

        # Add new positions from broker
        for bp in broker_positions:
            symbol = bp["symbol"]
            if symbol not in self.positions:
                # Position opened outside our system
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=PositionSide.LONG if bp["qty"] > 0 else PositionSide.SHORT,
                    qty=abs(bp["qty"]),
                    entry_price=bp["avg_entry_price"],
                    entry_time=datetime.now(),  # Unknown actual entry time
                    current_price=bp["current_price"],
                )
                logger.info(f"Synced existing position: {symbol}")
            else:
                # Update existing position
                self.positions[symbol].update_price(bp["current_price"])
                self.positions[symbol].qty = abs(bp["qty"])

        # Mark positions closed if no longer at broker
        for symbol in list(self.positions.keys()):
            if symbol not in broker_symbols:
                pos = self.positions.pop(symbol)
                pos.close(pos.current_price, "closed_externally")
                self.closed_positions.append(pos)
                # Record to persistent trade ledger
                if self.trade_ledger:
                    self.trade_ledger.record_trade(pos)
                logger.info(f"Position closed externally: {symbol}")

    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        qty: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop_pct: Optional[float] = None,
    ) -> Position:
        """
        Record a new position.

        Called after order is filled.
        """
        if symbol in self.positions:
            raise ValueError(f"Position already exists for {symbol}")

        position = Position(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=entry_price,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=trailing_stop_pct,
            current_price=entry_price,
            highest_price=entry_price,
            lowest_price=entry_price,
        )

        self.positions[symbol] = position
        logger.info(
            f"Opened {side.value} position: {qty} {symbol} @ ${entry_price:.2f}"
        )

        return position

    def close_position(
        self, symbol: str, exit_price: float, reason: str
    ) -> Optional[Position]:
        """
        Record a position closure.

        Called after exit order is filled.
        """
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return None

        position = self.positions.pop(symbol)
        position.close(exit_price, reason)
        self.closed_positions.append(position)

        # Record to persistent trade ledger
        if self.trade_ledger:
            self.trade_ledger.record_trade(position)

        pnl = position.realized_pnl or 0
        logger.info(
            f"Closed position: {symbol} @ ${exit_price:.2f} "
            f"P&L: ${pnl:.2f} ({position.realized_pnl_pct:.1%}) "
            f"Reason: {reason}"
        )

        return position

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all positions."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol."""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if we have an open position for a symbol."""
        return symbol in self.positions

    def get_open_positions(self) -> list[Position]:
        """Get all open positions."""
        return list(self.positions.values())

    def get_symbols(self) -> list[str]:
        """Get symbols of all open positions."""
        return list(self.positions.keys())

    def get_metrics(self, current_equity: float) -> PortfolioMetrics:
        """Calculate current portfolio metrics."""
        metrics = PortfolioMetrics()
        metrics.total_equity = current_equity
        metrics.open_positions = len(self.positions)

        # Sum position values
        for pos in self.positions.values():
            metrics.positions_value += pos.market_value
            metrics.unrealized_pnl += pos.unrealized_pnl

        # Calculate from closed positions
        for pos in self.closed_positions:
            pnl = pos.realized_pnl or 0
            metrics.realized_pnl += pnl
            metrics.total_trades += 1
            if pnl > 0:
                metrics.win_count += 1
            elif pnl < 0:
                metrics.loss_count += 1

        return metrics

    def get_current_drawdown(self, current_equity: float) -> float:
        """Calculate current drawdown from peak equity."""
        if self._peak_equity == 0:
            return 0.0
        return (self._peak_equity - current_equity) / self._peak_equity

    def get_positions_needing_exit(self) -> list[tuple[Position, str]]:
        """
        Check all positions for exit conditions.

        Returns list of (position, reason) tuples for positions that should exit.
        """
        exits = []

        for pos in self.positions.values():
            if pos.should_stop_loss():
                exits.append((pos, "stop_loss"))
            elif pos.should_take_profit():
                exits.append((pos, "take_profit"))
            elif pos.should_trailing_stop():
                exits.append((pos, "trailing_stop"))

        return exits

    def to_dict(self) -> dict:
        """Serialize state for persistence."""
        return {
            "positions": {s: p.to_dict() for s, p in self.positions.items()},
            "closed_positions": [p.to_dict() for p in self.closed_positions[-100:]],
            "peak_equity": self._peak_equity,
            "starting_equity": self._starting_equity,
        }
