"""
Tests for position tracking and P&L calculations.
"""

from datetime import datetime

import pytest

from src.core.position_manager import (
    Position,
    PositionManager,
    PositionSide,
    PositionStatus,
)


class TestPosition:
    """Tests for Position dataclass."""

    def test_position_creation(self):
        """Test basic position creation."""
        pos = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            qty=10.0,
            entry_price=150.0,
            entry_time=datetime.now(),
        )

        assert pos.symbol == "AAPL"
        assert pos.side == PositionSide.LONG
        assert pos.qty == 10.0
        assert pos.entry_price == 150.0
        assert pos.status == PositionStatus.OPEN

    def test_cost_basis(self):
        """Test cost basis calculation."""
        pos = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            qty=10.0,
            entry_price=150.0,
            entry_time=datetime.now(),
        )

        assert pos.cost_basis == 1500.0

    def test_unrealized_pnl_long_profit(self):
        """Test unrealized P&L for profitable long."""
        pos = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            qty=10.0,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=160.0,
        )

        assert pos.unrealized_pnl == pytest.approx(100.0)
        assert pos.unrealized_pnl_pct == pytest.approx(0.0667, rel=0.01)

    def test_unrealized_pnl_long_loss(self):
        """Test unrealized P&L for losing long."""
        pos = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            qty=10.0,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=140.0,
        )

        assert pos.unrealized_pnl == pytest.approx(-100.0)

    def test_update_price(self):
        """Test price update tracking."""
        pos = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            qty=10.0,
            entry_price=150.0,
            entry_time=datetime.now(),
        )

        pos.update_price(155.0)
        pos.update_price(160.0)
        pos.update_price(158.0)

        assert pos.current_price == 158.0
        assert pos.highest_price == 160.0
        assert pos.lowest_price == 150.0  # Entry was lowest

    def test_close_position(self):
        """Test position closure."""
        pos = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            qty=10.0,
            entry_price=150.0,
            entry_time=datetime.now(),
        )

        pos.close(160.0, "take_profit")

        assert pos.status == PositionStatus.CLOSED
        assert pos.exit_price == 160.0
        assert pos.exit_reason == "take_profit"
        assert pos.realized_pnl == pytest.approx(100.0)

    def test_stop_loss_trigger_long(self):
        """Test stop-loss detection for long position."""
        pos = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            qty=10.0,
            entry_price=150.0,
            entry_time=datetime.now(),
            stop_loss=145.0,
            current_price=144.0,
        )

        assert pos.should_stop_loss() is True

    def test_stop_loss_not_triggered(self):
        """Test stop-loss not triggered when price above."""
        pos = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            qty=10.0,
            entry_price=150.0,
            entry_time=datetime.now(),
            stop_loss=145.0,
            current_price=148.0,
        )

        assert pos.should_stop_loss() is False

    def test_take_profit_trigger(self):
        """Test take-profit detection."""
        pos = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            qty=10.0,
            entry_price=150.0,
            entry_time=datetime.now(),
            take_profit=160.0,
            current_price=161.0,
        )

        assert pos.should_take_profit() is True

    def test_trailing_stop(self):
        """Test trailing stop calculation."""
        pos = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            qty=10.0,
            entry_price=150.0,
            entry_time=datetime.now(),
            trailing_stop_pct=0.05,
            highest_price=170.0,
            current_price=160.0,
        )

        # Trailing stop at 170 * 0.95 = 161.5
        assert pos.get_trailing_stop_price() == pytest.approx(161.5)
        assert pos.should_trailing_stop() is True  # Current 160 < 161.5

    def test_to_dict(self):
        """Test serialization to dict."""
        pos = Position(
            symbol="AAPL",
            side=PositionSide.LONG,
            qty=10.0,
            entry_price=150.0,
            entry_time=datetime.now(),
            current_price=155.0,
        )

        d = pos.to_dict()

        assert d["symbol"] == "AAPL"
        assert d["qty"] == 10.0
        assert d["unrealized_pnl"] == 50.0


class TestPositionManager:
    """Tests for PositionManager class."""

    def test_open_position(self):
        """Test opening a new position."""
        manager = PositionManager()

        pos = manager.open_position(
            symbol="AAPL",
            side=PositionSide.LONG,
            qty=10.0,
            entry_price=150.0,
            stop_loss=145.0,
        )

        assert pos.symbol == "AAPL"
        assert "AAPL" in manager.positions
        assert len(manager.get_open_positions()) == 1

    def test_cannot_open_duplicate(self):
        """Test duplicate position prevention."""
        manager = PositionManager()

        manager.open_position(
            symbol="AAPL",
            side=PositionSide.LONG,
            qty=10.0,
            entry_price=150.0,
        )

        with pytest.raises(ValueError, match="already exists"):
            manager.open_position(
                symbol="AAPL",
                side=PositionSide.LONG,
                qty=5.0,
                entry_price=155.0,
            )

    def test_close_position(self):
        """Test closing a position."""
        manager = PositionManager()

        manager.open_position(
            symbol="AAPL",
            side=PositionSide.LONG,
            qty=10.0,
            entry_price=150.0,
        )

        closed = manager.close_position("AAPL", 160.0, "take_profit")

        assert closed is not None
        assert closed.realized_pnl == pytest.approx(100.0)
        assert "AAPL" not in manager.positions
        assert len(manager.closed_positions) == 1

    def test_close_nonexistent(self):
        """Test closing non-existent position returns None."""
        manager = PositionManager()

        result = manager.close_position("AAPL", 160.0, "take_profit")

        assert result is None

    def test_update_prices(self):
        """Test batch price updates."""
        manager = PositionManager()

        manager.open_position("AAPL", PositionSide.LONG, 10.0, 150.0)
        manager.open_position("MSFT", PositionSide.LONG, 5.0, 300.0)

        manager.update_prices({"AAPL": 155.0, "MSFT": 310.0, "GOOG": 2800.0})

        assert manager.positions["AAPL"].current_price == 155.0
        assert manager.positions["MSFT"].current_price == 310.0

    def test_get_positions_needing_exit(self):
        """Test detection of positions needing exit."""
        manager = PositionManager()

        # Position hitting stop
        manager.open_position(
            symbol="AAPL",
            side=PositionSide.LONG,
            qty=10.0,
            entry_price=150.0,
            stop_loss=145.0,
        )
        manager.positions["AAPL"].current_price = 144.0

        # Position hitting target
        manager.open_position(
            symbol="MSFT",
            side=PositionSide.LONG,
            qty=5.0,
            entry_price=300.0,
            take_profit=320.0,
        )
        manager.positions["MSFT"].current_price = 325.0

        # Position OK
        manager.open_position(
            symbol="GOOG",
            side=PositionSide.LONG,
            qty=2.0,
            entry_price=2700.0,
            stop_loss=2600.0,
        )
        manager.positions["GOOG"].current_price = 2750.0

        exits = manager.get_positions_needing_exit()

        assert len(exits) == 2
        symbols = [pos.symbol for pos, _ in exits]
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "GOOG" not in symbols

    def test_metrics(self):
        """Test portfolio metrics calculation."""
        manager = PositionManager()

        # Open and close some positions
        manager.open_position("AAPL", PositionSide.LONG, 10.0, 150.0)
        manager.positions["AAPL"].current_price = 160.0

        manager.open_position("MSFT", PositionSide.LONG, 5.0, 300.0)
        manager.close_position("MSFT", 310.0, "take_profit")  # +$50

        manager.open_position("GOOG", PositionSide.LONG, 2.0, 2700.0)
        manager.close_position("GOOG", 2650.0, "stop_loss")  # -$100

        metrics = manager.get_metrics(current_equity=10000.0)

        assert metrics.open_positions == 1
        assert metrics.total_trades == 2
        assert metrics.win_count == 1
        assert metrics.loss_count == 1
        assert metrics.realized_pnl == pytest.approx(-50.0)  # +50 -100
        assert metrics.win_rate == pytest.approx(0.5)

    def test_sync_with_broker(self):
        """Test syncing with broker positions."""
        manager = PositionManager()

        # Simulate broker positions
        broker_positions = [
            {
                "symbol": "AAPL",
                "qty": 10.0,
                "avg_entry_price": 150.0,
                "current_price": 155.0,
            },
            {
                "symbol": "MSFT",
                "qty": 5.0,
                "avg_entry_price": 300.0,
                "current_price": 305.0,
            },
        ]

        manager.sync_with_broker(broker_positions, equity=10000.0)

        assert len(manager.positions) == 2
        assert "AAPL" in manager.positions
        assert "MSFT" in manager.positions
        assert manager.positions["AAPL"].current_price == 155.0

    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        manager = PositionManager()

        # Set peak equity
        manager.sync_with_broker([], equity=10000.0)

        # Calculate drawdown at lower equity
        dd = manager.get_current_drawdown(9000.0)

        assert dd == pytest.approx(0.10)  # 10% drawdown
