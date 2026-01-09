"""
Tests for portfolio-level risk limits.
"""

import pytest

from src.risk.portfolio_limits import (
    PortfolioLimits,
    RiskStatus,
    TradingAction,
)


class TestPortfolioLimits:
    """Tests for PortfolioLimits class."""

    def test_drawdown_ok(self):
        """Test drawdown check when within limits."""
        limits = PortfolioLimits(max_drawdown_pct=0.15)

        # Set peak equity
        limits.update_equity(10000.0)

        # Check with 5% drawdown
        result = limits.check_drawdown(9500.0)

        assert result.passed is True
        assert result.status == RiskStatus.OK
        assert result.action == TradingAction.ALLOW

    def test_drawdown_warning(self):
        """Test drawdown warning level."""
        limits = PortfolioLimits(
            max_drawdown_pct=0.15,
            warning_threshold_pct=0.70,  # Warn at 10.5%
        )

        limits.update_equity(10000.0)

        # Check with 12% drawdown (above 70% of limit)
        result = limits.check_drawdown(8800.0)

        assert result.passed is True
        assert result.status == RiskStatus.WARNING
        assert result.action == TradingAction.REDUCE_ONLY

    def test_drawdown_exceeded(self):
        """Test drawdown limit exceeded."""
        limits = PortfolioLimits(max_drawdown_pct=0.15)

        limits.update_equity(10000.0)

        # Check with 20% drawdown
        result = limits.check_drawdown(8000.0)

        assert result.passed is False
        assert result.status == RiskStatus.CRITICAL
        assert result.action == TradingAction.HALT

    def test_daily_loss_ok(self):
        """Test daily loss within limits."""
        limits = PortfolioLimits(max_daily_loss_pct=0.03)

        # Initialize for today
        limits.update_equity(10000.0)
        limits._daily_stats.current_equity = 9900.0  # 1% loss

        result = limits.check_daily_loss()

        assert result.passed is True
        assert result.status == RiskStatus.OK

    def test_daily_loss_exceeded(self):
        """Test daily loss limit exceeded."""
        limits = PortfolioLimits(max_daily_loss_pct=0.03)

        limits.update_equity(10000.0)
        limits._daily_stats.current_equity = 9600.0  # 4% loss

        result = limits.check_daily_loss()

        assert result.passed is False
        assert result.status == RiskStatus.CRITICAL
        assert result.action == TradingAction.HALT

    def test_position_count_ok(self):
        """Test position count within limits."""
        limits = PortfolioLimits(max_positions=5)

        result = limits.check_position_count(3)

        assert result.passed is True
        assert result.action == TradingAction.ALLOW

    def test_position_count_at_limit(self):
        """Test position count at limit."""
        limits = PortfolioLimits(max_positions=5)

        result = limits.check_position_count(5)

        assert result.passed is False
        assert result.action == TradingAction.REDUCE_ONLY

    def test_can_open_position_all_ok(self):
        """Test comprehensive check when all limits OK."""
        limits = PortfolioLimits(
            max_drawdown_pct=0.15,
            max_daily_loss_pct=0.03,
            max_positions=5,
        )

        limits.update_equity(10000.0)

        result = limits.check_can_open_position(
            current_equity=9800.0,  # 2% drawdown
            current_positions=2,
        )

        assert result.passed is True
        assert result.action == TradingAction.ALLOW

    def test_can_open_position_drawdown_exceeded(self):
        """Test opening blocked by drawdown."""
        limits = PortfolioLimits(max_drawdown_pct=0.10)

        limits.update_equity(10000.0)

        result = limits.check_can_open_position(
            current_equity=8500.0,  # 15% drawdown
            current_positions=1,
        )

        assert result.passed is False
        assert result.action == TradingAction.HALT

    def test_trading_halted_persists(self):
        """Test that trading halt persists."""
        limits = PortfolioLimits(max_drawdown_pct=0.10)

        limits.update_equity(10000.0)

        # Trigger halt
        limits.check_can_open_position(8000.0, 1)

        # Try again even with recovered equity
        result = limits.check_can_open_position(9500.0, 1)

        assert result.passed is False
        assert result.status == RiskStatus.STOPPED

    def test_force_halt(self):
        """Test manual trading halt."""
        limits = PortfolioLimits()

        limits.update_equity(10000.0)
        limits.force_halt("Emergency stop")

        result = limits.check_can_open_position(10000.0, 0)

        assert result.passed is False
        assert "Manual halt" in result.message

    def test_resume_trading(self):
        """Test resuming trading after halt."""
        limits = PortfolioLimits(max_drawdown_pct=0.15)

        limits.update_equity(10000.0)
        limits.force_halt("Test halt")

        # Resume should work if within limits
        resumed = limits.resume_trading()

        assert resumed is True
        assert limits._trading_halted is False

    def test_can_close_position_always(self):
        """Closing positions should always be allowed."""
        limits = PortfolioLimits()

        # Even if trading halted
        limits.force_halt("Test")

        result = limits.check_can_close_position()

        assert result.passed is True
        assert result.action == TradingAction.ALLOW

    def test_get_status(self):
        """Test comprehensive status report."""
        limits = PortfolioLimits(
            max_drawdown_pct=0.15,
            max_daily_loss_pct=0.03,
            max_positions=5,
        )

        limits.update_equity(10000.0)

        status = limits.get_status(9800.0, 2)

        assert "trading_halted" in status
        assert "peak_equity" in status
        assert "daily_stats" in status
        assert "limits" in status
        assert "checks" in status

        assert status["peak_equity"] == 10000.0
        assert status["current_equity"] == 9800.0

    def test_reset_daily_limits(self):
        """Test daily limit reset."""
        limits = PortfolioLimits(max_daily_loss_pct=0.03)

        limits.update_equity(10000.0)
        limits._daily_stats.current_equity = 9500.0  # 5% loss
        limits._trading_halted = True

        limits.reset_daily_limits()

        assert limits._trading_halted is False
        assert limits._daily_stats.starting_equity == 9500.0

    def test_record_trade(self):
        """Test trade recording."""
        limits = PortfolioLimits()

        limits.update_equity(10000.0)

        limits.record_trade(is_winner=True, is_open=False)
        limits.record_trade(is_winner=False, is_open=False)
        limits.record_trade(is_winner=True, is_open=True)

        assert limits._daily_stats.trades_opened == 1
        assert limits._daily_stats.trades_closed == 2
        assert limits._daily_stats.winning_trades == 1
        assert limits._daily_stats.losing_trades == 1
