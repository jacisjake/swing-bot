"""
Tests for stop-loss calculation and management.
"""

import pytest

from src.risk.stop_manager import StopManager, StopType


class TestStopManager:
    """Tests for StopManager class."""

    def test_fixed_stop_long(self):
        """Test fixed percentage stop for long position."""
        manager = StopManager(default_stop_pct=0.05)

        result = manager.calculate_fixed_stop(
            entry_price=100.0,
            side="long",
        )

        assert result.price == pytest.approx(95.0, rel=0.001)
        assert result.type == StopType.FIXED_PERCENT
        assert result.distance_pct == pytest.approx(0.05, rel=0.001)

    def test_fixed_stop_short(self):
        """Test fixed percentage stop for short position."""
        manager = StopManager(default_stop_pct=0.05)

        result = manager.calculate_fixed_stop(
            entry_price=100.0,
            side="short",
        )

        # Short stop is above entry
        assert result.price == pytest.approx(105.0, rel=0.001)

    def test_fixed_stop_with_target(self):
        """Test fixed stop with risk/reward calculation."""
        manager = StopManager()

        result = manager.calculate_fixed_stop(
            entry_price=100.0,
            stop_pct=0.05,
            side="long",
            target_price=115.0,  # 15% target
        )

        # Risk 5, Reward 15 = 3:1 R:R
        assert result.risk_reward == pytest.approx(3.0, rel=0.01)

    def test_atr_stop(self):
        """Test ATR-based stop calculation."""
        manager = StopManager(atr_multiplier=2.0)

        result = manager.calculate_atr_stop(
            entry_price=100.0,
            atr=2.5,  # ATR = $2.50
            side="long",
        )

        # Stop = 100 - (2.5 * 2) = 95
        assert result.price == pytest.approx(95.0, rel=0.001)
        assert result.type == StopType.ATR_BASED

    def test_atr_stop_high_volatility(self):
        """Higher ATR = wider stop."""
        manager = StopManager(atr_multiplier=2.0)

        result = manager.calculate_atr_stop(
            entry_price=100.0,
            atr=5.0,  # Higher ATR
            side="long",
        )

        # Stop = 100 - (5 * 2) = 90
        assert result.price == pytest.approx(90.0, rel=0.001)

    def test_support_stop(self, sample_bars):
        """Test support-based stop calculation."""
        manager = StopManager()

        entry_price = float(sample_bars["close"].iloc[-1])

        result = manager.calculate_support_stop(
            entry_price=entry_price,
            bars=sample_bars,
            lookback=20,
            side="long",
        )

        # Stop should be below recent low
        recent_low = float(sample_bars.tail(20)["low"].min())
        assert result.price < recent_low
        assert result.type == StopType.SUPPORT_LEVEL

    def test_trailing_stop_long(self):
        """Test trailing stop calculation."""
        manager = StopManager()

        result = manager.calculate_trailing_stop(
            entry_price=100.0,
            current_price=110.0,
            highest_price=115.0,  # Highest since entry
            trail_pct=0.05,
            side="long",
        )

        # Stop trails 5% below highest
        # 115 * 0.95 = 109.25
        assert result.price == pytest.approx(109.25, rel=0.01)
        assert result.type == StopType.TRAILING_PERCENT

    def test_trailing_stop_protects_entry(self):
        """Trailing stop never goes below entry."""
        manager = StopManager()

        result = manager.calculate_trailing_stop(
            entry_price=100.0,
            current_price=101.0,  # Barely profitable
            highest_price=102.0,
            trail_pct=0.05,  # Would put stop at 96.9
            side="long",
        )

        # Stop should be at entry (breakeven), not below
        assert result.price >= 100.0

    def test_breakeven_stop_activated(self):
        """Test breakeven stop when activated."""
        manager = StopManager(trailing_activation_pct=0.02)

        result = manager.calculate_breakeven_stop(
            entry_price=100.0,
            current_price=105.0,  # 5% profit
            original_stop=95.0,
            side="long",
        )

        # Should move to breakeven + buffer
        assert result.price > 100.0
        assert result.type == StopType.BREAKEVEN

    def test_breakeven_stop_not_activated(self):
        """Test breakeven stop before activation threshold."""
        manager = StopManager(trailing_activation_pct=0.05)

        result = manager.calculate_breakeven_stop(
            entry_price=100.0,
            current_price=102.0,  # Only 2% profit
            original_stop=95.0,
            side="long",
        )

        # Should stay at original stop
        assert result.price == 95.0
        assert result.type == StopType.FIXED_PERCENT

    def test_stop_from_bars(self, sample_bars):
        """Test calculating stop from bar data."""
        manager = StopManager()

        entry_price = float(sample_bars["close"].iloc[-1])

        result = manager.calculate_stop_from_bars(
            entry_price=entry_price,
            bars=sample_bars,
            atr_period=14,
            side="long",
        )

        assert result.price < entry_price
        assert result.type == StopType.ATR_BASED

    def test_should_tighten_stop(self):
        """Test stop tightening recommendation."""
        manager = StopManager()

        # Profitable position with loose stop
        should_tighten = manager.should_tighten_stop(
            entry_price=100.0,
            current_price=110.0,  # 10% profit
            current_stop=90.0,  # Stop way below entry
            profit_threshold=0.05,
            side="long",
        )

        assert should_tighten is True

    def test_should_not_tighten_stop_early(self):
        """Don't tighten stop if not enough profit."""
        manager = StopManager()

        should_tighten = manager.should_tighten_stop(
            entry_price=100.0,
            current_price=102.0,  # Only 2% profit
            current_stop=90.0,
            profit_threshold=0.05,
            side="long",
        )

        assert should_tighten is False
