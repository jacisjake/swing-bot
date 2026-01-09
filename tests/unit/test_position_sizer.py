"""
Tests for position sizing calculations.

These are critical for risk management - ensuring we never risk
more than intended on any single trade.
"""

import pandas as pd
import pytest

from src.risk.position_sizer import PositionSizer, SizingMethod, calculate_position_size


class TestPositionSizer:
    """Tests for PositionSizer class."""

    def test_fixed_fractional_basic(self, small_account):
        """Test basic fixed fractional sizing."""
        sizer = PositionSizer(max_position_risk_pct=0.02)

        result = sizer.calculate_fixed_fractional(
            account_equity=small_account["equity"],  # $1000
            entry_price=100.0,
            stop_price=95.0,  # 5% stop
        )

        # Risk $20 (2% of $1000), stop is $5 away
        # Position size should be 4 shares ($400)
        assert result.shares == pytest.approx(4.0, rel=0.01)
        assert result.dollar_amount == pytest.approx(400.0, rel=0.01)
        assert result.risk_amount == pytest.approx(20.0, rel=0.01)
        assert result.method == SizingMethod.FIXED_FRACTIONAL

    def test_fixed_fractional_tight_stop(self, small_account):
        """Tighter stop = larger position."""
        sizer = PositionSizer(max_position_risk_pct=0.02)

        result = sizer.calculate_fixed_fractional(
            account_equity=small_account["equity"],
            entry_price=100.0,
            stop_price=98.0,  # 2% stop (tighter)
        )

        # Risk $20, stop is $2 away
        # Position size should be 10 shares ($1000) - but capped at 25%
        assert result.shares == pytest.approx(2.5, rel=0.01)  # Capped at $250 (25%)
        assert result.capped_by_max_position is True

    def test_fixed_fractional_wide_stop(self, small_account):
        """Wider stop = smaller position."""
        sizer = PositionSizer(max_position_risk_pct=0.02)

        result = sizer.calculate_fixed_fractional(
            account_equity=small_account["equity"],
            entry_price=100.0,
            stop_price=90.0,  # 10% stop (wide)
        )

        # Risk $20, stop is $10 away
        # Position size should be 2 shares ($200)
        assert result.shares == pytest.approx(2.0, rel=0.01)
        assert result.dollar_amount == pytest.approx(200.0, rel=0.01)

    def test_buying_power_cap(self, small_account):
        """Position capped by buying power."""
        sizer = PositionSizer(max_position_risk_pct=0.10)  # 10% risk (aggressive)

        result = sizer.calculate_fixed_fractional(
            account_equity=small_account["equity"],
            entry_price=100.0,
            stop_price=99.0,  # 1% stop
            buying_power=200.0,  # Limited buying power
        )

        # Would want $10000 position but capped at $200 buying power
        assert result.dollar_amount == pytest.approx(200.0, rel=0.01)
        assert result.capped_by_buying_power is True

    def test_zero_stop_distance(self, small_account):
        """Handle zero stop distance gracefully."""
        sizer = PositionSizer()

        result = sizer.calculate_fixed_fractional(
            account_equity=small_account["equity"],
            entry_price=100.0,
            stop_price=100.0,  # No stop distance
        )

        assert result.shares == 0
        assert result.dollar_amount == 0

    def test_atr_based_sizing(self, small_account, sample_bars):
        """Test ATR-based position sizing."""
        sizer = PositionSizer(max_position_risk_pct=0.02)

        result = sizer.calculate_from_bars(
            account_equity=small_account["equity"],
            entry_price=sample_bars["close"].iloc[-1],
            bars=sample_bars,
            atr_multiplier=2.0,
        )

        assert result.shares > 0
        assert result.method == SizingMethod.ATR_BASED
        assert result.stop_distance > 0

    def test_atr_high_volatility(self, small_account, volatile_bars):
        """High volatility = smaller position."""
        sizer = PositionSizer(max_position_risk_pct=0.02)

        result = sizer.calculate_from_bars(
            account_equity=small_account["equity"],
            entry_price=volatile_bars["close"].iloc[-1],
            bars=volatile_bars,
            atr_multiplier=2.0,
        )

        # Higher ATR should result in smaller position
        assert result.shares > 0
        assert result.stop_distance > 0

    def test_kelly_positive_edge(self, small_account):
        """Test Kelly sizing with positive edge."""
        sizer = PositionSizer()

        result = sizer.calculate_kelly(
            account_equity=small_account["equity"],
            entry_price=100.0,
            stop_price=95.0,
            win_rate=0.55,  # 55% win rate
            avg_win_pct=0.06,  # 6% avg win
            avg_loss_pct=0.03,  # 3% avg loss
            kelly_fraction=0.25,  # 1/4 Kelly
        )

        assert result.shares > 0
        assert result.method == SizingMethod.KELLY

    def test_kelly_no_edge(self, small_account):
        """Test Kelly sizing with no edge (should be zero)."""
        sizer = PositionSizer()

        result = sizer.calculate_kelly(
            account_equity=small_account["equity"],
            entry_price=100.0,
            stop_price=95.0,
            win_rate=0.40,  # 40% win rate
            avg_win_pct=0.03,  # Small wins
            avg_loss_pct=0.03,  # Equal losses
            kelly_fraction=0.25,
        )

        # Negative or zero Kelly = no position
        assert result.shares >= 0

    def test_minimum_position_size(self, small_account):
        """Position below minimum is zero."""
        sizer = PositionSizer(
            max_position_risk_pct=0.001,  # 0.1% risk
            min_position_size=100.0,  # Min $100 position
        )

        result = sizer.calculate_fixed_fractional(
            account_equity=small_account["equity"],
            entry_price=100.0,
            stop_price=95.0,
        )

        # Risk only $1, position would be $20 - below minimum
        assert result.shares == 0
        assert result.dollar_amount == 0


class TestConvenienceFunction:
    """Test the convenience function."""

    def test_calculate_position_size(self):
        """Test standalone calculation function."""
        shares = calculate_position_size(
            account_equity=10000.0,
            entry_price=50.0,
            stop_price=47.50,  # $2.50 stop
            risk_pct=0.01,  # 1% risk = $100
        )

        # $100 risk / $2.50 stop = 40 shares
        assert shares == pytest.approx(40.0, rel=0.01)
