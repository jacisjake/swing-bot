"""
Position Sizer - Calculate optimal position sizes based on risk.

Implements multiple position sizing methods:
- Fixed fractional (risk X% of account per trade)
- ATR-based (size based on volatility)
- Kelly Criterion (optimal growth sizing)

CRITICAL for small accounts: Never risk more than you can afford to lose.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config import settings


class SizingMethod(str, Enum):
    FIXED_FRACTIONAL = "fixed_fractional"
    ATR_BASED = "atr_based"
    KELLY = "kelly"


@dataclass
class PositionSize:
    """Result of position sizing calculation."""

    shares: float
    dollar_amount: float
    risk_amount: float
    stop_distance: float
    method: SizingMethod

    # Constraints applied
    capped_by_max_position: bool = False
    capped_by_buying_power: bool = False

    def to_dict(self) -> dict:
        return {
            "shares": self.shares,
            "dollar_amount": self.dollar_amount,
            "risk_amount": self.risk_amount,
            "stop_distance": self.stop_distance,
            "method": self.method.value,
            "capped_by_max_position": self.capped_by_max_position,
            "capped_by_buying_power": self.capped_by_buying_power,
        }


class PositionSizer:
    """
    Calculate position sizes based on risk parameters.

    For a $1000 account with 2% risk per trade:
    - Max loss per trade = $20
    - If stop is 5% below entry, position size = $400
    - If stop is 10% below entry, position size = $200

    This ensures consistent risk regardless of stock price or volatility.
    """

    def __init__(
        self,
        max_position_risk_pct: Optional[float] = None,
        max_position_pct: float = 0.25,  # Max 25% of portfolio in one position
        min_position_size: float = 1.0,  # Minimum $1 position
    ):
        """
        Initialize position sizer.

        Args:
            max_position_risk_pct: Max % of account to risk per trade (default from settings)
            max_position_pct: Max % of account in single position
            min_position_size: Minimum position size in dollars
        """
        self.max_position_risk_pct = (
            max_position_risk_pct or settings.max_position_risk_pct
        )
        self.max_position_pct = max_position_pct
        self.min_position_size = min_position_size

    def calculate_fixed_fractional(
        self,
        account_equity: float,
        entry_price: float,
        stop_price: float,
        buying_power: Optional[float] = None,
    ) -> PositionSize:
        """
        Calculate position size using fixed fractional method.

        Risk a fixed percentage of account on each trade.

        Args:
            account_equity: Current account equity
            entry_price: Planned entry price
            stop_price: Stop-loss price
            buying_power: Available buying power (defaults to equity)

        Returns:
            PositionSize with calculated values
        """
        buying_power = buying_power or account_equity

        # Calculate risk amount
        risk_amount = account_equity * self.max_position_risk_pct

        # Calculate stop distance (must be > 0)
        stop_distance = abs(entry_price - stop_price)
        if stop_distance == 0:
            logger.warning("Stop distance is 0, cannot calculate position size")
            return PositionSize(
                shares=0,
                dollar_amount=0,
                risk_amount=risk_amount,
                stop_distance=0,
                method=SizingMethod.FIXED_FRACTIONAL,
            )

        # Calculate position size based on risk
        shares = risk_amount / stop_distance
        dollar_amount = shares * entry_price

        # Apply constraints
        capped_by_max = False
        capped_by_bp = False

        # Cap by max position size (% of portfolio)
        max_position_dollars = account_equity * self.max_position_pct
        if dollar_amount > max_position_dollars:
            dollar_amount = max_position_dollars
            shares = dollar_amount / entry_price
            capped_by_max = True

        # Cap by buying power
        if dollar_amount > buying_power:
            dollar_amount = buying_power
            shares = dollar_amount / entry_price
            capped_by_bp = True

        # Ensure minimum size
        if dollar_amount < self.min_position_size:
            shares = 0
            dollar_amount = 0

        return PositionSize(
            shares=round(shares, 6),  # Support fractional shares
            dollar_amount=round(dollar_amount, 2),
            risk_amount=round(risk_amount, 2),
            stop_distance=round(stop_distance, 4),
            method=SizingMethod.FIXED_FRACTIONAL,
            capped_by_max_position=capped_by_max,
            capped_by_buying_power=capped_by_bp,
        )

    def calculate_momentum_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_price: float,
        buying_power: float,
        max_equity_pct: float = 0.90,
    ) -> PositionSize:
        """
        Position sizing for momentum day trading (cash account style).

        Uses majority of buying power but still risk-constrained.
        For low-priced stocks ($2-$10), rounds to whole shares.

        Args:
            account_equity: Current account equity
            entry_price: Planned entry price
            stop_price: Stop-loss price
            buying_power: Available buying power
            max_equity_pct: Max % of equity to deploy (0.90 = 90%)

        Returns:
            PositionSize with calculated values
        """
        # Max dollar amount: min of buying power and equity cap
        max_dollars = min(buying_power, account_equity * max_equity_pct)

        # Calculate risk amount (still respect max risk per trade)
        risk_amount = account_equity * self.max_position_risk_pct
        stop_distance = abs(entry_price - stop_price)

        if stop_distance == 0:
            logger.warning("Stop distance is 0, cannot size position")
            return PositionSize(
                shares=0, dollar_amount=0, risk_amount=risk_amount,
                stop_distance=0, method=SizingMethod.FIXED_FRACTIONAL,
            )

        # Shares from risk limit
        shares_from_risk = risk_amount / stop_distance

        # Shares from dollar limit
        shares_from_dollars = max_dollars / entry_price if entry_price > 0 else 0

        # Use the smaller (risk-constrained)
        shares = min(shares_from_risk, shares_from_dollars)

        # Round to whole shares for low-priced stocks
        if entry_price < 10:
            shares = int(shares)

        if shares <= 0:
            return PositionSize(
                shares=0, dollar_amount=0, risk_amount=risk_amount,
                stop_distance=round(stop_distance, 4),
                method=SizingMethod.FIXED_FRACTIONAL,
            )

        dollar_amount = shares * entry_price
        actual_risk = shares * stop_distance

        capped_by_bp = shares == int(shares_from_dollars) if entry_price < 10 else (
            abs(shares - shares_from_dollars) < 0.01
        )

        logger.info(
            f"Momentum size: {shares} shares @ ${entry_price:.2f} = ${dollar_amount:.2f}, "
            f"risk=${actual_risk:.2f} ({actual_risk/account_equity*100:.1f}% of equity)"
        )

        return PositionSize(
            shares=shares,
            dollar_amount=round(dollar_amount, 2),
            risk_amount=round(actual_risk, 2),
            stop_distance=round(stop_distance, 4),
            method=SizingMethod.FIXED_FRACTIONAL,
            capped_by_max_position=False,
            capped_by_buying_power=capped_by_bp,
        )

    def calculate_atr_based(
        self,
        account_equity: float,
        entry_price: float,
        atr: float,
        atr_multiplier: float = 2.0,
        buying_power: Optional[float] = None,
    ) -> PositionSize:
        """
        Calculate position size based on ATR (Average True Range).

        Uses ATR to set stop distance dynamically based on volatility.
        Higher volatility = wider stop = smaller position.

        Args:
            account_equity: Current account equity
            entry_price: Planned entry price
            atr: Current ATR value
            atr_multiplier: Stop distance as multiple of ATR (default 2x)
            buying_power: Available buying power

        Returns:
            PositionSize with calculated values
        """
        # Stop distance is ATR * multiplier
        stop_distance = atr * atr_multiplier
        stop_price = entry_price - stop_distance  # Assuming long position

        # Use fixed fractional with calculated stop
        result = self.calculate_fixed_fractional(
            account_equity=account_equity,
            entry_price=entry_price,
            stop_price=stop_price,
            buying_power=buying_power,
        )

        # Override method
        return PositionSize(
            shares=result.shares,
            dollar_amount=result.dollar_amount,
            risk_amount=result.risk_amount,
            stop_distance=stop_distance,
            method=SizingMethod.ATR_BASED,
            capped_by_max_position=result.capped_by_max_position,
            capped_by_buying_power=result.capped_by_buying_power,
        )

    def calculate_kelly(
        self,
        account_equity: float,
        entry_price: float,
        stop_price: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        kelly_fraction: float = 0.25,  # Use 1/4 Kelly for safety
        buying_power: Optional[float] = None,
    ) -> PositionSize:
        """
        Calculate position size using Kelly Criterion.

        Kelly = W - (1-W)/R
        Where:
            W = Win rate (probability of winning)
            R = Win/Loss ratio (avg win / avg loss)

        We use fractional Kelly (default 1/4) to reduce volatility.

        Args:
            account_equity: Current account equity
            entry_price: Planned entry price
            stop_price: Stop-loss price
            win_rate: Historical win rate (0-1)
            avg_win_pct: Average winning trade % (e.g., 0.05 for 5%)
            avg_loss_pct: Average losing trade % (e.g., 0.02 for 2%)
            kelly_fraction: Fraction of Kelly to use (0.25 = 1/4 Kelly)
            buying_power: Available buying power

        Returns:
            PositionSize with calculated values
        """
        buying_power = buying_power or account_equity

        # Calculate Kelly percentage
        if avg_loss_pct == 0:
            kelly_pct = 0
        else:
            win_loss_ratio = avg_win_pct / avg_loss_pct
            kelly_pct = win_rate - (1 - win_rate) / win_loss_ratio

        # Kelly can be negative (don't trade) or > 1 (impossible)
        kelly_pct = max(0, min(1, kelly_pct))

        # Apply fraction for safety
        kelly_pct *= kelly_fraction

        # Calculate position size
        dollar_amount = account_equity * kelly_pct
        shares = dollar_amount / entry_price if entry_price > 0 else 0

        # Calculate actual risk (for the position size)
        stop_distance = abs(entry_price - stop_price)
        risk_amount = shares * stop_distance

        # Apply constraints
        capped_by_max = False
        capped_by_bp = False

        max_position_dollars = account_equity * self.max_position_pct
        if dollar_amount > max_position_dollars:
            dollar_amount = max_position_dollars
            shares = dollar_amount / entry_price
            risk_amount = shares * stop_distance
            capped_by_max = True

        if dollar_amount > buying_power:
            dollar_amount = buying_power
            shares = dollar_amount / entry_price
            risk_amount = shares * stop_distance
            capped_by_bp = True

        if dollar_amount < self.min_position_size:
            shares = 0
            dollar_amount = 0
            risk_amount = 0

        return PositionSize(
            shares=round(shares, 6),
            dollar_amount=round(dollar_amount, 2),
            risk_amount=round(risk_amount, 2),
            stop_distance=round(stop_distance, 4),
            method=SizingMethod.KELLY,
            capped_by_max_position=capped_by_max,
            capped_by_buying_power=capped_by_bp,
        )

    def calculate_from_bars(
        self,
        account_equity: float,
        entry_price: float,
        bars: pd.DataFrame,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        buying_power: Optional[float] = None,
    ) -> PositionSize:
        """
        Calculate position size from price bars (convenience method).

        Calculates ATR from bars and uses ATR-based sizing.

        Args:
            account_equity: Current account equity
            entry_price: Planned entry price
            bars: DataFrame with 'high', 'low', 'close' columns
            atr_period: Period for ATR calculation
            atr_multiplier: Stop distance as multiple of ATR
            buying_power: Available buying power

        Returns:
            PositionSize with calculated values
        """
        atr = self._calculate_atr(bars, atr_period)

        return self.calculate_atr_based(
            account_equity=account_equity,
            entry_price=entry_price,
            atr=atr,
            atr_multiplier=atr_multiplier,
            buying_power=buying_power,
        )

    def _calculate_atr(self, bars: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range from price bars."""
        if len(bars) < period:
            # Not enough data, use simple high-low range
            return float((bars["high"] - bars["low"]).mean())

        high = bars["high"]
        low = bars["low"]
        close = bars["close"].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]

        return float(atr) if not np.isnan(atr) else float(tr1.mean())


def calculate_position_size(
    account_equity: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float = 0.02,
) -> float:
    """
    Simple position size calculation (convenience function).

    Args:
        account_equity: Account equity
        entry_price: Entry price
        stop_price: Stop-loss price
        risk_pct: Percentage of account to risk (default 2%)

    Returns:
        Number of shares to buy
    """
    sizer = PositionSizer(max_position_risk_pct=risk_pct)
    result = sizer.calculate_fixed_fractional(
        account_equity=account_equity,
        entry_price=entry_price,
        stop_price=stop_price,
    )
    return result.shares
