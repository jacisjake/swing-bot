"""
Stop Manager - Calculate and manage stop-loss levels.

Provides multiple stop-loss strategies:
- Fixed percentage stops
- ATR-based stops (volatility adjusted)
- Support/resistance-based stops
- Trailing stops (percentage or ATR-based)
- Breakeven stops (move stop to entry after profit)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class StopType(str, Enum):
    FIXED_PERCENT = "fixed_percent"
    ATR_BASED = "atr_based"
    SUPPORT_LEVEL = "support_level"
    TRAILING_PERCENT = "trailing_percent"
    TRAILING_ATR = "trailing_atr"
    BREAKEVEN = "breakeven"


@dataclass
class StopLevel:
    """Calculated stop-loss level."""

    price: float
    type: StopType
    distance_pct: float  # Distance from entry as percentage
    risk_reward: Optional[float] = None  # R:R if target provided

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "type": self.type.value,
            "distance_pct": self.distance_pct,
            "risk_reward": self.risk_reward,
        }


class StopManager:
    """
    Calculate and manage stop-loss levels.

    For swing trading, stops should be:
    - Wide enough to avoid noise
    - Tight enough to limit losses
    - Based on market structure when possible
    """

    def __init__(
        self,
        default_stop_pct: float = 0.05,  # 5% default stop
        atr_multiplier: float = 2.0,  # 2x ATR for volatility stops
        trailing_activation_pct: float = 0.02,  # Activate trailing after 2% profit
    ):
        """
        Initialize stop manager.

        Args:
            default_stop_pct: Default stop-loss percentage
            atr_multiplier: ATR multiplier for ATR-based stops
            trailing_activation_pct: Profit required to activate trailing stop
        """
        self.default_stop_pct = default_stop_pct
        self.atr_multiplier = atr_multiplier
        self.trailing_activation_pct = trailing_activation_pct

    def calculate_fixed_stop(
        self,
        entry_price: float,
        stop_pct: Optional[float] = None,
        side: str = "long",
        target_price: Optional[float] = None,
    ) -> StopLevel:
        """
        Calculate fixed percentage stop-loss.

        Args:
            entry_price: Entry price
            stop_pct: Stop distance as percentage (default from init)
            side: "long" or "short"
            target_price: Optional target for R:R calculation

        Returns:
            StopLevel with calculated values
        """
        stop_pct = stop_pct or self.default_stop_pct

        if side == "long":
            stop_price = entry_price * (1 - stop_pct)
        else:
            stop_price = entry_price * (1 + stop_pct)

        risk_reward = None
        if target_price:
            risk = abs(entry_price - stop_price)
            reward = abs(target_price - entry_price)
            risk_reward = reward / risk if risk > 0 else 0

        return StopLevel(
            price=round(stop_price, 4),
            type=StopType.FIXED_PERCENT,
            distance_pct=stop_pct,
            risk_reward=risk_reward,
        )

    def calculate_atr_stop(
        self,
        entry_price: float,
        atr: float,
        multiplier: Optional[float] = None,
        side: str = "long",
        target_price: Optional[float] = None,
    ) -> StopLevel:
        """
        Calculate ATR-based stop-loss.

        Stop is placed at entry - (ATR * multiplier) for long positions.
        This adapts to market volatility automatically.

        Args:
            entry_price: Entry price
            atr: Current ATR value
            multiplier: ATR multiplier (default from init)
            side: "long" or "short"
            target_price: Optional target for R:R calculation

        Returns:
            StopLevel with calculated values
        """
        multiplier = multiplier or self.atr_multiplier
        stop_distance = atr * multiplier

        if side == "long":
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance

        distance_pct = stop_distance / entry_price

        risk_reward = None
        if target_price:
            risk = abs(entry_price - stop_price)
            reward = abs(target_price - entry_price)
            risk_reward = reward / risk if risk > 0 else 0

        return StopLevel(
            price=round(stop_price, 4),
            type=StopType.ATR_BASED,
            distance_pct=round(distance_pct, 4),
            risk_reward=risk_reward,
        )

    def calculate_support_stop(
        self,
        entry_price: float,
        bars: pd.DataFrame,
        lookback: int = 20,
        buffer_pct: float = 0.005,  # 0.5% below support
        side: str = "long",
        target_price: Optional[float] = None,
    ) -> StopLevel:
        """
        Calculate stop-loss based on support/resistance level.

        Finds recent swing low (for long) or swing high (for short)
        and places stop just beyond it.

        Args:
            entry_price: Entry price
            bars: DataFrame with 'high', 'low', 'close' columns
            lookback: Number of bars to look back for support/resistance
            buffer_pct: Buffer below support (or above resistance)
            side: "long" or "short"
            target_price: Optional target for R:R calculation

        Returns:
            StopLevel with calculated values
        """
        if len(bars) < lookback:
            # Not enough data, fall back to fixed stop
            return self.calculate_fixed_stop(entry_price, side=side)

        recent_bars = bars.tail(lookback)

        if side == "long":
            # Find swing low (support)
            support = float(recent_bars["low"].min())
            stop_price = support * (1 - buffer_pct)
        else:
            # Find swing high (resistance)
            resistance = float(recent_bars["high"].max())
            stop_price = resistance * (1 + buffer_pct)

        distance_pct = abs(entry_price - stop_price) / entry_price

        risk_reward = None
        if target_price:
            risk = abs(entry_price - stop_price)
            reward = abs(target_price - entry_price)
            risk_reward = reward / risk if risk > 0 else 0

        return StopLevel(
            price=round(stop_price, 4),
            type=StopType.SUPPORT_LEVEL,
            distance_pct=round(distance_pct, 4),
            risk_reward=risk_reward,
        )

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,  # Highest price since entry (for long)
        trail_pct: float,
        side: str = "long",
    ) -> StopLevel:
        """
        Calculate trailing stop level.

        Trail stop follows price, locking in profits as price moves favorably.

        Args:
            entry_price: Original entry price
            current_price: Current price
            highest_price: Highest (long) or lowest (short) price since entry
            trail_pct: Trail distance as percentage
            side: "long" or "short"

        Returns:
            StopLevel with calculated values
        """
        if side == "long":
            # Stop trails below highest price
            stop_price = highest_price * (1 - trail_pct)
            # But never below entry for breakeven protection
            stop_price = max(stop_price, entry_price)
        else:
            # For shorts, trail above lowest price
            stop_price = highest_price * (1 + trail_pct)  # highest_price is lowest for shorts
            stop_price = min(stop_price, entry_price)

        distance_pct = abs(current_price - stop_price) / current_price

        return StopLevel(
            price=round(stop_price, 4),
            type=StopType.TRAILING_PERCENT,
            distance_pct=round(distance_pct, 4),
        )

    def calculate_trailing_atr_stop(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        atr: float,
        multiplier: Optional[float] = None,
        side: str = "long",
    ) -> StopLevel:
        """
        Calculate ATR-based trailing stop.

        Similar to trailing stop but uses ATR for dynamic trail distance.

        Args:
            entry_price: Original entry price
            current_price: Current price
            highest_price: Highest price since entry
            atr: Current ATR value
            multiplier: ATR multiplier
            side: "long" or "short"

        Returns:
            StopLevel with calculated values
        """
        multiplier = multiplier or self.atr_multiplier
        trail_distance = atr * multiplier

        if side == "long":
            stop_price = highest_price - trail_distance
            stop_price = max(stop_price, entry_price)  # Breakeven floor
        else:
            stop_price = highest_price + trail_distance
            stop_price = min(stop_price, entry_price)

        distance_pct = abs(current_price - stop_price) / current_price

        return StopLevel(
            price=round(stop_price, 4),
            type=StopType.TRAILING_ATR,
            distance_pct=round(distance_pct, 4),
        )

    def calculate_breakeven_stop(
        self,
        entry_price: float,
        current_price: float,
        original_stop: float,
        activation_pct: Optional[float] = None,
        buffer_pct: float = 0.001,  # Small buffer above entry
        side: str = "long",
    ) -> StopLevel:
        """
        Calculate breakeven stop (move stop to entry after profit).

        Once position is profitable by activation_pct, move stop to entry.

        Args:
            entry_price: Entry price
            current_price: Current price
            original_stop: Original stop-loss price
            activation_pct: Profit required to activate (default from init)
            buffer_pct: Buffer above entry (to cover commissions)
            side: "long" or "short"

        Returns:
            StopLevel with stop at entry or original stop
        """
        activation_pct = activation_pct or self.trailing_activation_pct

        if side == "long":
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= activation_pct:
                # Move to breakeven (plus buffer)
                stop_price = entry_price * (1 + buffer_pct)
                distance_pct = (current_price - stop_price) / current_price
                return StopLevel(
                    price=round(stop_price, 4),
                    type=StopType.BREAKEVEN,
                    distance_pct=round(distance_pct, 4),
                )
        else:
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct >= activation_pct:
                stop_price = entry_price * (1 - buffer_pct)
                distance_pct = (stop_price - current_price) / current_price
                return StopLevel(
                    price=round(stop_price, 4),
                    type=StopType.BREAKEVEN,
                    distance_pct=round(distance_pct, 4),
                )

        # Not yet activated, return original stop
        distance_pct = abs(current_price - original_stop) / current_price
        return StopLevel(
            price=original_stop,
            type=StopType.FIXED_PERCENT,
            distance_pct=round(distance_pct, 4),
        )

    def calculate_stop_from_bars(
        self,
        entry_price: float,
        bars: pd.DataFrame,
        atr_period: int = 14,
        side: str = "long",
        target_price: Optional[float] = None,
    ) -> StopLevel:
        """
        Calculate best stop using bar data (convenience method).

        Uses ATR-based stop with ATR calculated from bars.

        Args:
            entry_price: Entry price
            bars: DataFrame with OHLC data
            atr_period: Period for ATR calculation
            side: "long" or "short"
            target_price: Optional target for R:R

        Returns:
            StopLevel with ATR-based stop
        """
        atr = self._calculate_atr(bars, atr_period)
        return self.calculate_atr_stop(
            entry_price=entry_price,
            atr=atr,
            side=side,
            target_price=target_price,
        )

    def _calculate_atr(self, bars: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR from price bars."""
        if len(bars) < period:
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

    def should_tighten_stop(
        self,
        entry_price: float,
        current_price: float,
        current_stop: float,
        profit_threshold: float = 0.05,  # 5% profit
        side: str = "long",
    ) -> bool:
        """
        Check if stop should be tightened based on profit.

        After significant profit, tighten stop to protect gains.

        Args:
            entry_price: Entry price
            current_price: Current price
            current_stop: Current stop-loss price
            profit_threshold: Profit level to trigger tightening
            side: "long" or "short"

        Returns:
            True if stop should be tightened
        """
        if side == "long":
            profit_pct = (current_price - entry_price) / entry_price
            # Tighten if profitable and stop is below entry
            return profit_pct >= profit_threshold and current_stop < entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price
            return profit_pct >= profit_threshold and current_stop > entry_price
