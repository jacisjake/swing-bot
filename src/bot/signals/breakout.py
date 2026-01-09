"""
Breakout Strategy.

Buy when price breaks above N-day high with volume confirmation.
Best for stocks showing momentum.
"""

from datetime import datetime
from typing import Optional

import pandas as pd

from src.bot.signals.base import Signal, SignalDirection, SignalGenerator
from src.data.indicators import (
    atr,
    donchian_channel,
    volume_sma,
)


class BreakoutStrategy(SignalGenerator):
    """
    Donchian Channel breakout strategy with volume confirmation.

    Entry conditions (long):
    - Price breaks above N-period high (Donchian upper)
    - Volume above average (multiplier * SMA)

    Exit conditions:
    - Price breaks below M-period low (tighter channel)
    - Stop-loss hit
    - Trailing stop hit
    """

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        volume_period: int = 20,
        volume_multiplier: float = 1.5,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
        min_signal_strength: float = 0.5,
        risk_reward_target: float = 2.0,
    ):
        """
        Initialize breakout strategy.

        Args:
            entry_period: Donchian channel period for entry (default 20)
            exit_period: Donchian channel period for exit (default 10)
            volume_period: Volume SMA period
            volume_multiplier: Volume must be this times average
            atr_period: ATR period for stop calculation
            atr_stop_multiplier: ATR multiplier for stop distance
            min_signal_strength: Minimum strength to generate signal
            risk_reward_target: Target R:R ratio for take profit
        """
        super().__init__(name="breakout")
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.volume_period = volume_period
        self.volume_multiplier = volume_multiplier
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.min_signal_strength = min_signal_strength
        self.risk_reward_target = risk_reward_target

        # Minimum bars needed
        self.min_periods = max(entry_period, volume_period, atr_period) + 5

    def generate(
        self,
        symbol: str,
        bars: pd.DataFrame,
        current_price: Optional[float] = None,
    ) -> Optional[Signal]:
        """
        Generate breakout signal.

        Args:
            symbol: Asset symbol
            bars: OHLCV DataFrame
            current_price: Optional current price

        Returns:
            Signal if breakout conditions met, None otherwise
        """
        if not self.validate_bars(bars, self.min_periods):
            return None

        bars = self.normalize_bars(bars)

        close = bars["close"]
        high = bars["high"]
        low = bars["low"]
        volume = bars["volume"]

        # Calculate indicators
        upper_channel, _, lower_channel = donchian_channel(
            high, low, self.entry_period
        )
        exit_upper, _, exit_lower = donchian_channel(
            high, low, self.exit_period
        )
        avg_volume = volume_sma(volume, self.volume_period)
        atr_values = atr(high, low, close, self.atr_period)

        # Get latest and previous values
        current = current_price if current_price else float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        prev_upper = float(upper_channel.iloc[-2])  # Previous period high
        latest_volume = float(volume.iloc[-1])
        latest_avg_volume = float(avg_volume.iloc[-1])
        latest_atr = float(atr_values.iloc[-1])
        latest_exit_lower = float(exit_lower.iloc[-1])

        # Check breakout conditions
        is_price_breakout = current > prev_upper and prev_close <= prev_upper
        is_volume_confirm = latest_volume > (latest_avg_volume * self.volume_multiplier)

        if not (is_price_breakout and is_volume_confirm):
            return None

        # Calculate signal strength
        # Higher volume and cleaner breakout = stronger
        breakout_magnitude = (current - prev_upper) / prev_upper
        volume_ratio = latest_volume / latest_avg_volume

        strength = min(1.0, (breakout_magnitude * 10 + (volume_ratio - 1) * 0.2) + 0.4)

        if strength < self.min_signal_strength:
            return None

        # Calculate stop and target
        stop_price = max(
            current - (latest_atr * self.atr_stop_multiplier),
            latest_exit_lower,  # Don't go below exit channel
        )

        risk = current - stop_price
        target_price = current + (risk * self.risk_reward_target)

        return Signal(
            symbol=symbol,
            direction=SignalDirection.LONG,
            strength=strength,
            entry_price=current,
            stop_price=stop_price,
            target_price=target_price,
            timeframe=self._detect_timeframe(bars),
            strategy=self.name,
            timestamp=datetime.now(),
            metadata={
                "breakout_level": round(prev_upper, 2),
                "breakout_pct": round(breakout_magnitude * 100, 2),
                "volume_ratio": round(volume_ratio, 2),
                "atr": round(latest_atr, 4),
                "exit_channel_low": round(latest_exit_lower, 2),
            },
        )

    def should_exit(
        self,
        symbol: str,
        bars: pd.DataFrame,
        entry_price: float,
        direction: SignalDirection,
        current_price: Optional[float] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if position should exit.

        Exit when:
        - Price breaks below exit period low (Donchian lower)
        - Trailing stop would be tighter than channel

        Args:
            symbol: Asset symbol
            bars: OHLCV DataFrame
            entry_price: Position entry price
            direction: Position direction
            current_price: Optional current price

        Returns:
            Tuple of (should_exit, reason)
        """
        if not self.validate_bars(bars, self.min_periods):
            return False, None

        bars = self.normalize_bars(bars)

        close = bars["close"]
        high = bars["high"]
        low = bars["low"]

        _, _, exit_lower = donchian_channel(high, low, self.exit_period)

        current = current_price if current_price else float(close.iloc[-1])
        latest_exit_lower = float(exit_lower.iloc[-1])

        if direction == SignalDirection.LONG:
            # Exit long when price breaks below exit channel
            if current < latest_exit_lower:
                return True, f"Price broke exit channel ({current:.2f} < {latest_exit_lower:.2f})"

        else:
            # For short positions (if implemented)
            _, exit_upper, _ = donchian_channel(high, low, self.exit_period)
            latest_exit_upper = float(exit_upper.iloc[-1])

            if current > latest_exit_upper:
                return True, f"Price broke exit channel ({current:.2f} > {latest_exit_upper:.2f})"

        return False, None

    def _detect_timeframe(self, bars: pd.DataFrame) -> str:
        """Detect timeframe from bar index."""
        if len(bars) < 2:
            return "unknown"

        if hasattr(bars.index, "to_pydatetime"):
            try:
                delta = bars.index[-1] - bars.index[-2]
                minutes = delta.total_seconds() / 60

                if minutes <= 1:
                    return "1Min"
                elif minutes <= 5:
                    return "5Min"
                elif minutes <= 15:
                    return "15Min"
                elif minutes <= 60:
                    return "1Hour"
                elif minutes <= 240:
                    return "4Hour"
                elif minutes <= 1440:
                    return "1Day"
                else:
                    return "1Week"
            except Exception:
                pass

        return "unknown"
