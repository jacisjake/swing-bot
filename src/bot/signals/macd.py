"""
MACD Crossover Strategy.

Buy when MACD crosses above signal line with 2 green candle confirmation.
Exit on red candle when MACD crosses below signal line.
"""

from datetime import datetime
from typing import Optional

import pandas as pd

from src.bot.signals.base import Signal, SignalDirection, SignalGenerator
from src.data.indicators import atr, macd


class MACDStrategy(SignalGenerator):
    """
    MACD Crossover strategy with candlestick confirmation.

    Entry conditions (long):
    - MACD line crosses above signal line
    - 2 consecutive green candles (close > open)

    Exit conditions:
    - Red candle (close < open) AND MACD crosses below signal line
    """

    def __init__(
        self,
        fast_period: int = 8,
        slow_period: int = 17,
        signal_period: int = 9,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
        min_signal_strength: float = 0.5,
        risk_reward_target: float = 2.0,
    ):
        """
        Initialize MACD strategy.

        Args:
            fast_period: Fast EMA period (default 8)
            slow_period: Slow EMA period (default 17)
            signal_period: Signal line EMA period (default 9)
            atr_period: ATR period for stop calculation
            atr_stop_multiplier: ATR multiplier for stop distance
            min_signal_strength: Minimum strength to generate signal
            risk_reward_target: Target R:R ratio for take profit
        """
        super().__init__(name="macd")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.min_signal_strength = min_signal_strength
        self.risk_reward_target = risk_reward_target

        # Minimum bars needed
        self.min_periods = max(slow_period + signal_period, atr_period) + 5

    def generate(
        self,
        symbol: str,
        bars: pd.DataFrame,
        current_price: Optional[float] = None,
    ) -> Optional[Signal]:
        """
        Generate MACD crossover signal.

        Args:
            symbol: Asset symbol
            bars: OHLCV DataFrame
            current_price: Optional current price

        Returns:
            Signal if entry conditions met, None otherwise
        """
        if not self.validate_bars(bars, self.min_periods):
            return None

        bars = self.normalize_bars(bars)

        close = bars["close"]
        high = bars["high"]
        low = bars["low"]
        open_price = bars["open"]

        # Calculate MACD
        macd_line, signal_line, histogram = macd(
            close, self.fast_period, self.slow_period, self.signal_period
        )

        # Calculate ATR for stop
        atr_values = atr(high, low, close, self.atr_period)

        # Get current and previous values
        current = current_price if current_price else float(close.iloc[-1])

        curr_macd = float(macd_line.iloc[-1])
        curr_signal = float(signal_line.iloc[-1])
        prev_macd = float(macd_line.iloc[-2])
        prev_signal = float(signal_line.iloc[-2])

        latest_atr = float(atr_values.iloc[-1])

        # Check for MACD crossover (current above signal, previous at or below)
        macd_crossed_up = curr_macd > curr_signal and prev_macd <= prev_signal

        # Check for 2 consecutive green candles
        candle_1_green = float(close.iloc[-1]) > float(open_price.iloc[-1])
        candle_2_green = float(close.iloc[-2]) > float(open_price.iloc[-2])
        two_green_candles = candle_1_green and candle_2_green

        if not (macd_crossed_up and two_green_candles):
            return None

        # Calculate signal strength based on histogram momentum
        hist_curr = float(histogram.iloc[-1])
        hist_prev = float(histogram.iloc[-2])

        # Strength based on histogram increasing and MACD momentum
        histogram_momentum = (hist_curr - hist_prev) / abs(hist_prev) if hist_prev != 0 else 0
        macd_momentum = abs(curr_macd - curr_signal) / abs(curr_signal) if curr_signal != 0 else 0

        strength = min(1.0, 0.5 + histogram_momentum * 0.3 + macd_momentum * 0.2)

        if strength < self.min_signal_strength:
            return None

        # Calculate stop and target
        stop_price = current - (latest_atr * self.atr_stop_multiplier)
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
                "macd": round(curr_macd, 4),
                "signal": round(curr_signal, 4),
                "histogram": round(hist_curr, 4),
                "atr": round(latest_atr, 4),
                "crossover": True,
                "green_candles": 2,
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
        - Red candle (close < open) AND MACD crosses below signal line

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
        open_price = bars["open"]

        # Calculate MACD
        macd_line, signal_line, _ = macd(
            close, self.fast_period, self.slow_period, self.signal_period
        )

        curr_macd = float(macd_line.iloc[-1])
        curr_signal = float(signal_line.iloc[-1])
        prev_macd = float(macd_line.iloc[-2])
        prev_signal = float(signal_line.iloc[-2])

        # Check for red candle
        current_close = float(close.iloc[-1])
        current_open = float(open_price.iloc[-1])
        is_red_candle = current_close < current_open

        # Check for MACD cross below signal
        macd_crossed_down = curr_macd < curr_signal and prev_macd >= prev_signal

        if direction == SignalDirection.LONG:
            if is_red_candle and macd_crossed_down:
                return True, f"Red candle + MACD crossed down (MACD={curr_macd:.4f} < Signal={curr_signal:.4f})"

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
