"""
Mean Reversion Strategy.

Buy when oversold (RSI + Bollinger Bands), sell when normalized.
Best for crypto in ranging markets.
"""

from datetime import datetime
from typing import Optional

import pandas as pd

from src.bot.signals.base import Signal, SignalDirection, SignalGenerator
from src.data.indicators import (
    atr,
    bollinger_bands,
    rsi,
)


class MeanReversionStrategy(SignalGenerator):
    """
    Mean reversion strategy using RSI and Bollinger Bands.

    Entry conditions (long):
    - RSI below oversold threshold (default 30)
    - Price below lower Bollinger Band

    Exit conditions:
    - RSI above exit threshold (default 50)
    - Price reaches middle Bollinger Band
    - Stop-loss hit
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
        rsi_exit: int = 50,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
        min_signal_strength: float = 0.5,
    ):
        """
        Initialize mean reversion strategy.

        Args:
            rsi_period: RSI calculation period
            rsi_oversold: RSI level for oversold (entry)
            rsi_overbought: RSI level for overbought
            rsi_exit: RSI level for exit
            bb_period: Bollinger Band period
            bb_std: Bollinger Band standard deviation
            atr_period: ATR period for stop calculation
            atr_stop_multiplier: ATR multiplier for stop distance
            min_signal_strength: Minimum strength to generate signal
        """
        super().__init__(name="mean_reversion")
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.rsi_exit = rsi_exit
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.min_signal_strength = min_signal_strength

        # Minimum bars needed
        self.min_periods = max(rsi_period, bb_period, atr_period) + 5

    def generate(
        self,
        symbol: str,
        bars: pd.DataFrame,
        current_price: Optional[float] = None,
    ) -> Optional[Signal]:
        """
        Generate mean reversion signal.

        Args:
            symbol: Asset symbol
            bars: OHLCV DataFrame
            current_price: Optional current price

        Returns:
            Signal if oversold conditions met, None otherwise
        """
        if not self.validate_bars(bars, self.min_periods):
            return None

        bars = self.normalize_bars(bars)

        close = bars["close"]
        high = bars["high"]
        low = bars["low"]

        # Calculate indicators
        rsi_values = rsi(close, self.rsi_period)
        upper_bb, middle_bb, lower_bb = bollinger_bands(
            close, self.bb_period, self.bb_std
        )
        atr_values = atr(high, low, close, self.atr_period)

        # Get latest values
        current = current_price if current_price else float(close.iloc[-1])
        latest_rsi = float(rsi_values.iloc[-1])
        latest_lower_bb = float(lower_bb.iloc[-1])
        latest_middle_bb = float(middle_bb.iloc[-1])
        latest_atr = float(atr_values.iloc[-1])

        # Check oversold conditions
        is_rsi_oversold = latest_rsi < self.rsi_oversold
        is_below_bb = current < latest_lower_bb

        if not (is_rsi_oversold and is_below_bb):
            return None

        # Calculate signal strength
        # Deeper oversold = stronger signal
        rsi_depth = (self.rsi_oversold - latest_rsi) / self.rsi_oversold
        bb_depth = (latest_lower_bb - current) / latest_lower_bb

        strength = min(1.0, (rsi_depth + bb_depth) / 2 + 0.3)

        if strength < self.min_signal_strength:
            return None

        # Calculate stop and target
        stop_price = current - (latest_atr * self.atr_stop_multiplier)
        target_price = latest_middle_bb  # Target middle band

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
                "rsi": round(latest_rsi, 2),
                "lower_bb": round(latest_lower_bb, 2),
                "middle_bb": round(latest_middle_bb, 2),
                "atr": round(latest_atr, 4),
                "rsi_depth": round(rsi_depth, 3),
                "bb_depth": round(bb_depth, 3),
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
        - RSI rises above exit threshold
        - Price reaches middle Bollinger Band
        - For shorts: RSI falls below exit threshold

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

        rsi_values = rsi(close, self.rsi_period)
        _, middle_bb, _ = bollinger_bands(close, self.bb_period, self.bb_std)

        current = current_price if current_price else float(close.iloc[-1])
        latest_rsi = float(rsi_values.iloc[-1])
        latest_middle_bb = float(middle_bb.iloc[-1])

        if direction == SignalDirection.LONG:
            # Exit long when RSI normalized or target hit
            if latest_rsi >= self.rsi_exit:
                return True, f"RSI normalized ({latest_rsi:.1f} >= {self.rsi_exit})"

            if current >= latest_middle_bb:
                return True, f"Price reached middle BB ({current:.2f} >= {latest_middle_bb:.2f})"

        else:
            # Exit short when RSI normalized (from overbought)
            if latest_rsi <= self.rsi_exit:
                return True, f"RSI normalized ({latest_rsi:.1f} <= {self.rsi_exit})"

            if current <= latest_middle_bb:
                return True, f"Price reached middle BB ({current:.2f} <= {latest_middle_bb:.2f})"

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
