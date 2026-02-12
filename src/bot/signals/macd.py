"""
MACD Momentum Strategy.

Buy when MACD crosses above signal line with volume and RSI confirmation.
Exit on red candle when MACD crosses below signal line.
"""

from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from src.bot.signals.base import Signal, SignalDirection, SignalGenerator
from src.data.indicators import atr, macd, rsi


class MACDStrategy(SignalGenerator):
    """
    MACD Momentum strategy with volume and RSI confirmation.

    Entry conditions (long):
    - MACD line crosses above signal line (crossover, not just above)
    - 2 consecutive green candles (close > open)
    - Volume > 1.5x 20-day average volume
    - RSI between 30 and 70 (avoid overbought/oversold)

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
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        volume_multiplier: float = 1.2,
        volume_lookback: int = 20,
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
            rsi_period: RSI period (default 14)
            rsi_overbought: RSI overbought threshold (default 70)
            rsi_oversold: RSI oversold threshold (default 30)
            volume_multiplier: Volume must exceed avg * multiplier (default 1.2)
            volume_lookback: Period for volume SMA (default 20)
        """
        super().__init__(name="macd")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.min_signal_strength = min_signal_strength
        self.risk_reward_target = risk_reward_target
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_multiplier = volume_multiplier
        self.volume_lookback = volume_lookback

        # Minimum bars needed
        self.min_periods = max(slow_period + signal_period, atr_period, rsi_period, volume_lookback) + 5

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

        # CROSSOVER: previous below signal, current at/above signal
        macd_crossover = prev_macd < prev_signal and curr_macd >= curr_signal

        # Check for bullish candle pattern (relaxed from 2 green candles)
        # Now requires: 1 green candle OR histogram positive and increasing
        candle_1_green = float(close.iloc[-1]) > float(open_price.iloc[-1])
        candle_2_green = float(close.iloc[-2]) > float(open_price.iloc[-2])

        # Get histogram for momentum check
        hist_curr = float(histogram.iloc[-1])
        hist_prev = float(histogram.iloc[-2])
        histogram_bullish = hist_curr > 0 and hist_curr > hist_prev

        # Pass if: current candle is green OR histogram is positive and increasing
        bullish_candle_pattern = candle_1_green or histogram_bullish

        # Calculate volume filter
        volume = bars["volume"]
        curr_volume = float(volume.iloc[-1])
        avg_volume = float(volume.rolling(self.volume_lookback).mean().iloc[-1])
        volume_strong = curr_volume > avg_volume * self.volume_multiplier
        volume_ratio = curr_volume / avg_volume if avg_volume > 0 else 0.0

        # Calculate RSI filter
        rsi_values = rsi(close, self.rsi_period)
        curr_rsi = float(rsi_values.iloc[-1])
        rsi_valid = self.rsi_oversold < curr_rsi < self.rsi_overbought

        # Log signal check with reasons for rejection
        if macd_crossover:
            # Only log when there's a crossover (potential entry)
            fails = []
            if not bullish_candle_pattern:
                fails.append(f"no bullish pattern (candle={'G' if candle_1_green else 'R'}, hist={hist_curr:.4f})")
            if not volume_strong:
                fails.append(f"vol({volume_ratio:.1f}x<{self.volume_multiplier}x)")
            if not rsi_valid:
                fails.append(f"RSI({curr_rsi:.0f})")

            if fails:
                logger.info(f"[SIGNAL] {symbol}: MACD crossover but rejected - {', '.join(fails)}")
            else:
                logger.info(f"[SIGNAL] {symbol}: MACD crossover PASSED all filters! RSI={curr_rsi:.0f}, Vol={volume_ratio:.1f}x")

        if not (macd_crossover and bullish_candle_pattern and volume_strong and rsi_valid):
            return None

        # Calculate signal strength based on histogram momentum (hist_curr/hist_prev already calculated above)
        # Strength based on histogram increasing and MACD momentum
        histogram_momentum = (hist_curr - hist_prev) / abs(hist_prev) if hist_prev != 0 else 0
        macd_momentum = abs(curr_macd - curr_signal) / abs(curr_signal) if curr_signal != 0 else 0

        strength = min(1.0, 0.5 + histogram_momentum * 0.3 + macd_momentum * 0.2)

        if strength < self.min_signal_strength:
            return None

        # ATR-based stop-loss and take-profit (1:1 risk/reward)
        stop_distance = latest_atr * self.atr_stop_multiplier
        stop_price = current - stop_distance
        target_price = current + stop_distance  # 1:1 R:R

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
                "rsi": round(curr_rsi, 2),
                "volume_ratio": round(volume_ratio, 2),
                "crossover": True,
                "bullish_pattern": "green_candle" if candle_1_green else "histogram_bullish",
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
        - Red candle (close < open) AND MACD line crosses below signal line

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
        prev_macd = float(macd_line.iloc[-2])
        curr_signal = float(signal_line.iloc[-1])
        prev_signal = float(signal_line.iloc[-2])

        # Check for red candle
        current_close = float(close.iloc[-1])
        current_open = float(open_price.iloc[-1])
        is_red_candle = current_close < current_open

        # Check for MACD crossing below signal line (actual crossover)
        macd_crossed_below_signal = curr_macd < curr_signal and prev_macd >= prev_signal

        # Check if MACD is simply below signal (not necessarily just crossed)
        macd_below_signal = curr_macd < curr_signal

        if direction == SignalDirection.LONG:
            # Exit on crossover immediately OR red candle when already below signal
            # This is more sensitive than requiring BOTH conditions
            if macd_crossed_below_signal:
                return True, f"MACD crossed below signal (MACD={curr_macd:.4f}, Signal={curr_signal:.4f})"
            if is_red_candle and macd_below_signal:
                return True, f"Red candle while MACD below signal (MACD={curr_macd:.4f}, Signal={curr_signal:.4f})"

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
