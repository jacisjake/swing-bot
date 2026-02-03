"""
MACD 3-System Trading Strategy.

Implements the proven MACD strategy with 71% win rate from the spec document.

System 1: Trend Trading - Zero line filter + crossover
System 2: Reversal - Divergence detection + histogram patterns
System 3: Confirmation - Multi-timeframe alignment + key levels
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.bot.signals.base import Signal, SignalDirection, SignalGenerator
from src.data.indicators import atr, ema, macd, rsi


class SystemType(int, Enum):
    """Which system generated the signal."""
    TREND = 1
    REVERSAL = 2
    CONFIRMATION = 3


class HistogramPattern(str, Enum):
    """Histogram reversal patterns."""
    FLIP = "flip"
    SHRINKING = "shrinking"
    NONE = "none"


class DivergenceType(str, Enum):
    """Price/MACD divergence types."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NONE = "none"


@dataclass
class SystemSignal:
    """Signal from one of the three systems."""
    system: SystemType
    direction: SignalDirection
    confidence: float  # 0-1
    reason: str
    metadata: dict


class MACDThreeSystemStrategy(SignalGenerator):
    """
    MACD 3-System Strategy for swing trading.

    Combines three complementary systems:
    1. Trend Trading: Zero line filter + MACD/signal crossover
    2. Reversal: Divergence + histogram patterns
    3. Confirmation: Multi-timeframe alignment

    Entry Rules:
    - System 1: MACD > 0 for longs, crossover above signal
    - System 2: Divergence + histogram flip/shrinking
    - System 3: Daily bias + 4H confirmation + 1H entry

    Exit Rules:
    - Opposite crossover on entry timeframe
    - Zero line violation
    - Trailing stop after 1R profit
    """

    def __init__(
        self,
        # MACD parameters (doc spec: 12/26/9)
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        # Zero line filter
        zero_line_buffer: float = 0.5,
        # Risk management
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
        min_reward_ratio: float = 1.5,
        # Signal filtering
        min_signal_strength: float = 0.5,
        # Volume filter (relaxed from 1.2 to allow more trades)
        volume_multiplier: float = 1.0,
        volume_lookback: int = 20,
        # RSI filter
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        # Divergence detection
        divergence_lookback: int = 20,
        # Histogram pattern detection
        histogram_flip_min_bars: int = 5,
        # Candle quality filter
        candle_quality_lookback: int = 3,
        min_body_ratio: float = 0.4,
        # Key level confluence
        key_level_lookback: int = 20,
        key_level_proximity_pct: float = 0.015,
        key_level_confidence_boost: float = 0.1,
    ):
        """Initialize 3-system MACD strategy."""
        super().__init__(name="macd_3system")

        # MACD params
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.zero_line_buffer = zero_line_buffer

        # Risk params
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.min_reward_ratio = min_reward_ratio
        self.min_signal_strength = min_signal_strength

        # Volume params
        self.volume_multiplier = volume_multiplier
        self.volume_lookback = volume_lookback

        # RSI params
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

        # Pattern detection params
        self.divergence_lookback = divergence_lookback
        self.histogram_flip_min_bars = histogram_flip_min_bars

        # Candle quality params
        self.candle_quality_lookback = candle_quality_lookback
        self.min_body_ratio = min_body_ratio

        # Key level params
        self.key_level_lookback = key_level_lookback
        self.key_level_proximity_pct = key_level_proximity_pct
        self.key_level_confidence_boost = key_level_confidence_boost

        # Min periods needed
        self.min_periods = max(
            slow_period + signal_period,
            atr_period,
            rsi_period,
            volume_lookback,
            divergence_lookback,
        ) + 10

    def generate(
        self,
        symbol: str,
        bars: pd.DataFrame,
        current_price: Optional[float] = None,
        higher_tf_bars: Optional[pd.DataFrame] = None,
        middle_tf_bars: Optional[pd.DataFrame] = None,
    ) -> Optional[Signal]:
        """
        Generate signal using 3-system approach.

        Priority order:
        1. System 3 (confirmation) - highest confidence
        2. System 1 (trend) - core strategy
        3. System 2 (reversal) - counter-trend

        Args:
            symbol: Asset symbol
            bars: OHLCV DataFrame (entry timeframe)
            current_price: Optional current price
            higher_tf_bars: Optional daily bars for System 3
            middle_tf_bars: Optional 4H bars for System 3

        Returns:
            Signal if conditions met, None otherwise
        """
        if not self.validate_bars(bars, self.min_periods):
            return None

        bars = self.normalize_bars(bars)
        close = bars["close"]
        high = bars["high"]
        low = bars["low"]
        open_price = bars["open"]
        volume = bars["volume"]

        # Calculate indicators
        macd_line, signal_line, histogram = macd(
            close, self.fast_period, self.slow_period, self.signal_period
        )
        atr_values = atr(high, low, close, self.atr_period)
        rsi_values = rsi(close, self.rsi_period)

        current = current_price if current_price else float(close.iloc[-1])
        latest_atr = float(atr_values.iloc[-1])
        curr_rsi = float(rsi_values.iloc[-1])

        # Get MACD values
        curr_macd = float(macd_line.iloc[-1])
        curr_signal = float(signal_line.iloc[-1])
        prev_macd = float(macd_line.iloc[-2])
        prev_signal = float(signal_line.iloc[-2])
        curr_hist = float(histogram.iloc[-1])

        # Volume check
        curr_volume = float(volume.iloc[-1])
        avg_volume = float(volume.rolling(self.volume_lookback).mean().iloc[-1])
        volume_ratio = curr_volume / avg_volume if avg_volume > 0 else 0.0
        volume_ok = volume_ratio >= self.volume_multiplier

        # RSI check (avoid extremes)
        rsi_ok = self.rsi_oversold < curr_rsi < self.rsi_overbought

        # Check each system in priority order
        system_signal: Optional[SystemSignal] = None

        # System 3: Multi-timeframe confirmation (highest priority)
        if higher_tf_bars is not None and middle_tf_bars is not None:
            system_signal = self._check_system3(
                bars, higher_tf_bars, middle_tf_bars,
                macd_line, signal_line, histogram
            )

        # System 1: Trend trading (core strategy)
        if system_signal is None:
            system_signal = self._check_system1(
                curr_macd, curr_signal, prev_macd, prev_signal, curr_hist
            )

        # System 2: Reversal (counter-trend)
        if system_signal is None:
            system_signal = self._check_system2(
                close, high, low, macd_line, signal_line, histogram
            )

        if system_signal is None:
            return None

        # Apply filters
        if not volume_ok:
            logger.info(
                f"[SIGNAL] {symbol}: System {system_signal.system.value} signal "
                f"rejected - vol({volume_ratio:.1f}x<{self.volume_multiplier}x)"
            )
            return None

        if not rsi_ok:
            logger.info(
                f"[SIGNAL] {symbol}: System {system_signal.system.value} signal "
                f"rejected - RSI({curr_rsi:.0f}) out of range"
            )
            return None

        # Check signal strength
        if system_signal.confidence < self.min_signal_strength:
            return None

        # Check candle quality (reject indecision candles)
        candle_ok, avg_body_ratio = self._check_candle_quality(
            open_price, high, low, close
        )
        if not candle_ok:
            logger.info(
                f"[SIGNAL] {symbol}: System {system_signal.system.value} signal "
                f"rejected - indecision candles (body_ratio={avg_body_ratio:.2f}<{self.min_body_ratio})"
            )
            return None

        # Check key level confluence (boost confidence if at S/R)
        near_key_level, key_level = self._check_key_level_confluence(
            current, system_signal.direction, high, low
        )
        if near_key_level:
            system_signal.confidence = min(1.0, system_signal.confidence + self.key_level_confidence_boost)
            logger.info(
                f"[SIGNAL] {symbol}: Key level confluence - "
                f"{'support' if system_signal.direction == SignalDirection.LONG else 'resistance'} "
                f"@ ${key_level:.2f} (boosted confidence to {system_signal.confidence:.2f})"
            )

        # Calculate stop and target
        stop_distance = latest_atr * self.atr_stop_multiplier

        if system_signal.direction == SignalDirection.LONG:
            stop_price = current - stop_distance
            target_price = current + (stop_distance * self.min_reward_ratio)
        else:
            stop_price = current + stop_distance
            target_price = current - (stop_distance * self.min_reward_ratio)

        logger.info(
            f"[SIGNAL] {symbol}: System {system_signal.system.value} "
            f"{system_signal.direction.value.upper()} - {system_signal.reason} "
            f"(confidence={system_signal.confidence:.2f}, RSI={curr_rsi:.0f}, Vol={volume_ratio:.1f}x)"
        )

        return Signal(
            symbol=symbol,
            direction=system_signal.direction,
            strength=system_signal.confidence,
            entry_price=current,
            stop_price=stop_price,
            target_price=target_price,
            timeframe=self._detect_timeframe(bars),
            strategy=self.name,
            timestamp=datetime.now(),
            metadata={
                "system": system_signal.system.value,
                "macd": round(curr_macd, 4),
                "signal": round(curr_signal, 4),
                "histogram": round(curr_hist, 4),
                "atr": round(latest_atr, 4),
                "rsi": round(curr_rsi, 2),
                "volume_ratio": round(volume_ratio, 2),
                "body_ratio": round(avg_body_ratio, 2),
                "near_key_level": near_key_level,
                "key_level": round(key_level, 2) if key_level else None,
                **system_signal.metadata,
            },
        )

    def _check_system1(
        self,
        curr_macd: float,
        curr_signal: float,
        prev_macd: float,
        prev_signal: float,
        curr_hist: float,
    ) -> Optional[SystemSignal]:
        """
        System 1: Trend Trading with Zero Line Filter.

        Rules:
        1. Zero Line Filter (MANDATORY):
           - MACD > 0 → ONLY LONG trades allowed
           - MACD < 0 → ONLY SHORT trades allowed

        2. Crossover Entry:
           - LONG: MACD crosses ABOVE signal (while MACD > 0)
           - SHORT: MACD crosses BELOW signal (while MACD < 0)

        3. Momentum Filter:
           - IGNORE crossovers if MACD between -buffer and +buffer
        """
        # Check for crossover
        bullish_crossover = prev_macd < prev_signal and curr_macd >= curr_signal
        bearish_crossover = prev_macd > prev_signal and curr_macd <= curr_signal

        if not (bullish_crossover or bearish_crossover):
            return None

        # Zero line filter - MANDATORY
        in_buffer_zone = -self.zero_line_buffer < curr_macd < self.zero_line_buffer

        if in_buffer_zone:
            # Weak signal in choppy zone - skip
            return None

        if bullish_crossover:
            # LONG only if MACD > 0
            if curr_macd <= 0:
                return None  # Don't trade against zero line

            # Confidence based on distance from zero
            confidence = min(1.0, 0.6 + abs(curr_macd) * 0.1)

            return SystemSignal(
                system=SystemType.TREND,
                direction=SignalDirection.LONG,
                confidence=confidence,
                reason=f"Bullish crossover above zero (MACD={curr_macd:.4f})",
                metadata={
                    "crossover": "bullish",
                    "zero_line_ok": True,
                },
            )

        if bearish_crossover:
            # SHORT only if MACD < 0
            if curr_macd >= 0:
                return None  # Don't trade against zero line

            confidence = min(1.0, 0.6 + abs(curr_macd) * 0.1)

            return SystemSignal(
                system=SystemType.TREND,
                direction=SignalDirection.SHORT,
                confidence=confidence,
                reason=f"Bearish crossover below zero (MACD={curr_macd:.4f})",
                metadata={
                    "crossover": "bearish",
                    "zero_line_ok": True,
                },
            )

        return None

    def _check_system2(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        macd_line: pd.Series,
        signal_line: pd.Series,
        histogram: pd.Series,
    ) -> Optional[SystemSignal]:
        """
        System 2: Reversal Strategy with Divergence Detection.

        Rules:
        1. Divergence Detection:
           - BEARISH: Price Higher Highs, MACD Lower Highs
           - BULLISH: Price Lower Lows, MACD Higher Lows

        2. Histogram Patterns:
           a) THE FLIP: First opposite-color bar after a series
           b) SHRINKING TOWER: Progressively smaller bars

        3. Entry: Divergence + histogram pattern = trade signal
        """
        # Detect divergence
        divergence = self._detect_divergence(close, high, low, macd_line)

        # Detect histogram pattern
        hist_pattern = self._detect_histogram_pattern(histogram)

        # Need both divergence AND histogram pattern
        if divergence == DivergenceType.NONE or hist_pattern == HistogramPattern.NONE:
            return None

        # Bullish divergence + bullish histogram pattern
        if divergence == DivergenceType.BULLISH:
            curr_hist = float(histogram.iloc[-1])
            if curr_hist > 0 or hist_pattern == HistogramPattern.FLIP:
                return SystemSignal(
                    system=SystemType.REVERSAL,
                    direction=SignalDirection.LONG,
                    confidence=0.7,  # Reversal signals slightly lower confidence
                    reason=f"Bullish divergence + {hist_pattern.value}",
                    metadata={
                        "divergence": divergence.value,
                        "histogram_pattern": hist_pattern.value,
                    },
                )

        # Bearish divergence + bearish histogram pattern
        if divergence == DivergenceType.BEARISH:
            curr_hist = float(histogram.iloc[-1])
            if curr_hist < 0 or hist_pattern == HistogramPattern.FLIP:
                return SystemSignal(
                    system=SystemType.REVERSAL,
                    direction=SignalDirection.SHORT,
                    confidence=0.7,
                    reason=f"Bearish divergence + {hist_pattern.value}",
                    metadata={
                        "divergence": divergence.value,
                        "histogram_pattern": hist_pattern.value,
                    },
                )

        return None

    def _check_system3(
        self,
        entry_bars: pd.DataFrame,
        daily_bars: pd.DataFrame,
        h4_bars: pd.DataFrame,
        entry_macd: pd.Series,
        entry_signal: pd.Series,
        entry_histogram: pd.Series,
    ) -> Optional[SystemSignal]:
        """
        System 3: Multi-Timeframe Confirmation.

        Rules:
        1. Daily MACD determines overall BIAS (> 0 = bullish, < 0 = bearish)
        2. 4H MACD must CONFIRM direction
        3. Entry timeframe provides entry TIMING

        FOR LONG:
        - Daily MACD > 0 (bullish bias)
        - 4H bullish crossover or MACD > signal
        - Entry timeframe bullish crossover

        FOR SHORT:
        - Daily MACD < 0 (bearish bias)
        - 4H bearish crossover or MACD < signal
        - Entry timeframe bearish crossover
        """
        if not self.validate_bars(daily_bars, 30):
            return None
        if not self.validate_bars(h4_bars, 30):
            return None

        daily_bars = self.normalize_bars(daily_bars)
        h4_bars = self.normalize_bars(h4_bars)

        # Calculate MACD for higher timeframes
        daily_macd, daily_sig, _ = macd(
            daily_bars["close"], self.fast_period, self.slow_period, self.signal_period
        )
        h4_macd, h4_sig, _ = macd(
            h4_bars["close"], self.fast_period, self.slow_period, self.signal_period
        )

        # Get current values
        daily_macd_val = float(daily_macd.iloc[-1])
        h4_macd_val = float(h4_macd.iloc[-1])
        h4_sig_val = float(h4_sig.iloc[-1])

        entry_macd_val = float(entry_macd.iloc[-1])
        entry_sig_val = float(entry_signal.iloc[-1])
        prev_entry_macd = float(entry_macd.iloc[-2])
        prev_entry_sig = float(entry_signal.iloc[-2])

        # Check entry timeframe crossover
        bullish_entry = prev_entry_macd < prev_entry_sig and entry_macd_val >= entry_sig_val
        bearish_entry = prev_entry_macd > prev_entry_sig and entry_macd_val <= entry_sig_val

        # LONG setup
        if bullish_entry:
            # Daily bias must be bullish
            if daily_macd_val <= 0:
                return None

            # 4H must confirm
            if h4_macd_val <= h4_sig_val:
                return None

            return SystemSignal(
                system=SystemType.CONFIRMATION,
                direction=SignalDirection.LONG,
                confidence=0.85,  # Highest confidence - all timeframes aligned
                reason="Triple timeframe bullish alignment",
                metadata={
                    "daily_macd": round(daily_macd_val, 4),
                    "h4_macd": round(h4_macd_val, 4),
                    "timeframe_alignment": True,
                },
            )

        # SHORT setup
        if bearish_entry:
            # Daily bias must be bearish
            if daily_macd_val >= 0:
                return None

            # 4H must confirm
            if h4_macd_val >= h4_sig_val:
                return None

            return SystemSignal(
                system=SystemType.CONFIRMATION,
                direction=SignalDirection.SHORT,
                confidence=0.85,
                reason="Triple timeframe bearish alignment",
                metadata={
                    "daily_macd": round(daily_macd_val, 4),
                    "h4_macd": round(h4_macd_val, 4),
                    "timeframe_alignment": True,
                },
            )

        return None

    def _detect_divergence(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        macd_line: pd.Series,
    ) -> DivergenceType:
        """
        Detect price/MACD divergence.

        Bullish: Price Lower Lows, MACD Higher Lows
        Bearish: Price Higher Highs, MACD Lower Highs
        """
        lookback = self.divergence_lookback

        if len(close) < lookback:
            return DivergenceType.NONE

        # Get recent swing points
        recent_highs = high.iloc[-lookback:]
        recent_lows = low.iloc[-lookback:]
        recent_macd = macd_line.iloc[-lookback:]

        # Find local peaks and troughs (simplified)
        price_highs_idx = self._find_peaks(recent_highs)
        price_lows_idx = self._find_troughs(recent_lows)
        macd_highs_idx = self._find_peaks(recent_macd)
        macd_lows_idx = self._find_troughs(recent_macd)

        # Need at least 2 swing points to compare
        if len(price_highs_idx) >= 2 and len(macd_highs_idx) >= 2:
            # Check bearish divergence
            last_price_high = float(recent_highs.iloc[price_highs_idx[-1]])
            prev_price_high = float(recent_highs.iloc[price_highs_idx[-2]])
            last_macd_high = float(recent_macd.iloc[macd_highs_idx[-1]])
            prev_macd_high = float(recent_macd.iloc[macd_highs_idx[-2]])

            # Price making higher highs but MACD making lower highs
            if last_price_high > prev_price_high and last_macd_high < prev_macd_high:
                return DivergenceType.BEARISH

        if len(price_lows_idx) >= 2 and len(macd_lows_idx) >= 2:
            # Check bullish divergence
            last_price_low = float(recent_lows.iloc[price_lows_idx[-1]])
            prev_price_low = float(recent_lows.iloc[price_lows_idx[-2]])
            last_macd_low = float(recent_macd.iloc[macd_lows_idx[-1]])
            prev_macd_low = float(recent_macd.iloc[macd_lows_idx[-2]])

            # Price making lower lows but MACD making higher lows
            if last_price_low < prev_price_low and last_macd_low > prev_macd_low:
                return DivergenceType.BULLISH

        return DivergenceType.NONE

    def _detect_histogram_pattern(self, histogram: pd.Series) -> HistogramPattern:
        """
        Detect histogram reversal patterns.

        THE FLIP: First opposite-color bar after a series of same-color bars
        SHRINKING TOWER: Progressively smaller bars (exhaustion)
        """
        if len(histogram) < self.histogram_flip_min_bars + 1:
            return HistogramPattern.NONE

        recent = histogram.iloc[-(self.histogram_flip_min_bars + 1):]
        curr_hist = float(recent.iloc[-1])
        prev_bars = recent.iloc[:-1]

        # Check for FLIP pattern
        if curr_hist > 0:  # Current bar is green
            # Check if previous N bars were all red
            if all(float(h) < 0 for h in prev_bars):
                return HistogramPattern.FLIP
        elif curr_hist < 0:  # Current bar is red
            # Check if previous N bars were all green
            if all(float(h) > 0 for h in prev_bars):
                return HistogramPattern.FLIP

        # Check for SHRINKING pattern
        abs_vals = [abs(float(h)) for h in recent]
        if len(abs_vals) >= 3:
            # Check if histogram is progressively shrinking
            is_shrinking = all(
                abs_vals[i] > abs_vals[i + 1]
                for i in range(len(abs_vals) - 3, len(abs_vals) - 1)
            )
            if is_shrinking:
                return HistogramPattern.SHRINKING

        return HistogramPattern.NONE

    def _find_peaks(self, series: pd.Series) -> list[int]:
        """Find local maxima indices."""
        peaks = []
        values = series.values
        for i in range(1, len(values) - 1):
            if values[i] > values[i - 1] and values[i] > values[i + 1]:
                peaks.append(i)
        return peaks

    def _find_troughs(self, series: pd.Series) -> list[int]:
        """Find local minima indices."""
        troughs = []
        values = series.values
        for i in range(1, len(values) - 1):
            if values[i] < values[i - 1] and values[i] < values[i + 1]:
                troughs.append(i)
        return troughs

    def should_exit(
        self,
        symbol: str,
        bars: pd.DataFrame,
        entry_price: float,
        direction: SignalDirection,
        current_price: Optional[float] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Check exit conditions.

        Exit when:
        1. Opposite crossover on entry timeframe
        2. Zero line violation (MACD crosses zero against position)
        3. Stop-loss/take-profit handled by position manager

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

        # Calculate MACD
        macd_line, signal_line, _ = macd(
            close, self.fast_period, self.slow_period, self.signal_period
        )

        curr_macd = float(macd_line.iloc[-1])
        prev_macd = float(macd_line.iloc[-2])
        curr_signal = float(signal_line.iloc[-1])
        prev_signal = float(signal_line.iloc[-2])

        if direction == SignalDirection.LONG:
            # Exit 1: Bearish crossover
            if prev_macd > prev_signal and curr_macd <= curr_signal:
                return True, f"Bearish crossover (MACD={curr_macd:.4f})"

            # Exit 2: MACD crossed below zero
            if prev_macd > 0 and curr_macd <= 0:
                return True, f"Zero line violation - MACD crossed below zero ({curr_macd:.4f})"

        elif direction == SignalDirection.SHORT:
            # Exit 1: Bullish crossover
            if prev_macd < prev_signal and curr_macd >= curr_signal:
                return True, f"Bullish crossover (MACD={curr_macd:.4f})"

            # Exit 2: MACD crossed above zero
            if prev_macd < 0 and curr_macd >= 0:
                return True, f"Zero line violation - MACD crossed above zero ({curr_macd:.4f})"

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

    def _check_candle_quality(
        self,
        open_price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> tuple[bool, float]:
        """
        Check if recent candles show conviction (not indecision).

        Rejects entries on doji, spinning tops, or other indecision patterns.
        Measures body size relative to total candle range.

        Args:
            open_price: Open prices
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            Tuple of (quality_ok, avg_body_ratio)
        """
        lookback = self.candle_quality_lookback
        if len(close) < lookback:
            return True, 1.0  # Not enough data, pass by default

        body_ratios = []
        for i in range(-lookback, 0):
            candle_range = float(high.iloc[i] - low.iloc[i])
            if candle_range == 0:
                body_ratios.append(0.0)
                continue
            body_size = abs(float(close.iloc[i] - open_price.iloc[i]))
            body_ratio = body_size / candle_range
            body_ratios.append(body_ratio)

        avg_body_ratio = sum(body_ratios) / len(body_ratios) if body_ratios else 0.0

        # Require average body ratio above threshold
        quality_ok = avg_body_ratio >= self.min_body_ratio

        return quality_ok, avg_body_ratio

    def _check_key_level_confluence(
        self,
        current_price: float,
        direction: SignalDirection,
        high: pd.Series,
        low: pd.Series,
    ) -> tuple[bool, Optional[float]]:
        """
        Check if price is near a key support/resistance level.

        For LONG: price near support is good (buying at support)
        For SHORT: price near resistance is good (selling at resistance)

        Args:
            current_price: Current price
            direction: Signal direction
            high: High prices
            low: Low prices

        Returns:
            Tuple of (is_near_key_level, nearest_level)
        """
        lookback = self.key_level_lookback
        if len(high) < lookback:
            return False, None

        recent_high = high.iloc[-lookback:]
        recent_low = low.iloc[-lookback:]

        # Find swing highs (resistance) using existing method
        peak_indices = self._find_peaks(recent_high)
        resistance_levels = [float(recent_high.iloc[i]) for i in peak_indices]

        # Find swing lows (support) using existing method
        trough_indices = self._find_troughs(recent_low)
        support_levels = [float(recent_low.iloc[i]) for i in trough_indices]

        # Determine which levels to check based on direction
        if direction == SignalDirection.LONG:
            # For longs, check proximity to support
            levels_to_check = support_levels
        else:
            # For shorts, check proximity to resistance
            levels_to_check = resistance_levels

        if not levels_to_check:
            return False, None

        # Find nearest level
        distances = [(abs(current_price - level), level) for level in levels_to_check]
        min_distance, nearest_level = min(distances, key=lambda x: x[0])

        # Check if within proximity threshold
        proximity_pct = min_distance / current_price if current_price > 0 else 1.0
        is_near = proximity_pct <= self.key_level_proximity_pct

        return is_near, nearest_level if is_near else None
