"""
Momentum Pullback Strategy for day trading.

Implements Ross Cameron's approach:
- Trade low-float, high-volume momentum stocks ($1-$10, prefer $2+)
- Enter on the first pullback after an initial surge
- MACD must be positive (above zero line) on 5-min chart
- Volume confirmation: green candle volume > pullback avg volume
- Target: 2x risk/reward, tight trailing stop

Entry conditions (all must pass):
1. MACD > 0 on 5-min chart (above zero line = trend is bullish)
2. Pullback detected: surge → 2-15 lower/red candles → first green new high
3. Pullback retracement < 65% of surge height
4. Entry candle volume >= average of pullback candles
5. Stop below pullback low (or 1.5× ATR, whichever is tighter)
6. Target = stop distance × risk_reward_target

Exit conditions:
1. MACD crosses below zero line (momentum died)
2. 3 consecutive declining histogram bars (fading)
3. Stop/target/trailing handled by position monitor
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from src.bot.signals.base import Signal, SignalDirection, SignalGenerator
from src.data.indicators import macd as calc_macd, atr as calc_atr

logger = logging.getLogger(__name__)


class MomentumPullbackStrategy(SignalGenerator):
    """
    Ross Cameron-style momentum pullback strategy on 5-min bars.

    Looks for stocks that have surged (already up big on the day),
    waits for a pullback, and enters on the first candle that makes
    a new high after the pullback — but ONLY when MACD is positive.
    """

    def __init__(
        self,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_period: int = 14,
        atr_stop_multiplier: float = 1.5,
        pullback_min_candles: int = 2,
        pullback_max_candles: int = 8,
        pullback_max_retracement: float = 0.50,
        volume_entry_multiplier: float = 1.0,
        risk_reward_target: float = 2.0,
        min_signal_strength: float = 0.5,
    ):
        """
        Initialize momentum pullback strategy.

        Args:
            macd_fast: MACD fast EMA period (12 standard)
            macd_slow: MACD slow EMA period (26 standard)
            macd_signal: MACD signal line period (9 standard)
            atr_period: ATR period for stop sizing
            atr_stop_multiplier: ATR multiplier for stop (1.5 = tighter for day trading)
            pullback_min_candles: Min candles in pullback before entry (2)
            pullback_max_candles: Max candles in pullback (8, beyond = momentum lost)
            pullback_max_retracement: Max pullback depth as % of surge (0.50 = 50%)
            volume_entry_multiplier: Entry candle vol must be > pullback avg × this (1.2)
            risk_reward_target: Take-profit = stop_distance × this (2.0)
            min_signal_strength: Minimum strength to generate signal (0.5)
        """
        super().__init__(name="momentum_pullback")
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.pullback_min_candles = pullback_min_candles
        self.pullback_max_candles = pullback_max_candles
        self.pullback_max_retracement = pullback_max_retracement
        self.volume_entry_multiplier = volume_entry_multiplier
        self.risk_reward_target = risk_reward_target
        self.min_signal_strength = min_signal_strength

        # Minimum bars needed for all indicators
        self.min_bars = max(macd_slow + macd_signal + 10, atr_period + 10, 40)

    def generate(
        self,
        symbol: str,
        bars: pd.DataFrame,
        current_price: Optional[float] = None,
        has_catalyst: bool = False,
    ) -> Optional[Signal]:
        """
        Generate a momentum pullback entry signal.

        Args:
            symbol: Stock symbol
            bars: 5-min OHLCV DataFrame
            current_price: Current price (uses last close if not provided)
            has_catalyst: Whether the stock has a news catalyst (boosts signal strength)

        Returns:
            Signal if pullback entry conditions are met, None otherwise
        """
        # Validate input
        if not self.validate_bars(bars, self.min_bars):
            return None

        bars = self.normalize_bars(bars)
        price = current_price or float(bars["close"].iloc[-1])

        # ── Step 1: Calculate MACD ──────────────────────────────────────
        macd_line, signal_line, histogram = calc_macd(
            bars["close"],
            fast_period=self.macd_fast,
            slow_period=self.macd_slow,
            signal_period=self.macd_signal,
        )

        current_macd = float(macd_line.iloc[-1])
        current_signal = float(signal_line.iloc[-1])
        current_histogram = float(histogram.iloc[-1])

        # ── Step 2: MACD must be ABOVE ZERO LINE ───────────────────────
        # This is the PRIMARY filter — trend must be bullish
        if current_macd <= 0:
            logger.debug(f"[{symbol}] MACD below zero ({current_macd:.4f}), skip")
            return None

        # Note: We do NOT require MACD > signal here. During a pullback,
        # MACD naturally dips below the signal line — that's what a pullback IS.
        # We only need MACD > 0 to confirm the overall trend is bullish.

        # ── Step 3: Detect pullback pattern ────────────────────────────
        pullback = self._detect_pullback(bars)
        if pullback is None:
            return None

        surge_high = pullback["surge_high"]
        pullback_low = pullback["pullback_low"]
        pullback_candles = pullback["pullback_candle_count"]
        surge_start = pullback["surge_start_price"]

        logger.info(
            f"[{symbol}] Pullback detected: surge ${surge_start:.2f}→${surge_high:.2f}, "
            f"pullback to ${pullback_low:.2f} ({pullback_candles} candles), "
            f"new high forming"
        )

        # ── Step 4: Volume confirmation ────────────────────────────────
        # Entry candle should have more volume than the average pullback candle
        entry_volume = float(bars["volume"].iloc[-1])
        pullback_start_idx = -(pullback_candles + 1)
        pullback_end_idx = -1
        pullback_volumes = bars["volume"].iloc[pullback_start_idx:pullback_end_idx]

        if len(pullback_volumes) > 0:
            avg_pullback_volume = float(pullback_volumes.mean())
        else:
            avg_pullback_volume = entry_volume

        volume_ratio = entry_volume / avg_pullback_volume if avg_pullback_volume > 0 else 0

        if volume_ratio < self.volume_entry_multiplier:
            logger.debug(
                f"[{symbol}] Entry volume too low: "
                f"{volume_ratio:.1f}x vs required {self.volume_entry_multiplier}x"
            )
            return None

        # ── Step 5: Calculate stop and target ──────────────────────────
        atr_value = float(calc_atr(
            bars["high"], bars["low"], bars["close"],
            period=self.atr_period,
        ).iloc[-1])

        # Stop: below pullback low or ATR-based, whichever is tighter
        atr_stop = price - (atr_value * self.atr_stop_multiplier)
        pullback_stop = pullback_low - (atr_value * 0.25)  # Small buffer below pullback low

        # Use the tighter (higher) stop for day trading
        stop_price = max(atr_stop, pullback_stop)

        # Safety: stop must be below current price
        if stop_price >= price:
            stop_price = price * 0.97  # Fallback: 3% stop

        stop_distance = price - stop_price

        # Target: risk_reward_target × stop distance above entry
        target_price = price + (stop_distance * self.risk_reward_target)

        # ── Step 6: Calculate signal strength ──────────────────────────
        strength = self._calculate_strength(
            macd_value=current_macd,
            histogram=current_histogram,
            volume_ratio=volume_ratio,
            pullback_depth_pct=pullback["retracement_pct"],
            atr_value=atr_value,
            price=price,
            has_catalyst=has_catalyst,
        )

        if strength < self.min_signal_strength:
            logger.debug(f"[{symbol}] Signal strength too low: {strength:.2f}")
            return None

        # ── Step 7: Generate signal ────────────────────────────────────
        signal = Signal(
            symbol=symbol,
            direction=SignalDirection.LONG,
            strength=strength,
            entry_price=price,
            stop_price=round(stop_price, 2),
            target_price=round(target_price, 2),
            strategy=self.name,
            timeframe="5Min",
            metadata={
                "system": "momentum_pullback",
                "macd": round(current_macd, 4),
                "signal": round(current_signal, 4),
                "histogram": round(current_histogram, 4),
                "atr": round(atr_value, 4),
                "surge_high": round(surge_high, 2),
                "pullback_low": round(pullback_low, 2),
                "pullback_candles": pullback_candles,
                "retracement_pct": round(pullback["retracement_pct"] * 100, 1),
                "volume_ratio": round(volume_ratio, 1),
                "stop_distance": round(stop_distance, 2),
                "risk_reward": round(self.risk_reward_target, 1),
            },
        )

        logger.info(
            f"[{symbol}] SIGNAL: LONG @ ${price:.2f}, "
            f"stop=${stop_price:.2f}, target=${target_price:.2f}, "
            f"strength={strength:.2f}, R:R={self.risk_reward_target:.1f}"
        )

        return signal

    def should_exit(
        self,
        symbol: str,
        bars: pd.DataFrame,
        entry_price: float,
        direction: SignalDirection,
        current_price: Optional[float] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if momentum has died and position should exit.

        Exit when:
        1. MACD crosses below zero line (momentum completely dead)
        2. 3 consecutive declining histogram bars (momentum fading fast)

        Note: Stop-loss, take-profit, and trailing stop are handled by
        the PositionMonitor separately.

        Args:
            symbol: Stock symbol
            bars: Recent 5-min OHLCV DataFrame
            entry_price: Position entry price
            direction: Position direction (always LONG for this strategy)
            current_price: Current price

        Returns:
            (should_exit, reason) tuple
        """
        if not self.validate_bars(bars, self.min_bars):
            return False, None

        bars = self.normalize_bars(bars)

        # Calculate MACD
        macd_line, signal_line, histogram = calc_macd(
            bars["close"],
            fast_period=self.macd_fast,
            slow_period=self.macd_slow,
            signal_period=self.macd_signal,
        )

        current_macd = float(macd_line.iloc[-1])
        current_histogram = float(histogram.iloc[-1])

        # Exit 1: MACD crosses below zero (momentum is dead)
        if current_macd < 0:
            prev_macd = float(macd_line.iloc[-2])
            if prev_macd >= 0:  # Just crossed below
                return True, "MACD crossed below zero line (momentum dead)"

        # Exit 2: 3 consecutive declining histogram bars (momentum fading)
        if len(histogram) >= 4:
            h_vals = [float(histogram.iloc[i]) for i in range(-4, 0)]
            declining = all(h_vals[i] > h_vals[i + 1] for i in range(3))
            if declining and current_histogram < 0:
                return True, "3 declining histogram bars with negative histogram"

        return False, None

    def _detect_pullback(self, bars: pd.DataFrame) -> Optional[dict]:
        """
        Detect a pullback pattern in the price action.

        A valid pullback:
        1. Price surged to a local high (the "surge peak")
        2. Price pulled back (2-15 candles of lower highs or red candles)
        3. Current candle is making a new high above recent pullback highs
        4. Pullback didn't retrace more than 65% of the surge

        Args:
            bars: Normalized OHLCV DataFrame

        Returns:
            Dict with pullback details, or None if no valid pullback found
        """
        lookback = 40  # Look back 40 candles (~3.3 hours on 5-min bars)
        if len(bars) < lookback:
            lookback = len(bars)

        recent = bars.iloc[-lookback:]
        highs = recent["high"].values
        lows = recent["low"].values
        closes = recent["close"].values
        opens = recent["open"].values

        # Find the local high (surge peak) in the lookback window
        # Skip the last candle (potential entry candle)
        peak_idx_relative = 0
        peak_high = 0.0

        for i in range(len(highs) - 2, max(0, len(highs) - lookback), -1):
            if highs[i] > peak_high:
                peak_high = highs[i]
                peak_idx_relative = i

        if peak_high == 0:
            return None

        # The current candle
        current_high = float(highs[-1])
        current_close = float(closes[-1])
        current_open = float(opens[-1])

        # Count pullback candles between peak and current
        # A pullback candle: high < previous high (lower highs) OR red candle
        pullback_candles = 0
        pullback_low = float("inf")
        pullback_highs = []

        for i in range(peak_idx_relative + 1, len(highs) - 1):
            candle_high = float(highs[i])
            candle_low = float(lows[i])
            candle_close = float(closes[i])
            candle_open = float(opens[i])

            # Is this a pullback candle? (lower high or red)
            is_lower_high = candle_high < peak_high
            is_red = candle_close < candle_open

            if is_lower_high or is_red:
                pullback_candles += 1
                pullback_low = min(pullback_low, candle_low)
                pullback_highs.append(candle_high)

        # Validate pullback length
        if pullback_candles < self.pullback_min_candles:
            logger.debug(f"Pullback too short: {pullback_candles} < {self.pullback_min_candles}")
            return None

        if pullback_candles > self.pullback_max_candles:
            logger.debug(f"Pullback too long: {pullback_candles} > {self.pullback_max_candles}")
            return None

        # Current candle must make a new high vs recent pullback candle highs
        # Use last 3 pullback highs instead of ALL — a single spike candle in
        # the pullback shouldn't disqualify the entire pattern
        recent_pullback_highs = pullback_highs[-3:] if len(pullback_highs) > 3 else pullback_highs
        if recent_pullback_highs and current_high <= max(recent_pullback_highs):
            logger.debug("Current candle not making new high above recent pullback")
            return None

        # Current candle should be green (bullish)
        if current_close <= current_open:
            logger.debug("Current candle is red, need green for entry")
            return None

        # Calculate surge start (the low before the peak)
        surge_start = float("inf")
        for i in range(max(0, peak_idx_relative - 10), peak_idx_relative):
            surge_start = min(surge_start, float(lows[i]))

        if surge_start == float("inf"):
            surge_start = float(lows[max(0, peak_idx_relative - 1)])

        # Calculate retracement
        surge_height = peak_high - surge_start
        if surge_height <= 0:
            return None

        pullback_depth = peak_high - pullback_low
        retracement_pct = pullback_depth / surge_height

        if retracement_pct > self.pullback_max_retracement:
            logger.debug(
                f"Pullback too deep: {retracement_pct:.1%} > "
                f"{self.pullback_max_retracement:.1%}"
            )
            return None

        return {
            "surge_high": peak_high,
            "surge_start_price": surge_start,
            "surge_height": surge_height,
            "pullback_low": pullback_low,
            "pullback_candle_count": pullback_candles,
            "retracement_pct": retracement_pct,
            "is_first_new_high": True,
        }

    def _calculate_strength(
        self,
        macd_value: float,
        histogram: float,
        volume_ratio: float,
        pullback_depth_pct: float,
        atr_value: float,
        price: float,
        has_catalyst: bool = False,
    ) -> float:
        """
        Calculate signal strength (0.0 to 1.0).

        Higher strength when:
        - MACD is strongly positive
        - Histogram is increasing
        - Volume is high on entry candle
        - Pullback was shallow (momentum still strong)
        - News catalyst present (confirmation, not requirement)

        Args:
            macd_value: Current MACD line value
            histogram: Current histogram value
            volume_ratio: Entry candle volume / pullback avg volume
            pullback_depth_pct: Pullback retracement as decimal (0.0-1.0)
            atr_value: Current ATR value
            price: Current price
            has_catalyst: Whether the stock has a news catalyst

        Returns:
            Signal strength between 0.0 and 1.0
        """
        strength = 0.5  # Base strength

        # MACD strength: stronger MACD = more confidence
        # Normalize MACD relative to ATR for comparability across price ranges
        if atr_value > 0:
            macd_strength = min(abs(macd_value) / atr_value, 1.0) * 0.15
        else:
            macd_strength = 0.0
        strength += macd_strength

        # Histogram momentum: positive and increasing = good
        if histogram > 0:
            strength += 0.05

        # Volume: higher relative volume on entry = more conviction
        if volume_ratio >= 3.0:
            strength += 0.15
        elif volume_ratio >= 2.0:
            strength += 0.10
        elif volume_ratio >= 1.5:
            strength += 0.05

        # Shallow pullback = momentum still strong
        if pullback_depth_pct < 0.25:
            strength += 0.10  # Very shallow pullback
        elif pullback_depth_pct < 0.35:
            strength += 0.05  # Moderate pullback

        # News catalyst: boosts confidence but not required
        if has_catalyst:
            strength += 0.10

        # Clamp to [0, 1]
        return max(0.0, min(1.0, strength))
