"""
Momentum Surge Strategy for day trading.

Enters on the initial surge — no pullback required. For stocks that are
actively breaking out with strong momentum, volume, and MACD confirmation.
Highly selective — fires rarely on the strongest setups only.

Entry conditions (ALL must be true):
1. Price near HOD: within 3% of the 20-bar high
2. MACD positive: MACD line > 0 AND histogram > 0
3. RSI sweet spot: between 55-80 (momentum without exhaustion)
4. Volume surge: current bar > 3x 20-bar volume SMA
5. Price momentum: 10-bar ROC > 3%
6. Bar closing strength: close in upper 60% of bar range

Stop: entry - (ATR × 2.0)
Exit: 2 consecutive MACD histogram negative bars, RSI < 40, or price below 10-bar low
"""

from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from src.bot.signals.base import Signal, SignalDirection, SignalGenerator
from src.data.indicators import atr, macd, rsi, volume_sma


class MomentumSurgeStrategy(SignalGenerator):
    """
    Momentum surge entry strategy on 5-min bars.

    Enters when a stock is actively surging with strong momentum indicators.
    Does NOT wait for a pullback — catches the initial move.
    """

    def __init__(
        self,
        macd_fast: int = 8,
        macd_slow: int = 21,
        macd_signal: int = 5,
        rsi_period: int = 14,
        atr_period: int = 14,
        atr_stop_multiplier: float = 2.0,
        volume_period: int = 20,
        volume_multiplier: float = 3.0,
        roc_period: int = 10,
        roc_min: float = 0.03,
        rsi_min: float = 55.0,
        rsi_max: float = 80.0,
        hod_proximity: float = 0.03,
        risk_reward_target: float = 10.0,
        min_signal_strength: float = 0.5,
    ):
        super().__init__(name="momentum_surge")
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.atr_stop_multiplier = atr_stop_multiplier
        self.volume_period = volume_period
        self.volume_multiplier = volume_multiplier
        self.roc_period = roc_period
        self.roc_min = roc_min
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max
        self.hod_proximity = hod_proximity
        self.risk_reward_target = risk_reward_target
        self.min_signal_strength = min_signal_strength

        self.min_periods = max(macd_slow, volume_period, atr_period, roc_period) + 10

    def generate(
        self,
        symbol: str,
        bars: pd.DataFrame,
        current_price: Optional[float] = None,
        has_catalyst: bool = False,
    ) -> Optional[Signal]:
        """Generate a momentum surge entry signal."""
        if not self.validate_bars(bars, self.min_periods):
            return None

        bars = self.normalize_bars(bars)

        close = bars["close"]
        high = bars["high"]
        low = bars["low"]
        volume = bars["volume"]

        current = current_price if current_price else float(close.iloc[-1])

        # ── Indicator calculations ──────────────────────────────────────
        macd_line, signal_line, histogram = macd(
            close, self.macd_fast, self.macd_slow, self.macd_signal
        )
        rsi_values = rsi(close, self.rsi_period)
        atr_values = atr(high, low, close, self.atr_period)
        avg_volume = volume_sma(volume, self.volume_period)

        # Rate of change (momentum)
        roc = close.pct_change(self.roc_period)

        # Current values
        cur_macd = float(macd_line.iloc[-1])
        cur_hist = float(histogram.iloc[-1])
        cur_rsi = float(rsi_values.iloc[-1])
        cur_atr = float(atr_values.iloc[-1])
        cur_volume = float(volume.iloc[-1])
        cur_avg_volume = float(avg_volume.iloc[-1])
        cur_roc = float(roc.iloc[-1])

        # 20-bar high (high of day proxy on 5-min bars)
        bar_high_20 = float(high.iloc[-20:].max())

        # ── Entry conditions ────────────────────────────────────────────

        # 1. Price near HOD (within hod_proximity of 20-bar high)
        if bar_high_20 > 0 and (bar_high_20 - current) / bar_high_20 > self.hod_proximity:
            logger.debug(
                f"[SURGE] {symbol}: price ${current:.2f} too far from "
                f"20-bar high ${bar_high_20:.2f} "
                f"({(bar_high_20 - current) / bar_high_20:.1%} > {self.hod_proximity:.0%})"
            )
            return None

        # 2. MACD positive with positive histogram
        if cur_macd <= 0 or cur_hist <= 0:
            logger.debug(
                f"[SURGE] {symbol}: MACD not positive "
                f"(macd={cur_macd:.4f}, hist={cur_hist:.4f})"
            )
            return None

        # 3. RSI in sweet spot
        if cur_rsi < self.rsi_min or cur_rsi > self.rsi_max:
            logger.debug(
                f"[SURGE] {symbol}: RSI {cur_rsi:.1f} outside "
                f"{self.rsi_min}-{self.rsi_max} range"
            )
            return None

        # 4. Volume surge
        volume_ratio = cur_volume / cur_avg_volume if cur_avg_volume > 0 else 0
        if volume_ratio < self.volume_multiplier:
            logger.debug(
                f"[SURGE] {symbol}: volume ratio {volume_ratio:.1f}x "
                f"< {self.volume_multiplier}x required"
            )
            return None

        # 5. Price momentum (ROC)
        if cur_roc < self.roc_min:
            logger.debug(
                f"[SURGE] {symbol}: ROC {cur_roc:.1%} < {self.roc_min:.0%} min"
            )
            return None

        # 6. Bar closing strength — close in upper 60% of bar range
        cur_high = float(high.iloc[-1])
        cur_low = float(low.iloc[-1])
        cur_close = float(close.iloc[-1])
        bar_range = cur_high - cur_low
        if bar_range > 0:
            close_position = (cur_close - cur_low) / bar_range
            if close_position < 0.40:
                logger.debug(
                    f"[SURGE] {symbol}: bar close weak "
                    f"({close_position:.0%} of range, need >40%)"
                )
                return None

        # ── All conditions met — calculate signal ───────────────────────
        logger.info(
            f"[SURGE] {symbol}: ENTRY signal @ ${current:.2f} | "
            f"MACD={cur_macd:.4f} hist={cur_hist:.4f} | "
            f"RSI={cur_rsi:.1f} | vol={volume_ratio:.1f}x | ROC={cur_roc:.1%}"
        )

        # Stop and target
        stop_price = current - (cur_atr * self.atr_stop_multiplier)
        risk = current - stop_price
        target_price = current + (risk * self.risk_reward_target)

        # Signal strength
        strength = self._calculate_strength(
            cur_rsi, volume_ratio, histogram, has_catalyst
        )

        if strength < self.min_signal_strength:
            return None

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
                "system": "momentum_surge",
                "macd": round(cur_macd, 4),
                "macd_histogram": round(cur_hist, 4),
                "rsi": round(cur_rsi, 1),
                "volume_ratio": round(volume_ratio, 1),
                "roc_10": round(cur_roc * 100, 2),
                "atr": round(cur_atr, 4),
                "bar_high_20": round(bar_high_20, 2),
            },
        )

    def _calculate_strength(
        self,
        cur_rsi: float,
        volume_ratio: float,
        histogram: pd.Series,
        has_catalyst: bool,
    ) -> float:
        """Calculate signal strength (0.0 - 1.0)."""
        strength = 0.50  # Base

        # RSI in ideal momentum zone (60-75)
        if 60 <= cur_rsi <= 75:
            strength += 0.10

        # Strong volume (>5x average — 3x is already the minimum entry)
        if volume_ratio > 5.0:
            strength += 0.10

        # MACD histogram accelerating (current > previous)
        if len(histogram) >= 2:
            cur_h = float(histogram.iloc[-1])
            prev_h = float(histogram.iloc[-2])
            if cur_h > prev_h > 0:
                strength += 0.10

        # News catalyst
        if has_catalyst:
            strength += 0.05

        return min(1.0, strength)

    def should_exit(
        self,
        symbol: str,
        bars: pd.DataFrame,
        entry_price: float,
        direction: SignalDirection,
        current_price: Optional[float] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a surge position should exit.

        Exit when:
        - MACD histogram negative 2 consecutive bars (momentum fading)
        - RSI drops below 40 (momentum lost)
        - Price below 10-bar low (trend broken)
        """
        if not self.validate_bars(bars, self.min_periods):
            return False, None

        bars = self.normalize_bars(bars)

        close = bars["close"]
        high = bars["high"]
        low = bars["low"]

        current = current_price if current_price else float(close.iloc[-1])

        # MACD histogram — require 2 consecutive negative bars
        # (single negative bar is normal oscillation during a surge)
        _, _, histogram = macd(
            close, self.macd_fast, self.macd_slow, self.macd_signal
        )
        if len(histogram) >= 2:
            cur_hist = float(histogram.iloc[-1])
            prev_hist = float(histogram.iloc[-2])
            if cur_hist < 0 and prev_hist < 0:
                return True, f"MACD histogram negative 2 bars ({prev_hist:.4f}, {cur_hist:.4f})"

        # RSI collapse
        rsi_values = rsi(close, self.rsi_period)
        cur_rsi = float(rsi_values.iloc[-1])

        if cur_rsi < 40:
            return True, f"RSI collapsed to {cur_rsi:.1f}"

        # Price below 10-bar low (Donchian exit)
        ten_bar_low = float(low.iloc[-10:].min())
        if current < ten_bar_low:
            return True, f"Price ${current:.2f} below 10-bar low ${ten_bar_low:.2f}"

        return False, None

    def _detect_timeframe(self, bars: pd.DataFrame) -> str:
        """Detect timeframe from bar index."""
        if len(bars) < 2:
            return "unknown"
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
            elif minutes <= 1440:
                return "1Day"
            else:
                return "1Week"
        except Exception:
            return "5Min"
