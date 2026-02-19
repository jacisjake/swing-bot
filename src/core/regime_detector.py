"""
HMM Regime Detector for live trading.

Trains a Gaussian HMM on a market proxy (SPY) to classify the current regime
as bullish, bearish, or neutral. Used as a market-level gate to prevent
entries during unfavorable regimes.

Reuses core functions from scripts/backtest_hmm.py.
"""

import threading
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from loguru import logger

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class RegimeDetector:
    """
    Live HMM regime detector.

    Trains on daily bars of a market proxy (default SPY), classifies the
    current bar's regime, and exposes simple is_bullish()/is_bearish() checks.

    Thread-safe: refresh() can be called from a scheduler while is_bullish()
    is called from the signal processing path.
    """

    def __init__(
        self,
        symbol: str = "SPY",
        n_states: int = 7,
        history_days: int = 730,
        min_confidence: float = 0.5,
    ):
        self.symbol = symbol
        self.n_states = n_states
        self.history_days = history_days
        self.min_confidence = min_confidence

        # State (protected by lock)
        self._lock = threading.Lock()
        self._current_category: Optional[str] = None  # "bullish", "bearish", "neutral"
        self._current_label: Optional[str] = None
        self._current_confidence: float = 0.0
        self._last_refresh: Optional[datetime] = None
        self._regime_info: Optional[dict] = None
        self._error: Optional[str] = None
        self._trained = False

    def refresh(self) -> bool:
        """
        Retrain HMM and update current regime classification.

        Returns True if successful, False on error.
        Safe to call from scheduler thread.
        """
        try:
            from scripts.backtest_hmm import (
                engineer_features,
                fetch_market_data,
                label_regimes,
                train_hmm,
            )

            logger.info(f"[REGIME] Refreshing {self.symbol} regime ({self.n_states} states, {self.history_days}d)...")

            # Fetch and train
            df = fetch_market_data(self.symbol, self.history_days, "daily")
            features = engineer_features(df)
            model, scaler, states, posteriors, converged, n_iters, score = train_hmm(
                features, n_states=self.n_states
            )
            regime_info = label_regimes(model, states, features, self.n_states)

            # Extract current regime from last bar
            current_state = states[-1]
            category = regime_info["categories"].get(current_state, "neutral")
            label = regime_info["labels"].get(current_state, "UNKNOWN")
            confidence = float(posteriors[-1][current_state])

            # Update state atomically
            with self._lock:
                self._current_category = category
                self._current_label = label
                self._current_confidence = confidence
                self._last_refresh = datetime.now()
                self._regime_info = regime_info
                self._error = None
                self._trained = True

            status_icon = {"bullish": "+", "bearish": "-", "neutral": "~"}.get(category, "?")
            converge_str = "converged" if converged else "NOT converged"
            logger.info(
                f"[REGIME] {status_icon} {label} ({category}, {confidence:.0%} confidence) "
                f"| {converge_str} in {n_iters} iters"
            )
            return True

        except Exception as e:
            with self._lock:
                self._error = str(e)
            logger.error(f"[REGIME] Refresh failed: {e}")
            return False

    def is_bullish(self) -> bool:
        """Check if current regime is bullish with sufficient confidence."""
        with self._lock:
            if not self._trained:
                # Not yet trained — don't block trades
                return True
            return (
                self._current_category == "bullish"
                and self._current_confidence >= self.min_confidence
            )

    def is_bearish(self) -> bool:
        """Check if current regime is bearish."""
        with self._lock:
            if not self._trained:
                return False
            return self._current_category == "bearish"

    @property
    def category(self) -> Optional[str]:
        with self._lock:
            return self._current_category

    @property
    def label(self) -> Optional[str]:
        with self._lock:
            return self._current_label

    @property
    def confidence(self) -> float:
        with self._lock:
            return self._current_confidence

    @property
    def trained(self) -> bool:
        with self._lock:
            return self._trained

    def get_status(self) -> dict:
        """Get full regime status for API/dashboard."""
        with self._lock:
            return {
                "symbol": self.symbol,
                "trained": self._trained,
                "category": self._current_category,
                "label": self._current_label,
                "confidence": self._current_confidence,
                "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None,
                "n_states": self.n_states,
                "history_days": self.history_days,
                "error": self._error,
            }
