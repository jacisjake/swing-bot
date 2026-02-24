"""Signal generation strategies."""

from src.bot.signals.base import Signal, SignalDirection, SignalGenerator
from src.bot.signals.breakout import BreakoutStrategy
from src.bot.signals.macd import MACDStrategy
from src.bot.signals.mean_reversion import MeanReversionStrategy
from src.bot.signals.momentum_surge import MomentumSurgeStrategy

__all__ = [
    "Signal",
    "SignalDirection",
    "SignalGenerator",
    "BreakoutStrategy",
    "MACDStrategy",
    "MeanReversionStrategy",
    "MomentumSurgeStrategy",
]
