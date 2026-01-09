"""
Position monitor.

Monitors open positions for exit conditions (stops, targets, signals).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from src.bot.signals.base import Signal, SignalDirection, SignalGenerator
from src.core.alpaca_client import AlpacaClient
from src.core.position_manager import Position, PositionManager, PositionSide


@dataclass
class ExitSignal:
    """Signal that a position should be exited."""
    symbol: str
    reason: str
    current_price: float
    position: Position
    urgency: str  # "immediate" for stops, "normal" for signals

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "reason": self.reason,
            "current_price": self.current_price,
            "entry_price": self.position.entry_price,
            "unrealized_pnl": self.position.unrealized_pnl,
            "urgency": self.urgency,
            "timestamp": datetime.now().isoformat(),
        }


class PositionMonitor:
    """
    Monitor positions for exit conditions.

    Checks:
    1. Stop-loss hits
    2. Take-profit hits
    3. Trailing stop hits
    4. Strategy exit signals
    """

    def __init__(
        self,
        client: AlpacaClient,
        position_manager: PositionManager,
        strategies: Optional[dict[str, SignalGenerator]] = None,
    ):
        """
        Initialize position monitor.

        Args:
            client: Alpaca API client
            position_manager: Position tracking
            strategies: Optional dict of strategy name -> generator for exit signals
        """
        self.client = client
        self.position_manager = position_manager
        self.strategies = strategies or {}

    async def check_all_positions(self) -> list[ExitSignal]:
        """
        Check all open positions for exit conditions.

        Returns:
            List of ExitSignals for positions that should exit
        """
        exit_signals = []
        positions = self.position_manager.get_open_positions()

        if not positions:
            return exit_signals

        # Get current prices for all positions
        symbols = [p.symbol for p in positions]
        prices = await self._get_current_prices(symbols)

        for position in positions:
            price = prices.get(position.symbol)
            if price is None:
                continue

            # Update position with current price
            position.update_price(price)

            # Check exit conditions
            exit_signal = await self._check_position_exit(position, price)
            if exit_signal:
                exit_signals.append(exit_signal)

        return exit_signals

    async def check_position(self, symbol: str) -> Optional[ExitSignal]:
        """
        Check a specific position for exit conditions.

        Args:
            symbol: Position symbol to check

        Returns:
            ExitSignal if should exit, None otherwise
        """
        position = self.position_manager.get_position(symbol)
        if not position:
            return None

        price = await self._get_current_price(symbol)
        if price is None:
            return None

        position.update_price(price)
        return await self._check_position_exit(position, price)

    async def _check_position_exit(
        self,
        position: Position,
        current_price: float,
    ) -> Optional[ExitSignal]:
        """
        Check if a position should exit.

        Priority:
        1. Stop-loss (immediate)
        2. Trailing stop (immediate)
        3. Take-profit (immediate)
        4. Strategy exit signal (normal)
        """
        # 1. Check stop-loss
        if position.should_stop_loss():
            return ExitSignal(
                symbol=position.symbol,
                reason=f"Stop-loss triggered (${position.stop_loss:.2f})",
                current_price=current_price,
                position=position,
                urgency="immediate",
            )

        # 2. Check trailing stop
        if position.trailing_stop_pct and position.should_trailing_stop():
            trail_stop = position.get_trailing_stop_price()
            return ExitSignal(
                symbol=position.symbol,
                reason=f"Trailing stop triggered (${trail_stop:.2f})",
                current_price=current_price,
                position=position,
                urgency="immediate",
            )

        # 3. Check take-profit
        if position.should_take_profit():
            return ExitSignal(
                symbol=position.symbol,
                reason=f"Take-profit triggered (${position.take_profit:.2f})",
                current_price=current_price,
                position=position,
                urgency="immediate",
            )

        # 4. Check strategy exit signal
        if self.strategies:
            exit_reason = await self._check_strategy_exit(position)
            if exit_reason:
                return ExitSignal(
                    symbol=position.symbol,
                    reason=exit_reason,
                    current_price=current_price,
                    position=position,
                    urgency="normal",
                )

        return None

    async def _check_strategy_exit(self, position: Position) -> Optional[str]:
        """
        Check if strategy signals an exit.

        Args:
            position: Position to check

        Returns:
            Exit reason or None
        """
        # Get strategy that opened the position (stored in metadata or default)
        strategy_name = getattr(position, "strategy", None)
        strategy = self.strategies.get(strategy_name)

        if not strategy:
            # Try all strategies
            for name, strat in self.strategies.items():
                exit_reason = await self._check_with_strategy(position, strat)
                if exit_reason:
                    return exit_reason
            return None

        return await self._check_with_strategy(position, strategy)

    async def _check_with_strategy(
        self,
        position: Position,
        strategy: SignalGenerator,
    ) -> Optional[str]:
        """
        Check exit with a specific strategy.

        Args:
            position: Position to check
            strategy: Strategy to use for exit check

        Returns:
            Exit reason or None
        """
        try:
            # Get bars for the position
            bars = await self._get_bars(position.symbol, 50)
            if bars is None or bars.empty:
                return None

            direction = (
                SignalDirection.LONG
                if position.side == PositionSide.LONG
                else SignalDirection.SHORT
            )

            should_exit, reason = strategy.should_exit(
                symbol=position.symbol,
                bars=bars,
                entry_price=position.entry_price,
                direction=direction,
                current_price=position.current_price,
            )

            if should_exit:
                return f"Strategy exit: {reason}"

            return None

        except Exception as e:
            # Don't exit on errors, just log
            print(f"Error checking strategy exit for {position.symbol}: {e}")
            return None

    async def _get_current_prices(self, symbols: list[str]) -> dict[str, float]:
        """Get current prices for multiple symbols."""
        prices = {}
        for symbol in symbols:
            try:
                price = await self._get_current_price(symbol)
                if price:
                    prices[symbol] = price
            except Exception:
                pass
        return prices

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        try:
            return self.client.get_latest_price(symbol)
        except Exception:
            return None

    async def _get_bars(
        self,
        symbol: str,
        limit: int = 50,
    ) -> Optional[pd.DataFrame]:
        """Get recent price bars for a symbol."""
        try:
            # Determine timeframe based on asset type
            is_crypto = "/" in symbol or symbol.endswith("USD")
            timeframe = "1Hour" if is_crypto else "1Day"

            return self.client.get_bars(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit,
            )
        except Exception:
            return None

    def update_position_prices(self, prices: dict[str, float]) -> None:
        """
        Update position manager with current prices.

        Args:
            prices: Dict of symbol -> price
        """
        self.position_manager.update_prices(prices)

    def get_positions_summary(self) -> dict:
        """Get summary of all positions."""
        positions = self.position_manager.get_open_positions()

        total_value = sum(p.cost_basis for p in positions)
        total_pnl = sum(p.unrealized_pnl for p in positions)

        return {
            "count": len(positions),
            "total_value": total_value,
            "total_unrealized_pnl": total_pnl,
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "qty": p.qty,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                }
                for p in positions
            ],
        }
