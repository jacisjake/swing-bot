"""
Position monitor.

Monitors open positions for exit conditions (stops, targets, signals).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

import pytz

from src.bot.signals.base import Signal, SignalDirection, SignalGenerator
from src.core.alpaca_client import AlpacaClient
from src.core.position_manager import Position, PositionManager, PositionSide

ET = pytz.timezone("America/New_York")


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
        trading_window_end: str = "10:00",
    ):
        """
        Initialize position monitor.

        Args:
            client: Alpaca API client
            position_manager: Position tracking
            strategies: Optional dict of strategy name -> generator for exit signals
            trading_window_end: End of trading window (HH:MM ET) for time-based exit
        """
        self.client = client
        self.position_manager = position_manager
        self.strategies = strategies or {}

        # Parse trading window end for time-based exit
        parts = trading_window_end.split(":")
        from datetime import time
        self._window_end = time(int(parts[0]), int(parts[1]))

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

    async def check_position_at_price(
        self, symbol: str, price: float
    ) -> Optional[ExitSignal]:
        """
        Check a position for exit at a given price (quote-driven).

        Called by StreamHandler on every quote update. Only checks fast
        exit conditions (stops, targets, trailing) — skips expensive
        strategy exit checks (those run on 5-min bar close instead).

        Args:
            symbol: Position symbol
            price: Current price from quote stream

        Returns:
            ExitSignal if should exit, None otherwise
        """
        position = self.position_manager.get_position(symbol)
        if not position:
            return None

        position.update_price(price)

        # Time-based exit
        if self._is_past_trading_window():
            return ExitSignal(
                symbol=position.symbol,
                reason=f"Time exit: past trading window ({self._window_end.strftime('%H:%M')} ET)",
                current_price=price,
                position=position,
                urgency="immediate",
            )

        # Breakeven stop adjustment
        self._check_breakeven_stop(position)

        # Stop-loss
        if position.should_stop_loss():
            return ExitSignal(
                symbol=position.symbol,
                reason=f"Stop-loss hit @ ${position.stop_loss:.2f}",
                current_price=price,
                position=position,
                urgency="immediate",
            )

        # Take-profit
        if position.should_take_profit():
            return ExitSignal(
                symbol=position.symbol,
                reason=f"Take-profit hit @ ${position.take_profit:.2f}",
                current_price=price,
                position=position,
                urgency="immediate",
            )

        # Trailing stop
        if position.should_trailing_stop():
            trailing_price = position.get_trailing_stop_price()
            return ExitSignal(
                symbol=position.symbol,
                reason=f"Trailing stop hit @ ${trailing_price:.2f} (high: ${position.highest_price:.2f})",
                current_price=price,
                position=position,
                urgency="immediate",
            )

        return None

    async def _check_position_exit(
        self,
        position: Position,
        current_price: float,
    ) -> Optional[ExitSignal]:
        """
        Check if a position should exit.

        Checks in order of urgency:
        1. Time-based exit (past trading window)
        2. Stop-loss (immediate)
        3. Take-profit (immediate)
        4. Trailing stop (immediate)
        5. Strategy exit signal (normal)
        """
        # Check time-based exit first (day trading: close after window)
        if self._is_past_trading_window():
            return ExitSignal(
                symbol=position.symbol,
                reason=f"Time exit: past trading window ({self._window_end.strftime('%H:%M')} ET)",
                current_price=current_price,
                position=position,
                urgency="immediate",
            )

        # Check and apply breakeven stop after 1R profit
        self._check_breakeven_stop(position)

        # Check stop-loss first (highest priority)
        if position.should_stop_loss():
            return ExitSignal(
                symbol=position.symbol,
                reason=f"Stop-loss hit @ ${position.stop_loss:.2f}",
                current_price=current_price,
                position=position,
                urgency="immediate",
            )

        # Check take-profit
        if position.should_take_profit():
            return ExitSignal(
                symbol=position.symbol,
                reason=f"Take-profit hit @ ${position.take_profit:.2f}",
                current_price=current_price,
                position=position,
                urgency="immediate",
            )

        # Check trailing stop
        if position.should_trailing_stop():
            trailing_price = position.get_trailing_stop_price()
            return ExitSignal(
                symbol=position.symbol,
                reason=f"Trailing stop hit @ ${trailing_price:.2f} (high: ${position.highest_price:.2f})",
                current_price=current_price,
                position=position,
                urgency="immediate",
            )

        # Check strategy exit signal (MACD crossover)
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

    def _check_breakeven_stop(self, position: Position) -> None:
        """
        Move stop to breakeven after 1R profit is reached.

        1R = initial risk (distance from entry to original stop)
        When profit >= 1R, move stop_loss to entry_price

        Args:
            position: Position to check
        """
        if position.stop_loss is None:
            return

        # Calculate initial risk (1R)
        initial_risk = abs(position.entry_price - position.stop_loss)
        if initial_risk == 0:
            return

        # Calculate current profit
        if position.side == PositionSide.LONG:
            current_profit = position.current_price - position.entry_price
            profit_in_r = current_profit / initial_risk

            # Move to breakeven after 1R profit
            if profit_in_r >= 1.0 and position.stop_loss < position.entry_price:
                old_stop = position.stop_loss
                position.stop_loss = position.entry_price
                logger.info(
                    f"[BREAKEVEN] {position.symbol}: Moved stop to breakeven "
                    f"@ ${position.entry_price:.2f} (was ${old_stop:.2f}, profit={profit_in_r:.1f}R)"
                )
        else:  # SHORT
            current_profit = position.entry_price - position.current_price
            profit_in_r = current_profit / initial_risk

            if profit_in_r >= 1.0 and position.stop_loss > position.entry_price:
                old_stop = position.stop_loss
                position.stop_loss = position.entry_price
                logger.info(
                    f"[BREAKEVEN] {position.symbol}: Moved stop to breakeven "
                    f"@ ${position.entry_price:.2f} (was ${old_stop:.2f}, profit={profit_in_r:.1f}R)"
                )

    def _is_past_trading_window(self) -> bool:
        """
        Check if current time is past the trading window end (ET).

        Uses Alpaca market clock to detect holidays (don't force-close
        positions on non-trading days when the time coincidentally matches).
        """
        now_et = datetime.now(ET)

        # Weekend check (fast path)
        if now_et.weekday() >= 5:
            return False

        # Use Alpaca market clock to check if today is a trading day
        # Don't force time-exit on holidays when market is closed
        try:
            clock = self.client.trading.get_clock()
            if not clock.is_open:
                # Market is closed — check if it was open today at all
                # If next_open is today, market hasn't opened yet (pre-market)
                # If next_open is tomorrow+, today might be a holiday
                if hasattr(clock.next_open, "astimezone"):
                    next_open_et = clock.next_open.astimezone(ET)
                else:
                    next_open_et = clock.next_open

                if next_open_et.date() > now_et.date() and now_et.time() < self._window_end:
                    # Holiday: next open is a future day and we haven't hit window end
                    # Don't trigger time exit
                    return False
        except Exception:
            pass  # Fall through to simple time check

        current_time = now_et.time()
        return current_time >= self._window_end

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

        # If no strategy name, default to momentum_pullback
        if not strategy_name:
            strategy_name = "momentum_pullback"

        strategy = self.strategies.get(strategy_name)
        if not strategy:
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
            logger.error(f"Error checking strategy exit for {position.symbol}: {e}")
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
            # Stocks use 5Min (MACD strategy), Crypto uses 1Hour (mean reversion)
            is_crypto = "/" in symbol or symbol.endswith("USD")
            timeframe = "1Hour" if is_crypto else "5Min"

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
                }
                for p in positions
            ],
        }
