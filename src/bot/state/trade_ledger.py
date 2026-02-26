"""
Trade Ledger - Persistent storage for closed trades.

Tracks all closed trades and calculates cumulative P&L for the experiment.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class TradeRecord:
    """Record of a completed trade."""

    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    realized_pnl: float
    realized_pnl_pct: float
    exit_reason: str
    fees: float = 0.0
    hold_duration_hours: float = 0.0
    strategy: str = ""


class TradeLedger:
    """
    Persistent trade journal for tracking experiment P&L.

    Stores all closed trades to JSON file, survives restarts.
    Tracks progress from starting capital toward goal.
    """

    def __init__(
        self,
        path: str = "state/trades.json",
        starting_capital: float = 250.0,
        goal: float = 25000.0,
    ):
        """
        Initialize trade ledger.

        Args:
            path: Path to JSON file for persistence
            starting_capital: Starting capital for experiment ($400)
            goal: Target capital ($4000)
        """
        self.path = Path(path)
        self.starting_capital = starting_capital
        self.goal = goal
        self.trades: list[TradeRecord] = []
        self._load()

    def _load(self) -> None:
        """Load trades from JSON file."""
        if not self.path.exists():
            logger.info(f"No trade ledger found at {self.path}, starting fresh")
            return

        try:
            with open(self.path, "r") as f:
                data = json.load(f)

            self.starting_capital = data.get("starting_capital", self.starting_capital)
            self.goal = data.get("goal", self.goal)

            for trade_data in data.get("trades", []):
                self.trades.append(TradeRecord(**trade_data))

            logger.info(
                f"Loaded {len(self.trades)} trades from ledger, "
                f"realized P&L: ${self.get_total_realized_pnl():.2f}"
            )
        except Exception as e:
            logger.error(f"Failed to load trade ledger: {e}")

    def _save(self) -> None:
        """Save trades to JSON file."""
        try:
            # Ensure directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "starting_capital": self.starting_capital,
                "goal": self.goal,
                "updated_at": datetime.now().isoformat(),
                "trades": [asdict(t) for t in self.trades],
            }

            with open(self.path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save trade ledger: {e}")

    def record_trade(self, position) -> None:
        """
        Record a closed position as a trade.

        Args:
            position: Closed Position object from position_manager
        """
        if position.exit_price is None:
            logger.warning(f"Cannot record unclosed position: {position.symbol}")
            return

        # Estimate fees (crypto 0.25%, stocks 0)
        is_crypto = "/" in position.symbol or position.symbol.endswith("USD")
        fee_rate = 0.0025 if is_crypto else 0.0
        fees = (position.cost_basis + position.market_value) * fee_rate

        trade = TradeRecord(
            symbol=position.symbol,
            side=position.side.value,
            qty=position.qty,
            entry_price=position.entry_price,
            exit_price=position.exit_price,
            entry_time=position.entry_time.isoformat(),
            exit_time=position.exit_time.isoformat() if position.exit_time else datetime.now().isoformat(),
            realized_pnl=position.realized_pnl or 0.0,
            realized_pnl_pct=(position.realized_pnl_pct or 0.0) * 100,
            exit_reason=position.exit_reason or "unknown",
            fees=fees,
            hold_duration_hours=position.hold_duration or 0.0,
            strategy=position.strategy or "",
        )

        self.trades.append(trade)
        self._save()

        logger.info(
            f"Recorded trade: {trade.symbol} "
            f"P&L: ${trade.realized_pnl:.2f} ({trade.realized_pnl_pct:+.1f}%) "
            f"Reason: {trade.exit_reason}"
        )

    def get_total_realized_pnl(self) -> float:
        """Get sum of all realized P&L from closed trades."""
        return sum(t.realized_pnl for t in self.trades)

    def get_stats(self) -> dict:
        """
        Get trading statistics.

        Returns:
            Dict with total_realized, win_count, loss_count, win_rate, etc.
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "total_realized_pnl": 0.0,
                "win_count": 0,
                "loss_count": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_hold_hours": 0.0,
            }

        wins = [t for t in self.trades if t.realized_pnl > 0]
        losses = [t for t in self.trades if t.realized_pnl < 0]

        return {
            "total_trades": len(self.trades),
            "total_realized_pnl": self.get_total_realized_pnl(),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": len(wins) / len(self.trades) * 100 if self.trades else 0.0,
            "avg_win": sum(t.realized_pnl for t in wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(t.realized_pnl for t in losses) / len(losses) if losses else 0.0,
            "largest_win": max((t.realized_pnl for t in wins), default=0.0),
            "largest_loss": min((t.realized_pnl for t in losses), default=0.0),
            "avg_hold_hours": sum(t.hold_duration_hours for t in self.trades) / len(self.trades),
        }

    def get_experiment_progress(self, unrealized_pnl: float = 0.0) -> dict:
        """
        Get experiment progress toward goal.

        Args:
            unrealized_pnl: Current unrealized P&L from open positions

        Returns:
            Dict with starting, current, goal, progress percentage
        """
        realized = self.get_total_realized_pnl()
        total_pnl = realized + unrealized_pnl
        current_value = self.starting_capital + total_pnl
        progress_pct = (current_value / self.goal) * 100

        return {
            "starting_capital": self.starting_capital,
            "goal": self.goal,
            "realized_pnl": realized,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "current_value": current_value,
            "progress_pct": progress_pct,
            "remaining": self.goal - current_value,
        }

    def get_trades(self, limit: int = 50) -> list[dict]:
        """Get recent trades as dicts."""
        return [asdict(t) for t in self.trades[-limit:]]
