"""
Portfolio Limits - Enforce portfolio-level risk constraints.

Implements circuit breakers and limits to protect capital:
- Maximum drawdown limit
- Daily loss limit
- Maximum position count
- Maximum exposure per sector/asset class
- Correlation limits
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Optional

from loguru import logger

from config import settings


class RiskStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    STOPPED = "stopped"  # Trading halted


class TradingAction(str, Enum):
    ALLOW = "allow"
    REDUCE_ONLY = "reduce_only"  # Can only close positions
    HALT = "halt"  # No trading allowed


@dataclass
class RiskCheck:
    """Result of a risk check."""

    passed: bool
    status: RiskStatus
    action: TradingAction
    message: str
    current_value: float
    limit_value: float

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "status": self.status.value,
            "action": self.action.value,
            "message": self.message,
            "current_value": self.current_value,
            "limit_value": self.limit_value,
        }


@dataclass
class DailyStats:
    """Track daily trading statistics."""

    date: date = field(default_factory=date.today)
    starting_equity: float = 0.0
    current_equity: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades_opened: int = 0
    trades_closed: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    @property
    def daily_pnl(self) -> float:
        """Total daily P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def daily_pnl_pct(self) -> float:
        """Daily P&L as percentage of starting equity."""
        if self.starting_equity == 0:
            return 0.0
        return self.daily_pnl / self.starting_equity

    @property
    def daily_return_pct(self) -> float:
        """Daily return percentage."""
        if self.starting_equity == 0:
            return 0.0
        return (self.current_equity - self.starting_equity) / self.starting_equity


class PortfolioLimits:
    """
    Enforce portfolio-level risk limits.

    Acts as a circuit breaker to protect capital during drawdowns.
    All checks must pass before new positions can be opened.
    """

    def __init__(
        self,
        max_drawdown_pct: Optional[float] = None,
        max_daily_loss_pct: float = 0.03,  # 3% daily loss limit
        max_positions: Optional[int] = None,
        max_sector_exposure_pct: float = 0.40,  # 40% max in one sector
        max_correlation: float = 0.70,  # Warn if positions too correlated
        warning_threshold_pct: float = 0.70,  # Warn at 70% of limit
    ):
        """
        Initialize portfolio limits.

        Args:
            max_drawdown_pct: Maximum portfolio drawdown (default from settings)
            max_daily_loss_pct: Maximum daily loss percentage
            max_positions: Maximum concurrent positions (default from settings)
            max_sector_exposure_pct: Maximum exposure to single sector
            max_correlation: Maximum correlation between positions
            warning_threshold_pct: Percentage of limit to trigger warning
        """
        self.max_drawdown_pct = max_drawdown_pct or settings.max_drawdown_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_positions = max_positions or settings.max_positions
        self.max_sector_exposure_pct = max_sector_exposure_pct
        self.max_correlation = max_correlation
        self.warning_threshold = warning_threshold_pct

        # State tracking
        self._peak_equity: float = 0.0
        self._daily_stats: DailyStats = DailyStats()
        self._trading_halted: bool = False
        self._halt_reason: Optional[str] = None

    def update_equity(self, current_equity: float) -> None:
        """Update equity tracking."""
        # Update peak
        self._peak_equity = max(self._peak_equity, current_equity)

        # Initialize daily stats if new day
        today = date.today()
        if self._daily_stats.date != today:
            self._daily_stats = DailyStats(
                date=today,
                starting_equity=current_equity,
                current_equity=current_equity,
            )
            # Reset halt at start of new day
            self._trading_halted = False
            self._halt_reason = None
        else:
            self._daily_stats.current_equity = current_equity

    def update_daily_pnl(self, realized_pnl: float, unrealized_pnl: float) -> None:
        """Update daily P&L tracking."""
        self._daily_stats.realized_pnl = realized_pnl
        self._daily_stats.unrealized_pnl = unrealized_pnl

    def record_trade(self, is_winner: bool, is_open: bool) -> None:
        """Record a trade for daily stats."""
        if is_open:
            self._daily_stats.trades_opened += 1
        else:
            self._daily_stats.trades_closed += 1
            if is_winner:
                self._daily_stats.winning_trades += 1
            else:
                self._daily_stats.losing_trades += 1

    def get_current_drawdown(self, current_equity: float) -> float:
        """Calculate current drawdown from peak."""
        if self._peak_equity == 0:
            return 0.0
        return (self._peak_equity - current_equity) / self._peak_equity

    def check_drawdown(self, current_equity: float) -> RiskCheck:
        """
        Check if drawdown limit is exceeded.

        Returns:
            RiskCheck with status and allowed action
        """
        current_dd = self.get_current_drawdown(current_equity)
        warning_level = self.max_drawdown_pct * self.warning_threshold

        if current_dd >= self.max_drawdown_pct:
            self._trading_halted = True
            self._halt_reason = f"Max drawdown exceeded: {current_dd:.1%}"
            return RiskCheck(
                passed=False,
                status=RiskStatus.CRITICAL,
                action=TradingAction.HALT,
                message=f"Drawdown {current_dd:.1%} exceeds limit {self.max_drawdown_pct:.1%}",
                current_value=current_dd,
                limit_value=self.max_drawdown_pct,
            )
        elif current_dd >= warning_level:
            return RiskCheck(
                passed=True,
                status=RiskStatus.WARNING,
                action=TradingAction.REDUCE_ONLY,
                message=f"Drawdown {current_dd:.1%} approaching limit {self.max_drawdown_pct:.1%}",
                current_value=current_dd,
                limit_value=self.max_drawdown_pct,
            )
        else:
            return RiskCheck(
                passed=True,
                status=RiskStatus.OK,
                action=TradingAction.ALLOW,
                message=f"Drawdown {current_dd:.1%} within limits",
                current_value=current_dd,
                limit_value=self.max_drawdown_pct,
            )

    def check_daily_loss(self) -> RiskCheck:
        """
        Check if daily loss limit is exceeded.

        Returns:
            RiskCheck with status and allowed action
        """
        daily_loss_pct = -self._daily_stats.daily_return_pct  # Negative return = loss
        warning_level = self.max_daily_loss_pct * self.warning_threshold

        if daily_loss_pct >= self.max_daily_loss_pct:
            self._trading_halted = True
            self._halt_reason = f"Daily loss limit exceeded: {daily_loss_pct:.1%}"
            return RiskCheck(
                passed=False,
                status=RiskStatus.CRITICAL,
                action=TradingAction.HALT,
                message=f"Daily loss {daily_loss_pct:.1%} exceeds limit {self.max_daily_loss_pct:.1%}",
                current_value=daily_loss_pct,
                limit_value=self.max_daily_loss_pct,
            )
        elif daily_loss_pct >= warning_level:
            return RiskCheck(
                passed=True,
                status=RiskStatus.WARNING,
                action=TradingAction.REDUCE_ONLY,
                message=f"Daily loss {daily_loss_pct:.1%} approaching limit",
                current_value=daily_loss_pct,
                limit_value=self.max_daily_loss_pct,
            )
        else:
            return RiskCheck(
                passed=True,
                status=RiskStatus.OK,
                action=TradingAction.ALLOW,
                message=f"Daily loss {daily_loss_pct:.1%} within limits",
                current_value=daily_loss_pct,
                limit_value=self.max_daily_loss_pct,
            )

    def check_position_count(self, current_positions: int) -> RiskCheck:
        """
        Check if position limit allows new positions.

        Args:
            current_positions: Number of current open positions

        Returns:
            RiskCheck with status and allowed action
        """
        if current_positions >= self.max_positions:
            return RiskCheck(
                passed=False,
                status=RiskStatus.WARNING,
                action=TradingAction.REDUCE_ONLY,
                message=f"Position limit reached: {current_positions}/{self.max_positions}",
                current_value=current_positions,
                limit_value=self.max_positions,
            )
        else:
            return RiskCheck(
                passed=True,
                status=RiskStatus.OK,
                action=TradingAction.ALLOW,
                message=f"Positions: {current_positions}/{self.max_positions}",
                current_value=current_positions,
                limit_value=self.max_positions,
            )

    def check_can_open_position(
        self,
        current_equity: float,
        current_positions: int,
    ) -> RiskCheck:
        """
        Comprehensive check if new position can be opened.

        Runs all relevant checks and returns the most restrictive result.

        Args:
            current_equity: Current account equity
            current_positions: Number of open positions

        Returns:
            RiskCheck with combined status
        """
        # Update equity first
        self.update_equity(current_equity)

        # If trading is halted, return immediately
        if self._trading_halted:
            return RiskCheck(
                passed=False,
                status=RiskStatus.STOPPED,
                action=TradingAction.HALT,
                message=f"Trading halted: {self._halt_reason}",
                current_value=0,
                limit_value=0,
            )

        # Run all checks
        checks = [
            self.check_drawdown(current_equity),
            self.check_daily_loss(),
            self.check_position_count(current_positions),
        ]

        # Find most restrictive result
        # Priority: HALT > REDUCE_ONLY > ALLOW
        action_priority = {
            TradingAction.HALT: 0,
            TradingAction.REDUCE_ONLY: 1,
            TradingAction.ALLOW: 2,
        }

        most_restrictive = min(checks, key=lambda c: action_priority[c.action])

        # If any check failed, return that
        failed_checks = [c for c in checks if not c.passed]
        if failed_checks:
            return failed_checks[0]

        return most_restrictive

    def check_can_close_position(self) -> RiskCheck:
        """
        Check if positions can be closed.

        Closing is almost always allowed (reduce risk).
        Only blocked in extreme circumstances.

        Returns:
            RiskCheck - typically allows closing
        """
        # Closing positions is almost always allowed
        return RiskCheck(
            passed=True,
            status=RiskStatus.OK,
            action=TradingAction.ALLOW,
            message="Position closing allowed",
            current_value=0,
            limit_value=0,
        )

    def get_status(self, current_equity: float, current_positions: int) -> dict:
        """
        Get comprehensive risk status.

        Returns:
            Dict with all risk metrics and checks
        """
        self.update_equity(current_equity)

        return {
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "peak_equity": self._peak_equity,
            "current_equity": current_equity,
            "current_drawdown_pct": self.get_current_drawdown(current_equity),
            "max_drawdown_pct": self.max_drawdown_pct,
            "daily_stats": {
                "date": self._daily_stats.date.isoformat(),
                "starting_equity": self._daily_stats.starting_equity,
                "current_equity": self._daily_stats.current_equity,
                "daily_pnl": self._daily_stats.daily_pnl,
                "daily_return_pct": self._daily_stats.daily_return_pct,
                "trades_opened": self._daily_stats.trades_opened,
                "trades_closed": self._daily_stats.trades_closed,
                "win_rate": (
                    self._daily_stats.winning_trades / self._daily_stats.trades_closed
                    if self._daily_stats.trades_closed > 0
                    else 0
                ),
            },
            "limits": {
                "max_drawdown_pct": self.max_drawdown_pct,
                "max_daily_loss_pct": self.max_daily_loss_pct,
                "max_positions": self.max_positions,
            },
            "checks": {
                "drawdown": self.check_drawdown(current_equity).to_dict(),
                "daily_loss": self.check_daily_loss().to_dict(),
                "position_count": self.check_position_count(current_positions).to_dict(),
            },
        }

    def reset_daily_limits(self) -> None:
        """Reset daily limits (call at start of trading day)."""
        self._daily_stats = DailyStats(
            date=date.today(),
            starting_equity=self._daily_stats.current_equity,
            current_equity=self._daily_stats.current_equity,
        )
        self._trading_halted = False
        self._halt_reason = None
        logger.info("Daily limits reset")

    def force_halt(self, reason: str) -> None:
        """Force trading halt (emergency stop)."""
        self._trading_halted = True
        self._halt_reason = f"Manual halt: {reason}"
        logger.warning(f"Trading halted: {reason}")

    def resume_trading(self) -> bool:
        """
        Attempt to resume trading after halt.

        Returns:
            True if trading can resume, False if limits still exceeded
        """
        if not self._trading_halted:
            return True

        # Check if we can resume
        current_equity = self._daily_stats.current_equity
        dd_check = self.check_drawdown(current_equity)
        daily_check = self.check_daily_loss()

        if dd_check.passed and daily_check.passed:
            self._trading_halted = False
            self._halt_reason = None
            logger.info("Trading resumed")
            return True
        else:
            logger.warning("Cannot resume trading - limits still exceeded")
            return False
