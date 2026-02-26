"""
State persistence for the trading bot.

Saves and loads bot state to JSON file for recovery after restarts.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.bot.signals.base import Signal, SignalDirection


class BotState:
    """
    Manages bot state persistence.

    State includes:
    - Active signals (not yet executed)
    - Signal history (last N signals)
    - Last check timestamps per job
    - Configuration version
    """

    def __init__(
        self,
        state_file: Path,
        max_signal_history: int = 100,
    ):
        """
        Initialize state manager.

        Args:
            state_file: Path to JSON state file
            max_signal_history: Max signals to keep in history
        """
        self.state_file = Path(state_file)
        self.max_signal_history = max_signal_history

        # Ensure parent directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # State structure
        self._state: dict = {
            "version": 1,
            "last_updated": None,
            "active_signals": [],
            "signal_history": [],
            "job_timestamps": {},
            "metrics": {
                "signals_generated": 0,
                "signals_executed": 0,
                "signals_rejected": 0,
            },
        }

        # Load existing state if present
        self._load()

    def _load(self) -> None:
        """Load state from file if it exists."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    loaded = json.load(f)

                # Merge loaded state
                self._state.update(loaded)

            except (json.JSONDecodeError, IOError) as e:
                # Log error but continue with default state
                print(f"Warning: Could not load state file: {e}")

    def save(self) -> None:
        """Save current state to file."""
        self._state["last_updated"] = datetime.now().isoformat()

        try:
            # Write to temp file first, then rename (atomic)
            temp_file = self.state_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(self._state, f, indent=2, default=str)

            temp_file.rename(self.state_file)

        except IOError as e:
            print(f"Error saving state: {e}")
            raise

    def add_signal(self, signal: Signal) -> None:
        """
        Add a generated signal to active list and history.

        Args:
            signal: Signal to add
        """
        signal_dict = signal.to_dict()

        self._state["active_signals"].append(signal_dict)
        self._state["signal_history"].insert(0, signal_dict)
        self._state["metrics"]["signals_generated"] += 1

        # Trim history
        if len(self._state["signal_history"]) > self.max_signal_history:
            self._state["signal_history"] = self._state["signal_history"][
                : self.max_signal_history
            ]

        self.save()

    def remove_active_signal(self, symbol: str, executed: bool = True) -> Optional[dict]:
        """
        Remove a signal from active list.

        Args:
            symbol: Symbol to remove
            executed: Whether signal was executed or rejected

        Returns:
            Removed signal dict or None
        """
        for i, sig in enumerate(self._state["active_signals"]):
            if sig["symbol"] == symbol:
                removed = self._state["active_signals"].pop(i)

                if executed:
                    self._state["metrics"]["signals_executed"] += 1
                else:
                    self._state["metrics"]["signals_rejected"] += 1

                self.save()
                return removed

        return None

    def get_active_signals(self) -> list[dict]:
        """Get all active (pending) signals."""
        return self._state["active_signals"].copy()

    def get_signal_history(self, limit: int = 20) -> list[dict]:
        """
        Get recent signal history.

        Args:
            limit: Max signals to return

        Returns:
            List of signal dicts
        """
        return self._state["signal_history"][:limit]

    def has_active_signal(self, symbol: str) -> bool:
        """Check if symbol has an active signal."""
        return any(s["symbol"] == symbol for s in self._state["active_signals"])

    def update_job_timestamp(self, job_name: str) -> None:
        """
        Record when a job last ran.

        Args:
            job_name: Name of the scheduler job
        """
        self._state["job_timestamps"][job_name] = datetime.now().isoformat()
        self.save()

    def get_job_timestamp(self, job_name: str) -> Optional[datetime]:
        """
        Get when a job last ran.

        Args:
            job_name: Name of the scheduler job

        Returns:
            Datetime or None if never run
        """
        ts = self._state["job_timestamps"].get(job_name)
        if ts:
            return datetime.fromisoformat(ts)
        return None

    def get_metrics(self) -> dict:
        """Get signal metrics."""
        return self._state["metrics"].copy()

    def clear_active_signals(self) -> int:
        """
        Clear all active signals.

        Returns:
            Number of signals cleared
        """
        count = len(self._state["active_signals"])
        self._state["active_signals"] = []
        self.save()
        return count

    def get_state_summary(self) -> dict:
        """Get a summary of current state."""
        return {
            "version": self._state["version"],
            "last_updated": self._state["last_updated"],
            "active_signals": len(self._state["active_signals"]),
            "signal_history": len(self._state["signal_history"]),
            "metrics": self._state["metrics"],
            "job_timestamps": self._state["job_timestamps"],
        }
