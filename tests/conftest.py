"""
Pytest fixtures for swing-trader tests.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_bars() -> pd.DataFrame:
    """Generate sample OHLCV bars for testing."""
    np.random.seed(42)
    n_bars = 100

    # Generate random walk price
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_bars)
    prices = base_price * np.cumprod(1 + returns)

    # Generate OHLC from close prices
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq="1D")

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.uniform(-0.01, 0.01, n_bars)),
            "high": prices * (1 + np.random.uniform(0, 0.02, n_bars)),
            "low": prices * (1 - np.random.uniform(0, 0.02, n_bars)),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, n_bars),
        },
        index=dates,
    )

    # Ensure high >= open, close, low and low <= open, close, high
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    return df


@pytest.fixture
def volatile_bars() -> pd.DataFrame:
    """Generate highly volatile bars for testing."""
    np.random.seed(123)
    n_bars = 50

    base_price = 50.0
    returns = np.random.normal(0.002, 0.05, n_bars)  # Higher volatility
    prices = base_price * np.cumprod(1 + returns)

    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq="1D")

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.uniform(-0.03, 0.03, n_bars)),
            "high": prices * (1 + np.random.uniform(0, 0.05, n_bars)),
            "low": prices * (1 - np.random.uniform(0, 0.05, n_bars)),
            "close": prices,
            "volume": np.random.randint(500000, 2000000, n_bars),
        },
        index=dates,
    )

    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

    return df


@pytest.fixture
def small_account() -> dict:
    """Small aggressive account parameters."""
    return {
        "equity": 1000.0,
        "buying_power": 1000.0,
        "risk_pct": 0.02,  # 2% risk per trade
    }


@pytest.fixture
def medium_account() -> dict:
    """Medium sized account parameters."""
    return {
        "equity": 10000.0,
        "buying_power": 10000.0,
        "risk_pct": 0.01,  # 1% risk per trade
    }
