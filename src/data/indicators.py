"""
Technical indicators for signal generation.

All functions take pandas Series/DataFrame and return pandas Series.
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average.

    Args:
        series: Price series (typically close)
        period: Lookback period

    Returns:
        SMA values
    """
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average.

    Args:
        series: Price series (typically close)
        period: Lookback period

    Returns:
        EMA values
    """
    return series.ewm(span=period, adjust=False).mean()


def macd(
    series: pd.Series,
    fast_period: int = 8,
    slow_period: int = 17,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence).

    Args:
        series: Price series (typically close)
        fast_period: Fast EMA period (default 8)
        slow_period: Slow EMA period (default 17)
        signal_period: Signal line EMA period (default 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    fast_ema = ema(series, fast_period)
    slow_ema = ema(series, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    Args:
        series: Price series (typically close)
        period: RSI period (default 14)

    Returns:
        RSI values (0-100)
    """
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands.

    Args:
        series: Price series (typically close)
        period: Moving average period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = sma(series, period)
    std = series.rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return upper, middle, lower


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default 14)

    Returns:
        ATR values
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return true_range.ewm(alpha=1/period, min_periods=period).mean()


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average Directional Index.

    Measures trend strength (not direction).
    ADX > 25 indicates strong trend.
    ADX < 20 indicates weak/no trend.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period (default 14)

    Returns:
        ADX values (0-100)
    """
    # Calculate +DM and -DM
    up_move = high.diff()
    down_move = (-low).diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Get ATR
    atr_values = atr(high, low, close, period)

    # Smoothed +DI and -DI
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr_values)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr_values)

    # DX and ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx_values = dx.ewm(alpha=1/period, min_periods=period).mean()

    return adx_values


def donchian_channel(
    high: pd.Series,
    low: pd.Series,
    period: int = 20,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Donchian Channel.

    Used for breakout detection.

    Args:
        high: High prices
        low: Low prices
        period: Lookback period (default 20)

    Returns:
        Tuple of (upper_channel, middle_channel, lower_channel)
    """
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    middle = (upper + lower) / 2

    return upper, middle, lower


def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Volume Simple Moving Average.

    Used for volume confirmation in breakouts.

    Args:
        volume: Volume series
        period: Lookback period (default 20)

    Returns:
        Volume SMA values
    """
    return sma(volume, period)


def is_breakout(
    close: pd.Series,
    high: pd.Series,
    volume: pd.Series,
    period: int = 20,
    volume_multiplier: float = 1.5,
) -> pd.Series:
    """
    Detect breakout signals.

    A breakout occurs when:
    - Price breaks above the N-period high
    - Volume is above average (multiplier * SMA)

    Args:
        close: Close prices
        high: High prices
        volume: Volume
        period: Lookback period (default 20)
        volume_multiplier: Volume threshold multiplier (default 1.5)

    Returns:
        Boolean series (True = breakout signal)
    """
    upper_channel, _, _ = donchian_channel(high, high.shift(1).fillna(high), period)
    prev_upper = upper_channel.shift(1)

    avg_volume = volume_sma(volume, period)

    price_breakout = close > prev_upper
    volume_confirm = volume > (avg_volume * volume_multiplier)

    return price_breakout & volume_confirm


def is_oversold(
    close: pd.Series,
    rsi_period: int = 14,
    rsi_threshold: int = 30,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> pd.Series:
    """
    Detect oversold conditions for mean reversion.

    Oversold when:
    - RSI is below threshold (default 30)
    - Price is below lower Bollinger Band

    Args:
        close: Close prices
        rsi_period: RSI period (default 14)
        rsi_threshold: RSI oversold level (default 30)
        bb_period: Bollinger Band period (default 20)
        bb_std: Bollinger Band std dev (default 2.0)

    Returns:
        Boolean series (True = oversold signal)
    """
    rsi_values = rsi(close, rsi_period)
    _, _, lower_bb = bollinger_bands(close, bb_period, bb_std)

    rsi_oversold = rsi_values < rsi_threshold
    below_bb = close < lower_bb

    return rsi_oversold & below_bb


def is_overbought(
    close: pd.Series,
    rsi_period: int = 14,
    rsi_threshold: int = 70,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> pd.Series:
    """
    Detect overbought conditions for mean reversion exits.

    Overbought when:
    - RSI is above threshold (default 70)
    - Price is above upper Bollinger Band

    Args:
        close: Close prices
        rsi_period: RSI period (default 14)
        rsi_threshold: RSI overbought level (default 70)
        bb_period: Bollinger Band period (default 20)
        bb_std: Bollinger Band std dev (default 2.0)

    Returns:
        Boolean series (True = overbought signal)
    """
    rsi_values = rsi(close, rsi_period)
    upper_bb, _, _ = bollinger_bands(close, bb_period, bb_std)

    rsi_overbought = rsi_values > rsi_threshold
    above_bb = close > upper_bb

    return rsi_overbought & above_bb
