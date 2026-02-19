#!/usr/bin/env python3
"""
HMM Regime Detection Backtest.

Trains a Gaussian HMM on market data to classify regimes (bull, bear, choppy),
then backtests a regime-filtered strategy: only enter long when the regime is
bullish AND 7/8 technical confirmations pass. Exit on regime flip to bearish.
48-hour cooldown after each exit.

Compares regime-filtered performance vs buy-and-hold.

Usage:
    python scripts/backtest_hmm.py
    python scripts/backtest_hmm.py --symbol QQQ --days 365 --states 5
    python scripts/backtest_hmm.py --timeframe hourly --confirmations 6
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytz

sys.path.insert(0, ".")

from src.data.indicators import (
    macd as calc_macd,
    rsi as calc_rsi,
    atr as calc_atr,
    adx as calc_adx,
    bollinger_bands as calc_bbands,
    sma as calc_sma,
    volume_sma as calc_vol_sma,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("hmmlearn").setLevel(logging.WARNING)
logging.getLogger("sklearn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

ET = pytz.timezone("America/New_York")


# ═══════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════


def fetch_market_data(symbol: str, days: int, timeframe: str) -> pd.DataFrame:
    """Fetch historical OHLCV data from yfinance."""
    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=days)
    interval = "1d" if timeframe == "daily" else "1h"

    logger.info(f"  Fetching {symbol} {timeframe} data ({days} days)...")
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    # Flatten multi-index columns if present (yfinance sometimes returns these)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Normalize column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Drop adj close if present
    if "adj close" in df.columns:
        df = df.drop(columns=["adj close"])

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the 3 HMM input features from OHLCV data."""
    features = pd.DataFrame(index=df.index)

    # Feature 1: Log returns
    features["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Feature 2: Normalized range (intrabar volatility proxy)
    features["norm_range"] = (df["high"] - df["low"]) / df["close"]

    # Feature 3: Volume change (percent change of volume)
    features["volume_change"] = df["volume"].pct_change()

    # Drop NaN rows (first row from shift/pct_change)
    features = features.dropna()

    # Replace inf with 0 (can happen with zero volume bars)
    features = features.replace([np.inf, -np.inf], 0.0)

    return features


# ═══════════════════════════════════════════════════════════════════════════
#  HMM TRAINING
# ═══════════════════════════════════════════════════════════════════════════


def train_hmm(features: pd.DataFrame, n_states: int = 7, n_iter: int = 200):
    """Train a Gaussian HMM on the feature matrix."""
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    # Scale features (HMM is sensitive to feature magnitude)
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=42,
    )
    model.fit(X)

    states = model.predict(X)

    # Get posterior probabilities for confidence scoring
    posteriors = model.predict_proba(X)

    converged = model.monitor_.converged
    n_iters = model.monitor_.iter
    score = model.score(X)

    return model, scaler, states, posteriors, converged, n_iters, score


# ═══════════════════════════════════════════════════════════════════════════
#  REGIME LABELING
# ═══════════════════════════════════════════════════════════════════════════


def label_regimes(
    model, states: np.ndarray, features: pd.DataFrame, n_states: int
) -> dict:
    """Auto-label each hidden state by sorting on mean return."""
    state_stats = {}

    for i in range(n_states):
        mask = states == i
        if not np.any(mask):
            continue
        state_returns = features["log_return"].values[mask]
        state_range = features["norm_range"].values[mask]
        state_vol = features["volume_change"].values[mask]

        state_stats[i] = {
            "mean_return": float(np.mean(state_returns)),
            "std_return": float(np.std(state_returns)),
            "mean_range": float(np.mean(state_range)),
            "mean_vol_change": float(np.mean(state_vol)),
            "frequency": float(np.sum(mask)) / len(states),
            "count": int(np.sum(mask)),
        }

    # Sort states by mean return descending
    sorted_states = sorted(
        state_stats.keys(), key=lambda s: state_stats[s]["mean_return"], reverse=True
    )

    n = len(sorted_states)

    # Label with human-readable names
    labels = {}
    for rank, state_id in enumerate(sorted_states):
        if rank == 0:
            labels[state_id] = "BULL RUN"
        elif rank == 1:
            labels[state_id] = "BULL MILD"
        elif rank == n - 1:
            labels[state_id] = "CRASH"
        elif rank == n - 2:
            labels[state_id] = "BEAR"
        else:
            labels[state_id] = f"CHOPPY {rank - 1}"

    # Broad categories for trading decisions
    categories = {}
    for rank, state_id in enumerate(sorted_states):
        if rank < n * 0.3:
            categories[state_id] = "bullish"
        elif rank >= n * 0.7:
            categories[state_id] = "bearish"
        else:
            categories[state_id] = "neutral"

    return {
        "stats": state_stats,
        "labels": labels,
        "categories": categories,
        "sorted_states": sorted_states,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  TECHNICAL CONFIRMATIONS
# ═══════════════════════════════════════════════════════════════════════════


def compute_indicators(df: pd.DataFrame) -> dict:
    """Pre-compute all technical indicators for the full dataset."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    macd_line, macd_signal, macd_hist = calc_macd(
        close, fast_period=12, slow_period=26, signal_period=9
    )
    rsi_vals = calc_rsi(close, period=14)
    atr_vals = calc_atr(high, low, close, period=14)
    adx_vals = calc_adx(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calc_bbands(close, period=20, std_dev=2.0)
    sma_20 = calc_sma(close, period=20)
    vol_sma_20 = calc_vol_sma(volume, period=20)

    # ATR rolling average for volatility comparison
    atr_avg_50 = atr_vals.rolling(window=50).mean()

    return {
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "rsi": rsi_vals,
        "atr": atr_vals,
        "atr_avg_50": atr_avg_50,
        "adx": adx_vals,
        "bb_upper": bb_upper,
        "bb_mid": bb_mid,
        "bb_lower": bb_lower,
        "sma_20": sma_20,
        "vol_sma_20": vol_sma_20,
    }


def check_confirmations(
    idx: int,
    df: pd.DataFrame,
    indicators: dict,
    regime_prob: float,
) -> tuple[int, dict]:
    """
    Check 8 technical confirmations at a given bar index.

    Returns (count_passed, details_dict).
    """
    close = float(df["close"].iloc[idx])
    volume = float(df["volume"].iloc[idx])

    results = {}

    # 1. RSI not overbought (< 80)
    rsi_val = float(indicators["rsi"].iloc[idx])
    results["RSI < 80"] = (not np.isnan(rsi_val)) and rsi_val < 80

    # 2. Positive momentum: close > SMA(20)
    sma_val = float(indicators["sma_20"].iloc[idx])
    results["Close > SMA(20)"] = (not np.isnan(sma_val)) and close > sma_val

    # 3. Volatility OK: ATR within 0.5x-3x of 50-period ATR average
    atr_val = float(indicators["atr"].iloc[idx])
    atr_avg = float(indicators["atr_avg_50"].iloc[idx])
    if not np.isnan(atr_val) and not np.isnan(atr_avg) and atr_avg > 0:
        ratio = atr_val / atr_avg
        results["ATR in range"] = 0.5 <= ratio <= 3.0
    else:
        results["ATR in range"] = False

    # 4. Volume above average
    vol_sma = float(indicators["vol_sma_20"].iloc[idx])
    results["Vol > avg"] = (not np.isnan(vol_sma)) and vol_sma > 0 and volume > vol_sma

    # 5. ADX trending (> 20)
    adx_val = float(indicators["adx"].iloc[idx])
    results["ADX > 20"] = (not np.isnan(adx_val)) and adx_val > 20

    # 6. Price above support: close > Bollinger lower band
    bb_lower = float(indicators["bb_lower"].iloc[idx])
    results["Close > BB lower"] = (not np.isnan(bb_lower)) and close > bb_lower

    # 7. MACD bullish: MACD line > signal line
    macd_l = float(indicators["macd_line"].iloc[idx])
    macd_s = float(indicators["macd_signal"].iloc[idx])
    results["MACD > signal"] = (
        not np.isnan(macd_l) and not np.isnan(macd_s) and macd_l > macd_s
    )

    # 8. Regime confidence > 0.6
    results["Confidence > 60%"] = regime_prob > 0.6

    count = sum(1 for v in results.values() if v)
    return count, results


# ═══════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════


def simulate_trades(
    df: pd.DataFrame,
    states: np.ndarray,
    posteriors: np.ndarray,
    regime_info: dict,
    indicators: dict,
    min_confirmations: int = 7,
    cooldown_hours: int = 48,
    starting_equity: float = 10000.0,
    timeframe: str = "daily",
) -> dict:
    """
    Walk bar-by-bar, trading on regime + confirmations.

    Entry: regime is bullish + min_confirmations/8 pass
    Exit: regime flips to bearish
    Cooldown: cooldown_hours after any exit
    """
    # Offset: features/states start at df index 1 (first row dropped for diff)
    # So states[i] corresponds to df.iloc[i + 1]
    offset = len(df) - len(states)

    # Bars per hour for cooldown calculation
    if timeframe == "daily":
        bars_per_hour = 1 / 6.5  # ~1 bar per 6.5 trading hours
    else:
        bars_per_hour = 1.0  # 1 bar per hour

    cooldown_bars = int(cooldown_hours * bars_per_hour)

    equity = starting_equity
    position = None  # {"entry_price", "entry_idx", "entry_date", "regime"}
    trades = []
    equity_curve = []
    bars_since_exit = cooldown_bars + 1  # Start eligible

    # Need enough bars for indicators to warm up (50 bars for ATR avg)
    start_idx = max(offset, 60)

    for i in range(start_idx, len(df)):
        state_idx = i - offset
        if state_idx < 0 or state_idx >= len(states):
            equity_curve.append(equity)
            continue

        current_state = states[state_idx]
        current_category = regime_info["categories"].get(current_state, "neutral")
        current_label = regime_info["labels"].get(current_state, "UNKNOWN")
        state_prob = float(posteriors[state_idx][current_state])

        bar_close = float(df["close"].iloc[i])
        bar_date = df.index[i]

        if position is not None:
            # ── Check exit: regime flipped to bearish ──
            if current_category == "bearish":
                exit_price = bar_close
                pnl_pct = (exit_price - position["entry_price"]) / position[
                    "entry_price"
                ]
                pnl_dollar = equity * pnl_pct
                equity += pnl_dollar

                trades.append(
                    {
                        "entry_date": position["entry_date"],
                        "exit_date": bar_date,
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "pnl_pct": pnl_pct,
                        "pnl_dollar": pnl_dollar,
                        "regime_entry": position["regime"],
                        "regime_exit": current_label,
                        "confirmations": position["confirmations"],
                        "exit_reason": "Regime flip to bearish",
                    }
                )
                position = None
                bars_since_exit = 0
            else:
                # Mark-to-market for equity curve
                unrealized_pct = (bar_close - position["entry_price"]) / position[
                    "entry_price"
                ]
                equity_curve.append(equity * (1 + unrealized_pct))
                continue
        else:
            bars_since_exit += 1

            # ── Check entry conditions ──
            if current_category == "bullish" and bars_since_exit > cooldown_bars:
                count, details = check_confirmations(i, df, indicators, state_prob)

                if count >= min_confirmations:
                    position = {
                        "entry_price": bar_close,
                        "entry_idx": i,
                        "entry_date": bar_date,
                        "regime": current_label,
                        "confirmations": count,
                    }

        equity_curve.append(
            equity * (1 + (bar_close - position["entry_price"]) / position["entry_price"])
            if position
            else equity
        )

    # Close any open position at end
    if position is not None:
        exit_price = float(df["close"].iloc[-1])
        pnl_pct = (exit_price - position["entry_price"]) / position["entry_price"]
        pnl_dollar = equity * pnl_pct
        equity += pnl_dollar

        trades.append(
            {
                "entry_date": position["entry_date"],
                "exit_date": df.index[-1],
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "pnl_dollar": pnl_dollar,
                "regime_entry": position["regime"],
                "regime_exit": "END",
                "confirmations": position["confirmations"],
                "exit_reason": "End of data",
            }
        )

    # Calculate metrics
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]

    # Max drawdown from equity curve
    peak = starting_equity
    max_dd = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

    # Buy and hold
    first_close = float(df["close"].iloc[start_idx])
    last_close = float(df["close"].iloc[-1])
    buy_hold_return = (last_close - first_close) / first_close

    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0.0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0.0
    gross_profit = sum(t["pnl_pct"] for t in wins)
    gross_loss = abs(sum(t["pnl_pct"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "final_equity": equity,
        "starting_equity": starting_equity,
        "total_return": (equity - starting_equity) / starting_equity,
        "buy_hold_return": buy_hold_return,
        "alpha": (equity - starting_equity) / starting_equity - buy_hold_return,
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  OUTPUT
# ═══════════════════════════════════════════════════════════════════════════


def print_header(symbol: str, days: int, n_states: int, timeframe: str):
    now_et = datetime.now(ET)
    print()
    print("=" * 78)
    print("  HMM REGIME DETECTION BACKTEST")
    print(f"  {now_et.strftime('%A, %B %d, %Y %I:%M %p ET')}")
    print(f"  Symbol: {symbol} | History: {days} days | States: {n_states} | {timeframe}")
    print("=" * 78)


def print_data_summary(df: pd.DataFrame, features: pd.DataFrame):
    print()
    print("=" * 78)
    print("  PHASE 1: DATA FETCH & FEATURE ENGINEERING")
    print("=" * 78)
    print()
    print(f"  Fetched {len(df)} bars")
    print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Features: log_return, norm_range, volume_change ({len(features)} samples)")
    print(f"  Close range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")


def print_training_summary(converged: bool, n_iters: int, score: float, n_states: int):
    print()
    print("=" * 78)
    print(f"  PHASE 2: HMM TRAINING ({n_states} hidden states)")
    print("=" * 78)
    print()
    status = "converged" if converged else "did NOT converge (results may be unstable)"
    print(f"  Model {status} in {n_iters} iterations")
    print(f"  Log-likelihood: {score:.2f}")


def print_regime_table(regime_info: dict, states: np.ndarray):
    print()
    print("=" * 78)
    print("  PHASE 3: REGIME SUMMARY")
    print("=" * 78)
    print()
    print(
        f"  {'':2} {'State':<10} {'Label':<14} {'Category':<10} "
        f"{'Mean Ret':>9} {'Volatility':>11} {'Frequency':>10} {'Count':>7}"
    )
    print(
        f"  {'':2} {'─' * 10} {'─' * 14} {'─' * 10} "
        f"{'─' * 9} {'─' * 11} {'─' * 10} {'─' * 7}"
    )

    icon_map = {"bullish": "+", "bearish": "-", "neutral": "~"}

    for state_id in regime_info["sorted_states"]:
        stats = regime_info["stats"][state_id]
        label = regime_info["labels"][state_id]
        cat = regime_info["categories"][state_id]
        icon = icon_map.get(cat, " ")

        print(
            f"  {icon} State {state_id:<4} {label:<14} {cat:<10} "
            f"{stats['mean_return']:>+8.4f}  {stats['std_return']:>9.4f}  "
            f"{stats['frequency']:>8.1%}  {stats['count']:>6}"
        )


def print_current_regime(
    states: np.ndarray,
    posteriors: np.ndarray,
    regime_info: dict,
    df: pd.DataFrame,
):
    print()
    print("=" * 78)
    print("  PHASE 4: CURRENT REGIME DETECTION")
    print("=" * 78)
    print()

    current_state = states[-1]
    current_label = regime_info["labels"].get(current_state, "UNKNOWN")
    current_cat = regime_info["categories"].get(current_state, "neutral")
    current_prob = float(posteriors[-1][current_state])

    icon_map = {"bullish": "+", "bearish": "-", "neutral": "~"}
    icon = icon_map.get(current_cat, " ")

    print(f"  {icon} CURRENT: State {current_state} ({current_label}) -- {current_cat}")
    print(f"    Confidence: {current_prob:.1%}")
    print(f"    Date: {df.index[-1].strftime('%Y-%m-%d')}")

    # Last 5 regime transitions
    transitions = []
    prev_state = states[0]
    for i in range(1, len(states)):
        if states[i] != prev_state:
            transitions.append(regime_info["labels"].get(states[i], f"S{states[i]}"))
            prev_state = states[i]

    if transitions:
        recent = transitions[-5:]
        print(f"    Recent transitions: {' > '.join(recent)}")


def print_backtest_results(results: dict, min_confirmations: int, cooldown: int):
    print()
    print("=" * 78)
    print("  PHASE 5: BACKTEST RESULTS")
    print("=" * 78)
    print()
    print(f"  Strategy: Long when regime=bullish + {min_confirmations}/8 confirmations")
    print(f"  Exit: regime flips to bearish | Cooldown: {cooldown}h")
    print()
    print(f"  {'Metric':<30} {'Value':>18}")
    print(f"  {'─' * 30} {'─' * 18}")
    print(f"  {'Total trades':<30} {results['total_trades']:>18}")
    print(f"  {'Wins / Losses':<30} {results['wins']:>8} / {results['losses']:<8}")
    print(f"  {'Win rate':<30} {results['win_rate']:>17.1%}")
    print(f"  {'Total return':<30} {results['total_return']:>+17.1%}")
    print(f"  {'Buy & hold return':<30} {results['buy_hold_return']:>+17.1%}")
    print(f"  {'Alpha vs buy & hold':<30} {results['alpha']:>+17.1%}")
    print(f"  {'Max drawdown':<30} {results['max_drawdown']:>17.1%}")
    print(f"  {'Avg win':<30} {results['avg_win']:>+17.2%}")
    print(f"  {'Avg loss':<30} {results['avg_loss']:>+17.2%}")
    pf = results["profit_factor"]
    pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
    print(f"  {'Profit factor':<30} {pf_str:>18}")
    print(
        f"  {'Final equity':<30} "
        f"{'${:,.2f}'.format(results['final_equity']):>18}"
    )


def print_trade_log(trades: list, limit: int = 20):
    print()
    print("=" * 78)
    print(f"  PHASE 6: TRADE LOG (last {min(limit, len(trades))} of {len(trades)} trades)")
    print("=" * 78)

    if not trades:
        print()
        print("  No trades taken.")
        return

    print()
    print(
        f"  {'#':>3}  {'Entry Date':<12} {'Entry':>9} {'Exit':>9} "
        f"{'P&L':>8} {'Regime':<12} {'Conf':>4} {'Exit Reason':<20}"
    )
    print(
        f"  {'─' * 3}  {'─' * 12} {'─' * 9} {'─' * 9} "
        f"{'─' * 8} {'─' * 12} {'─' * 4} {'─' * 20}"
    )

    for i, t in enumerate(trades[-limit:], 1):
        entry_date = (
            t["entry_date"].strftime("%Y-%m-%d")
            if hasattr(t["entry_date"], "strftime")
            else str(t["entry_date"])[:10]
        )
        pnl_str = f"{t['pnl_pct']:+.1%}"
        result_icon = "+" if t["pnl_pct"] > 0 else "-"

        print(
            f"  {result_icon}{i:>2}  {entry_date:<12} ${t['entry_price']:>7.2f} "
            f"${t['exit_price']:>7.2f} {pnl_str:>7}  {t['regime_entry']:<12} "
            f"{t['confirmations']:>3}/8 {t['exit_reason']:<20}"
        )


def print_notes():
    print()
    print("=" * 78)
    print("  NOTES")
    print("=" * 78)
    print()
    print("  This backtest trains and trades on the same symbol (e.g. SPY).")
    print("  In production, the HMM regime would filter entries on individual")
    print("  low-float momentum stocks via the existing pullback strategy.")
    print()
    print("  Next step: if results show alpha, integrate a RegimeDetector")
    print("  into the live bot to gate new entries during bearish regimes.")
    print("=" * 78)
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="HMM Regime Detection Backtest")
    parser.add_argument(
        "--symbol", default="SPY", help="Market proxy symbol (default: SPY)"
    )
    parser.add_argument(
        "--days", type=int, default=730, help="Days of history (default: 730)"
    )
    parser.add_argument(
        "--states", type=int, default=7, help="HMM hidden states (default: 7)"
    )
    parser.add_argument(
        "--timeframe",
        choices=["daily", "hourly"],
        default="daily",
        help="Bar timeframe (default: daily)",
    )
    parser.add_argument(
        "--confirmations",
        type=int,
        default=7,
        help="Min confirmations required out of 8 (default: 7)",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=48,
        help="Hours cooldown after exit (default: 48)",
    )
    args = parser.parse_args()

    # ── Header ──
    print_header(args.symbol, args.days, args.states, args.timeframe)

    # ── Phase 1: Data ──
    df = fetch_market_data(args.symbol, args.days, args.timeframe)
    features = engineer_features(df)
    print_data_summary(df, features)

    # ── Phase 2: Train HMM ──
    model, scaler, states, posteriors, converged, n_iters, score = train_hmm(
        features, n_states=args.states
    )
    print_training_summary(converged, n_iters, score, args.states)

    # ── Phase 3: Regime summary ──
    regime_info = label_regimes(model, states, features, args.states)
    print_regime_table(regime_info, states)

    # ── Phase 4: Current regime ──
    print_current_regime(states, posteriors, regime_info, df)

    # ── Phase 5: Backtest ──
    indicators = compute_indicators(df)

    results = simulate_trades(
        df=df,
        states=states,
        posteriors=posteriors,
        regime_info=regime_info,
        indicators=indicators,
        min_confirmations=args.confirmations,
        cooldown_hours=args.cooldown,
        timeframe=args.timeframe,
    )

    print_backtest_results(results, args.confirmations, args.cooldown)

    # ── Phase 6: Trade log ──
    print_trade_log(results["trades"])

    # ── Notes ──
    print_notes()


if __name__ == "__main__":
    main()
