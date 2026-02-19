#!/usr/bin/env python3
"""
HMM Regime Terminal — Streamlit Dashboard.

Interactive regime detection and backtest visualization.
Reuses core logic from scripts/backtest_hmm.py.

Run:
    streamlit run scripts/regime_terminal.py
"""

import sys
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, ".")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from scripts.backtest_hmm import (
    check_confirmations,
    compute_indicators,
    engineer_features,
    fetch_market_data,
    label_regimes,
    simulate_trades,
    train_hmm,
)

# ─── Page config ─────────────────────────────────────────────────────────

st.set_page_config(page_title="HMM Regime Terminal", layout="wide")

# ─── Color maps ──────────────────────────────────────────────────────────

REGIME_COLORS = {
    "BULL RUN": "#22c55e",
    "BULL MILD": "#86efac",
    "CRASH": "#dc2626",
    "BEAR": "#f87171",
}
CATEGORY_COLORS = {"bullish": "#22c55e", "bearish": "#ef4444", "neutral": "#fbbf24"}
CATEGORY_BG = {
    "bullish": "rgba(34,197,94,0.12)",
    "bearish": "rgba(239,68,68,0.12)",
    "neutral": "rgba(251,191,36,0.08)",
}


def _regime_color(label: str, category: str) -> str:
    return REGIME_COLORS.get(label, CATEGORY_COLORS.get(category, "#a3a3a3"))


# ─── Sidebar ─────────────────────────────────────────────────────────────

st.sidebar.title("HMM Regime Terminal")
st.sidebar.markdown("---")

ticker = st.sidebar.text_input("Ticker(s)", value="SPY", help="Comma-separated for multiple")
col_start, col_end = st.sidebar.columns(2)
with col_start:
    start_date = st.date_input("Start", value=pd.Timestamp.now() - pd.DateOffset(years=2))
with col_end:
    end_date = st.date_input("End", value=pd.Timestamp.now())

timeframe = st.sidebar.selectbox("Timeframe", ["daily", "hourly"], index=0)
n_states = st.sidebar.slider("HMM States", min_value=3, max_value=10, value=7)
min_conf = st.sidebar.slider("Confirmations Required", min_value=4, max_value=8, value=7)
cooldown = st.sidebar.number_input("Cooldown Hours", min_value=0, max_value=168, value=48)

run_btn = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)

# ─── Main area ───────────────────────────────────────────────────────────

if not run_btn and "results" not in st.session_state:
    st.title("HMM Regime Terminal")
    st.info("Configure parameters in the sidebar and click **Run Analysis** to begin.")
    st.stop()


def run_analysis(symbol: str):
    """Run the full HMM pipeline for one symbol and return all artifacts."""
    days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
    if days < 30:
        st.error(f"Date range too short ({days} days). Need at least 30.")
        return None

    # Phase 1: Data
    with st.spinner(f"Fetching {symbol} data..."):
        df = fetch_market_data(symbol, days, timeframe)

    # Phase 2: Features + HMM
    with st.spinner("Training HMM..."):
        features = engineer_features(df)
        model, scaler, states, posteriors, converged, n_iters, score = train_hmm(
            features, n_states=n_states
        )

    # Phase 3: Regimes
    regime_info = label_regimes(model, states, features, n_states)

    # Phase 4: Indicators + Backtest
    with st.spinner("Running backtest..."):
        indicators = compute_indicators(df)
        results = simulate_trades(
            df=df,
            states=states,
            posteriors=posteriors,
            regime_info=regime_info,
            indicators=indicators,
            min_confirmations=min_conf,
            cooldown_hours=cooldown,
            timeframe=timeframe,
        )

    # Current bar confirmations
    offset = len(df) - len(states)
    last_state_idx = len(states) - 1
    last_bar_idx = last_state_idx + offset
    last_prob = float(posteriors[last_state_idx][states[last_state_idx]])
    conf_count, conf_details = check_confirmations(last_bar_idx, df, indicators, last_prob)

    return {
        "df": df,
        "features": features,
        "states": states,
        "posteriors": posteriors,
        "regime_info": regime_info,
        "indicators": indicators,
        "results": results,
        "converged": converged,
        "n_iters": n_iters,
        "score": score,
        "conf_count": conf_count,
        "conf_details": conf_details,
        "offset": offset,
    }


# Run analysis on button click
if run_btn:
    symbols = [s.strip().upper() for s in ticker.split(",") if s.strip()]
    if not symbols:
        st.error("Enter at least one ticker.")
        st.stop()

    all_results = {}
    for sym in symbols:
        try:
            result = run_analysis(sym)
            if result:
                all_results[sym] = result
        except Exception as e:
            st.error(f"Error analyzing {sym}: {e}")

    if all_results:
        st.session_state["results"] = all_results
        st.session_state["symbols"] = list(all_results.keys())

if "results" not in st.session_state:
    st.stop()

all_results = st.session_state["results"]
symbols = st.session_state["symbols"]

# ─── Tabs for multiple symbols ───────────────────────────────────────────

tabs = st.tabs(symbols) if len(symbols) > 1 else [st.container()]

for tab, sym in zip(tabs, symbols):
    with tab:
        data = all_results[sym]
        df = data["df"]
        states = data["states"]
        posteriors = data["posteriors"]
        regime_info = data["regime_info"]
        results = data["results"]
        offset = data["offset"]

        # ─── Row 1: Status Cards ─────────────────────────────────
        st.markdown(f"### {sym} — Regime Analysis")

        c1, c2, c3 = st.columns(3)

        # Current regime
        current_state = states[-1]
        current_label = regime_info["labels"].get(current_state, "UNKNOWN")
        current_cat = regime_info["categories"].get(current_state, "neutral")
        current_prob = float(posteriors[-1][current_state])
        color = CATEGORY_COLORS.get(current_cat, "#a3a3a3")

        with c1:
            st.markdown(
                f"""<div style="background:{CATEGORY_BG.get(current_cat,'#1a1a2e')};
                border-left:4px solid {color}; padding:16px; border-radius:8px;">
                <div style="color:#888; font-size:0.8em;">CURRENT REGIME</div>
                <div style="color:{color}; font-size:1.6em; font-weight:700;">
                {current_label}</div>
                <div style="color:#ccc;">Confidence: {current_prob:.0%} &nbsp;|&nbsp;
                {current_cat.upper()}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        # Current signal
        with c2:
            trades = results["trades"]
            # Determine signal from last backtest state
            if results["trades"] and results["trades"][-1]["exit_reason"] == "End of data":
                signal = "LONG HOLDING"
                sig_color = "#22c55e"
            else:
                signal = "CASH"
                sig_color = "#a3a3a3"

            st.markdown(
                f"""<div style="background:rgba(30,30,50,0.5);
                border-left:4px solid {sig_color}; padding:16px; border-radius:8px;">
                <div style="color:#888; font-size:0.8em;">CURRENT SIGNAL</div>
                <div style="color:{sig_color}; font-size:1.6em; font-weight:700;">
                {signal}</div>
                <div style="color:#ccc;">Trades: {results['total_trades']}
                &nbsp;|&nbsp; Win rate: {results['win_rate']:.0%}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        # Confirmations
        with c3:
            conf_count = data["conf_count"]
            conf_details = data["conf_details"]
            passed = sum(1 for v in conf_details.values() if v)
            conf_color = "#22c55e" if passed >= min_conf else "#ef4444"

            checks_html = ""
            for name, ok in conf_details.items():
                icon = "&#10003;" if ok else "&#10007;"
                ic = "#22c55e" if ok else "#ef4444"
                checks_html += f'<span style="color:{ic}; margin-right:8px;">{icon} {name}</span>'

            st.markdown(
                f"""<div style="background:rgba(30,30,50,0.5);
                border-left:4px solid {conf_color}; padding:16px; border-radius:8px;">
                <div style="color:#888; font-size:0.8em;">CONFIRMATIONS</div>
                <div style="color:{conf_color}; font-size:1.6em; font-weight:700;">
                {passed}/8</div>
                <div style="color:#ccc; font-size:0.75em; line-height:1.8;">
                {checks_html}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("")

        # ─── Row 2: Price Chart with Regime Overlay ──────────────
        st.subheader("Price & Regime Overlay")

        fig_price = go.Figure()

        # Background regime shading
        regime_dates = df.index[offset:]
        prev_cat = None
        span_start = None

        for i, idx in enumerate(regime_dates):
            state_i = states[i] if i < len(states) else states[-1]
            cat = regime_info["categories"].get(state_i, "neutral")

            if cat != prev_cat:
                # Close previous span
                if prev_cat is not None and span_start is not None:
                    fig_price.add_vrect(
                        x0=span_start,
                        x1=regime_dates[i - 1],
                        fillcolor=CATEGORY_BG.get(prev_cat, "rgba(0,0,0,0)"),
                        layer="below",
                        line_width=0,
                    )
                span_start = idx
                prev_cat = cat

        # Close final span
        if prev_cat is not None and span_start is not None:
            fig_price.add_vrect(
                x0=span_start,
                x1=regime_dates[-1],
                fillcolor=CATEGORY_BG.get(prev_cat, "rgba(0,0,0,0)"),
                layer="below",
                line_width=0,
            )

        # Price line
        fig_price.add_trace(
            go.Scatter(
                x=df.index,
                y=df["close"],
                mode="lines",
                name="Close",
                line=dict(color="#60a5fa", width=1.5),
            )
        )

        # Entry/exit markers
        for t in results["trades"]:
            fig_price.add_trace(
                go.Scatter(
                    x=[t["entry_date"]],
                    y=[t["entry_price"]],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=10, color="#22c55e"),
                    name="Entry",
                    showlegend=False,
                    hovertext=f"Entry: ${t['entry_price']:.2f}<br>Regime: {t['regime_entry']}",
                )
            )
            fig_price.add_trace(
                go.Scatter(
                    x=[t["exit_date"]],
                    y=[t["exit_price"]],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=10,
                        color="#ef4444" if t["pnl_pct"] < 0 else "#22c55e",
                    ),
                    name="Exit",
                    showlegend=False,
                    hovertext=f"Exit: ${t['exit_price']:.2f}<br>P&L: {t['pnl_pct']:+.1%}<br>{t['exit_reason']}",
                )
            )

        fig_price.update_layout(
            template="plotly_dark",
            height=450,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="",
            yaxis_title="Price ($)",
            legend=dict(orientation="h", y=1.02),
            xaxis_rangeslider_visible=False,
        )

        st.plotly_chart(fig_price, use_container_width=True)

        # ─── Row 3: Backtest Performance ─────────────────────────
        st.subheader("Backtest Performance")
        perf_left, perf_right = st.columns([1, 2])

        with perf_left:
            pf = results["profit_factor"]
            pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"

            metrics = {
                "Total Return": f"{results['total_return']:+.1%}",
                "Buy & Hold": f"{results['buy_hold_return']:+.1%}",
                "Alpha": f"{results['alpha']:+.1%}",
                "Total Trades": str(results["total_trades"]),
                "Win Rate": f"{results['win_rate']:.0%}",
                "Avg Win": f"{results['avg_win']:+.2%}",
                "Avg Loss": f"{results['avg_loss']:+.2%}",
                "Profit Factor": pf_str,
                "Max Drawdown": f"{results['max_drawdown']:.1%}",
                "Final Equity": f"${results['final_equity']:,.2f}",
            }

            metrics_df = pd.DataFrame(
                list(metrics.items()), columns=["Metric", "Value"]
            )
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)

        with perf_right:
            eq_curve = results["equity_curve"]
            start_idx = max(offset, 60)
            eq_dates = df.index[start_idx : start_idx + len(eq_curve)]

            # Buy-and-hold curve
            bh_start = float(df["close"].iloc[start_idx])
            bh_equity = [
                results["starting_equity"] * float(df["close"].iloc[start_idx + i]) / bh_start
                for i in range(len(eq_curve))
                if start_idx + i < len(df)
            ]
            bh_dates = eq_dates[: len(bh_equity)]

            fig_eq = go.Figure()
            fig_eq.add_trace(
                go.Scatter(
                    x=eq_dates,
                    y=eq_curve,
                    mode="lines",
                    name="HMM Strategy",
                    line=dict(color="#22c55e", width=2),
                )
            )
            fig_eq.add_trace(
                go.Scatter(
                    x=bh_dates,
                    y=bh_equity,
                    mode="lines",
                    name="Buy & Hold",
                    line=dict(color="#60a5fa", width=1.5, dash="dash"),
                )
            )
            fig_eq.update_layout(
                template="plotly_dark",
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis_title="Equity ($)",
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        # ─── Row 4: Regime Details ───────────────────────────────
        st.subheader("Regime Details")
        reg_left, reg_right = st.columns(2)

        with reg_left:
            rows = []
            for state_id in regime_info["sorted_states"]:
                s = regime_info["stats"][state_id]
                rows.append(
                    {
                        "State": state_id,
                        "Label": regime_info["labels"][state_id],
                        "Category": regime_info["categories"][state_id],
                        "Mean Return": f"{s['mean_return']:+.4f}",
                        "Volatility": f"{s['std_return']:.4f}",
                        "Frequency": f"{s['frequency']:.1%}",
                        "Bars": s["count"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            model_info = "Converged" if data["converged"] else "Did NOT converge"
            st.caption(
                f"Model: {n_states} states, {model_info} in "
                f"{data['n_iters']} iters, log-likelihood {data['score']:.1f}"
            )

        with reg_right:
            # Regime timeline
            fig_timeline = go.Figure()

            regime_labels_arr = [
                regime_info["labels"].get(states[i], "?") for i in range(len(states))
            ]
            regime_cats_arr = [
                regime_info["categories"].get(states[i], "neutral")
                for i in range(len(states))
            ]
            colors_arr = [CATEGORY_COLORS.get(c, "#a3a3a3") for c in regime_cats_arr]

            fig_timeline.add_trace(
                go.Scatter(
                    x=regime_dates,
                    y=regime_labels_arr,
                    mode="markers",
                    marker=dict(color=colors_arr, size=4),
                    hovertext=[
                        f"{regime_labels_arr[i]} ({regime_cats_arr[i]})"
                        for i in range(len(regime_labels_arr))
                    ],
                )
            )
            fig_timeline.update_layout(
                template="plotly_dark",
                height=300,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis_title="Regime",
                showlegend=False,
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

        # ─── Row 5: Trade Log ────────────────────────────────────
        st.subheader("Trade Log")

        if results["trades"]:
            trade_rows = []
            for i, t in enumerate(results["trades"], 1):
                entry_d = (
                    t["entry_date"].strftime("%Y-%m-%d")
                    if hasattr(t["entry_date"], "strftime")
                    else str(t["entry_date"])[:10]
                )
                exit_d = (
                    t["exit_date"].strftime("%Y-%m-%d")
                    if hasattr(t["exit_date"], "strftime")
                    else str(t["exit_date"])[:10]
                )
                trade_rows.append(
                    {
                        "#": i,
                        "Entry Date": entry_d,
                        "Entry $": f"{t['entry_price']:.2f}",
                        "Exit Date": exit_d,
                        "Exit $": f"{t['exit_price']:.2f}",
                        "P&L": f"{t['pnl_pct']:+.1%}",
                        "P&L $": f"${t['pnl_dollar']:+,.2f}",
                        "Regime": t["regime_entry"],
                        "Conf": f"{t['confirmations']}/8",
                        "Exit Reason": t["exit_reason"],
                    }
                )
            st.dataframe(pd.DataFrame(trade_rows), hide_index=True, use_container_width=True)
        else:
            st.info("No trades were taken with current parameters.")
