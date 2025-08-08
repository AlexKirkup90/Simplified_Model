# app.py
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

import backend

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(layout="wide", page_title="Hybrid Momentum Portfolio")

# ---------------------------------
# Auto-log 1-day performance on load
# (uses last saved portfolio from Gist)
# ---------------------------------
st.sidebar.header("⚙️ App Controls")
auto_log = st.sidebar.checkbox("Auto-log 1-day performance on load", value=True, help="Logs 1-day return vs QQQ using last saved portfolio (from Gist).")

auto_log_msg = ""
if auto_log:
    try:
        prev_port = backend.load_previous_portfolio()
        if prev_port is not None and not prev_port.empty:
            out = backend.record_live_snapshot(prev_port, note="auto")
            if out.get("ok"):
                auto_log_msg = f"📌 Auto-logged: Strategy {out['strat_ret']:.2%} vs QQQ {out['qqq_ret']:.2%} (rows: {out['rows']})"
            else:
                auto_log_msg = f"⚠️ Auto-log skipped: {out.get('msg','No message')}"
        else:
            auto_log_msg = "ℹ️ Auto-log skipped (no saved portfolio found)."
    except Exception as e:
        auto_log_msg = f"⚠️ Auto-log error: {e}"

if auto_log_msg:
    st.sidebar.caption(auto_log_msg)

# ---------------------------------
# Strategy selection
# ---------------------------------
st.sidebar.header("🎛️ Strategy Mode")
mode = st.sidebar.selectbox(
    "Choose a strategy",
    ["Classic 90/10 (sliders)", "ISA Dynamic (0.75)"]
)

# Classic controls
if mode == "Classic 90/10 (sliders)":
    st.sidebar.header("🧮 Classic Parameters (Momentum Sleeve)")
    momentum_window = st.sidebar.slider("Momentum Lookback (Months)", 3, 12, 6, 1)
    top_n = st.sidebar.slider("Number of Stocks in Portfolio", 5, 20, 15, 1)
    cap = st.sidebar.slider("Max Weight Cap per Stock", 0.10, 0.50, 0.25, 0.01, format="%.2f")
    tol = st.sidebar.slider("Rebalancing Tolerance", 0.005, 0.05, 0.01, 0.005, format="%.3f")
else:
    st.sidebar.header("Preset: ISA Dynamic (0.75)")
    preset = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]
    st.sidebar.write(f"Momentum: {preset['mom_lb']}m, top {preset['mom_topn']}, cap {preset['mom_cap']:.2f}")
    st.sidebar.write(f"Mean-Reversion: {preset['mr_lb']}d, top {preset['mr_topn']}, MA {preset['mr_ma']}")
    st.sidebar.write(f"Weights: {int(preset['mom_w']*100)}% / {int(preset['mr_w']*100)}%")
    st.sidebar.write(f"Trigger: {preset['trigger']:.2f}")
    tol = st.sidebar.slider("Rebalancing Tolerance (display diffs)", 0.005, 0.05, 0.01, 0.005, format="%.3f")

# ---------------------------------
# Title
# ---------------------------------
st.title("🚀 Hybrid Momentum Portfolio Manager")
if mode == "Classic 90/10 (sliders)":
    st.markdown("Classic **90/10 Hybrid**: 90% Momentum, 10% Mean-Reversion.")
else:
    st.markdown("ISA **Dynamic (0.75)** preset: lower churn, strong compounding. Rebalance only when portfolio health < 75% of top signal.")

# ---------------------------------
# Generate
# ---------------------------------
if st.button("Generate Portfolio & Rebalancing Plan", type="primary", use_container_width=True):
    with st.spinner("Crunching data…"):
        try:
            prev_portfolio = backend.load_previous_portfolio()

            if mode == "Classic 90/10 (sliders)":
                # Live portfolio (classic)
                new_portfolio_display, new_portfolio_raw = backend.generate_live_portfolio_classic(
                    momentum_window, top_n, cap
                )
                decision_note = "Classic mode (no trigger)."

                # Quick backtest (classic)
                strat_cum, qqq_cum = backend.run_backtest_for_app(momentum_window, top_n, cap)
            else:
                # Live portfolio (ISA Dynamic with trigger vs previous)
                display_df, raw_df, decision_note = backend.generate_live_portfolio_isa(
                    backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"],
                    prev_portfolio if prev_portfolio is not None and not prev_portfolio.empty else None
                )
                new_portfolio_display, new_portfolio_raw = display_df, raw_df

                # Backtest (ISA preset, trigger-free for fairness)
                strat_cum, qqq_cum = backend.run_backtest_isa_dynamic()

            # -------------------------
            # Tabs
            # -------------------------
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["📊 Rebalancing Plan", "✅ New Portfolio", "📈 Performance", "📉 Market Volatility", "📡 Regime & Live"]
            )

            # --- Tab 1: Plan ---
            with tab1:
                st.subheader("Rebalancing Plan")
                st.info(decision_note)
                if new_portfolio_raw is not None and not new_portfolio_raw.empty:
                    signals = backend.diff_portfolios(
                        prev_portfolio if prev_portfolio is not None else st.session_state.get('latest_portfolio', backend.load_previous_portfolio()),
                        new_portfolio_raw, tol
                    )
                    if not any(signals.values()):
                        st.success("✅ No major rebalancing needed!")
                    else:
                        cols = st.columns(3)
                        with cols[0]:
                            if signals['sell']:
                                st.error("🔴 Sell Completely")
                                for t in signals['sell']:
                                    st.markdown(f"- **{t}**")
                        with cols[1]:
                            if signals['buy']:
                                st.success("🟢 New Buys")
                                for t in signals['buy']:
                                    w = float(new_portfolio_raw.at[t, 'Weight'])
                                    st.markdown(f"- **{t}** (Target: {w:.2%})")
                        with cols[2]:
                            if signals['rebalance']:
                                st.info("🔄 Rebalance")
                                for t, old_w, new_w in signals['rebalance']:
                                    st.markdown(f"- **{t}**: {old_w:.2%} → **{new_w:.2%}**")
                else:
                    st.warning("No portfolio generated.")
                    
# === Explain changes (new) ===
with st.expander("🔎 What changed and why?", expanded=False):
    try:
        # Fetch a short window for the signal snapshot
        uni = backend.get_nasdaq_100_plus_tickers()
        if mode == "Classic 90/10 (sliders)":
            params_for_explain = {
                'mom_lb': momentum_window,   # months
                'mom_topn': top_n,
                'mom_cap': cap,
                'mr_lb': 21,                 # classic MR
                'mr_topn': 5,
                'mr_ma': 200,
                'mom_w': 0.90,
                'mr_w': 0.10
            }
            start_explain = (datetime.today() - relativedelta(months=momentum_window + 8)).strftime('%Y-%m-%d')
        else:
            params_for_explain = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]
            start_explain = (datetime.today() - relativedelta(months=params_for_explain['mom_lb'] + 8)).strftime('%Y-%m-%d')

        end_explain = datetime.today().strftime('%Y-%m-%d')
        px_explain = backend.fetch_market_data(uni, start_explain, end_explain)

        if px_explain.empty:
            st.info("Could not fetch prices to build signal snapshot.")
        else:
            expl = backend.explain_portfolio_changes(
                prev_portfolio if prev_portfolio is not None else st.session_state.get('latest_portfolio', backend.load_previous_portfolio()),
                new_portfolio_raw,
                px_explain,
                params_for_explain
            )
            if expl.empty:
                st.write("No material changes to explain.")
            else:
                st.dataframe(expl, use_container_width=True)
                st.caption(
                    "Notes: Momentum rank is within the current universe (1 = best). "
                    "Short-term return is the last {} trading days (more negative = larger 'dip'). "
                    "Above 200DMA indicates long-term uptrend filter for MR sleeve."
                    .format(params_for_explain['mr_lb'])
                )
    except Exception as e:
        st.warning(f"Could not build explanation: {e}")
        
            # --- Tab 2: Portfolio ---
            with tab2:
                st.subheader("New Target Portfolio")
                if new_portfolio_display is not None and not new_portfolio_display.empty:
                    st.dataframe(new_portfolio_display, use_container_width=True)
                    st.subheader("Portfolio Weights")
                    st.bar_chart(new_portfolio_raw['Weight'])
                else:
                    st.warning("No portfolio to display.")

            # --- Tab 3: Performance ---
            with tab3:
                st.subheader("Backtest vs. QQQ (Since 2018)")
                if strat_cum is not None and qqq_cum is not None:
                    # KPI table (frequency-aware)
                    strat_rets = strat_cum.pct_change().dropna()
                    qqq_rets   = qqq_cum.pct_change().dropna()
                    rows = [
                        backend.kpi_row("Strategy", strat_rets),
                        backend.kpi_row("QQQ Benchmark", qqq_rets),
                    ]
                    kpi_df = pd.DataFrame(
                        rows,
                        columns=["Model","Freq","CAGR","Sharpe","Sortino","Calmar","MaxDD","Trades/yr","Turnover/yr","Equity Multiple"]
                    )
                    st.table(kpi_df)

                    # Plot log equity
                    fig, ax = plt.subplots(figsize=(10,5))
                    ax.plot(strat_cum.index, strat_cum.values, label='Strategy')
                    ax.plot(qqq_cum.index, qqq_cum.values, label='QQQ Benchmark', linestyle='--')
                    ax.set_ylabel("Cumulative Growth")
                    ax.set_yscale('log')
                    ax.legend()
                    ax.grid(True, which="both", ls="--")
                    st.pyplot(fig)
                else:
                    st.warning("Could not generate backtest.")

            # --- Tab 4: Volatility ---
            with tab4:
                st.subheader("QQQ 10-Day Rolling Volatility")
                try:
                    qqq_vol_data = yf.download("QQQ", period="60d", auto_adjust=True, progress=False)
                    qqq_close = qqq_vol_data['Close']
                    vol_series = qqq_close.pct_change().rolling(window=10).std().dropna()
                    current_vol = vol_series.iloc[-1]
                    if current_vol > 0.025:
                        st.warning(f"High market volatility detected: {current_vol:.2%}")
                    else:
                        st.success(f"Market volatility is normal: {current_vol:.2%}")
                    fig, ax = plt.subplots(figsize=(10,5))
                    ax.plot(vol_series.index, vol_series.values, label='10-Day Volatility')
                    ax.axhline(y=0.025, color='r', linestyle='--', label='High Volatility Threshold (2.5%)')
                    ax.legend()
                    ax.grid(True, which="both", ls="--")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Could not fetch volatility data: {e}")

            # --- Tab 5: Regime & Live ---
            with tab5:
                st.subheader("Market Regime Snapshot")
                try:
                    uni = backend.get_nasdaq_100_plus_tickers()
                    start_date = (datetime.today() - relativedelta(days=400)).strftime('%Y-%m-%d')
                    end_date = datetime.today().strftime('%Y-%m-%d')
                    daily = backend.fetch_market_data(uni, start_date, end_date)

                    metrics = backend.compute_regime_metrics(daily)
                    if metrics:
                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("% Universe >200DMA", f"{metrics['universe_above_200dma']*100:.0f}%")
                        c2.metric("QQQ >200DMA", "Yes" if metrics['qqq_above_200dma']>=1 else "No")
                        c3.metric("QQQ 10d Vol", f"{metrics['qqq_vol_10d']:.2%}")
                        c4.metric("Breadth +6m", f"{metrics['breadth_pos_6m']*100:.0f}%")
                        slope = metrics.get('qqq_50dma_slope_10d', np.nan)
                        c5.metric("QQQ 50DMA slope (10d)", f"{slope:.2%}" if pd.notna(slope) else "—")
                    else:
                        st.warning("Not enough data for regime snapshot.")
                except Exception as e:
                    st.error(f"Regime snapshot failed: {e}")

                st.divider()
                st.subheader("Live Paper Tracking")
                if 'latest_portfolio' in st.session_state and not st.session_state.latest_portfolio.empty:
                    if st.button("📌 Record live snapshot now", use_container_width=False):
                        out = backend.record_live_snapshot(st.session_state.latest_portfolio, note=mode)
                        if out.get("ok"):
                            st.success(f"Logged 1-day: Strategy {out['strat_ret']:.2%} vs QQQ {out['qqq_ret']:.2%}  • total rows: {out['rows']}")
                        else:
                            st.warning(out.get("msg","Could not record snapshot."))

                live_eq = backend.get_live_equity()
                if not live_eq.empty:
                    st.line_chart(live_eq.set_index('date')[['strat_eq','qqq_eq']])
                    st.dataframe(live_eq.tail(10), use_container_width=True)
                else:
                    st.info("No live snapshots yet. Click the button above to start logging.")

            # Keep raw for save button
            if new_portfolio_raw is not None and not new_portfolio_raw.empty:
                st.session_state.latest_portfolio = new_portfolio_raw

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# --- Save button ---
if 'latest_portfolio' in st.session_state and not st.session_state.latest_portfolio.empty:
    st.sidebar.header("💾 Save Portfolio")
    if st.sidebar.button("Save this portfolio for next month"):
        backend.save_portfolio_to_gist(st.session_state.latest_portfolio)
