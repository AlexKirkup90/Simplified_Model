# app.py
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

import backend

st.set_page_config(layout="wide", page_title="Hybrid Momentum Portfolio")

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("⚙️ App Controls")
auto_log = st.sidebar.checkbox(
    "Auto-log 1-day performance on load",
    value=True,
    help="Logs 1-day return vs QQQ using last saved portfolio (from Gist or local CSV)."
)

st.sidebar.header("🎛️ Strategy Mode")
mode = st.sidebar.selectbox(
    "Choose a strategy",
    ["Classic 90/10 (sliders)", "ISA Dynamic (Monthly Lock)"]
)

st.sidebar.header("💸 Costs & Liquidity")
apply_costs = st.sidebar.checkbox("Show net of costs in backtest", value=True)
roundtrip_bps = st.sidebar.number_input(
    "Round-trip trading cost (bps)", min_value=0, max_value=100, value=10, step=1
)
min_dollar_volume = st.sidebar.number_input(
    "Min median $ volume (60d)",
    min_value=0, value=10_000_000, step=1_000_000,
    help="Filters illiquid names using median (Close×Volume) over ~60 trading days."
)

if mode == "Classic 90/10 (sliders)":
    st.sidebar.header("🧮 Classic Parameters (Momentum Sleeve)")
    momentum_window = st.sidebar.slider("Momentum Lookback (Months)", 3, 12, 6, 1)
    top_n = st.sidebar.slider("Number of Stocks in Portfolio", 5, 20, 15, 1)
    cap = st.sidebar.slider("Max Weight Cap per Stock", 0.10, 0.50, 0.25, 0.01, format="%.2f")
    tol = st.sidebar.slider("Rebalancing Tolerance", 0.005, 0.05, 0.01, 0.005, format="%.3f")
else:
    st.sidebar.header("Preset: ISA Dynamic (Monthly Lock)")
    preset = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]
    st.sidebar.write(f"Momentum: blended 3/6/12m, top {preset['mom_topn']}, cap {preset['mom_cap']:.2f}")
    st.sidebar.write(f"Mean-Reversion: {preset['mr_lb']}d dip, top {preset['mr_topn']}, MA {preset['mr_ma']}")
    st.sidebar.write(f"Weights: {int(preset['mom_w']*100)}% / {int(preset['mr_w']*100)}%")
    st.sidebar.write(f"Trigger: {preset['trigger']:.2f}")
    st.sidebar.write(f"Stability filter: {preset['stability_days']} days")
    tol = st.sidebar.slider("Rebalancing Tolerance (display diffs)", 0.005, 0.05, 0.01, 0.005, format="%.3f")

# =========================
# Title
# =========================
st.title("🚀 Hybrid Momentum Portfolio Manager")
if mode == "Classic 90/10 (sliders)":
    st.markdown("Classic **90/10 Hybrid**: blended momentum (3/6/12m) + MR, with liquidity filter and optional trading costs.")
else:
    st.markdown("**ISA Dynamic (Monthly Lock)**: stability-gated momentum, MR sleeve, 0.75 trigger. No mid-month rebalances.")

# =========================
# Auto-log (1d)
# =========================
if auto_log:
    try:
        prev_port = backend.load_previous_portfolio()
        if prev_port is not None and not prev_port.empty:
            out = backend.record_live_snapshot(prev_port, note="auto")
            if out.get("ok"):
                st.sidebar.caption(
                    f"📌 Auto-logged: Strategy {out['strat_ret']:.2%} vs QQQ {out['qqq_ret']:.2%} (rows: {out['rows']})"
                )
            else:
                st.sidebar.caption(f"⚠️ Auto-log skipped: {out.get('msg','No message')}")
        else:
            st.sidebar.caption("ℹ️ Auto-log skipped (no saved portfolio found).")
    except Exception as e:
        st.sidebar.caption(f"⚠️ Auto-log error: {e}")

# =========================
# Main Action
# =========================
if st.button("Generate Portfolio & Rebalancing Plan", type="primary", use_container_width=True):
    with st.spinner("Crunching data…"):
        try:
            prev_portfolio = backend.load_previous_portfolio()

            # --- Build live portfolio & run backtest ---
            if mode == "Classic 90/10 (sliders)":
                new_display, new_raw = backend.generate_live_portfolio_classic(
                    momentum_window, top_n, cap,
                    min_dollar_volume=min_dollar_volume
                )
                decision_note = "Classic mode (no monthly lock)."

                strat_cum_gross, strat_cum_net, qqq_cum, hybrid_tno = backend.run_backtest_for_app(
                    momentum_window, top_n, cap,
                    roundtrip_bps=roundtrip_bps,
                    min_dollar_volume=min_dollar_volume,
                    show_net=apply_costs
                )
            else:
                new_display, new_raw, decision_note = backend.generate_live_portfolio_isa_monthly(
                    backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"],
                    prev_portfolio if (prev_portfolio is not None and not prev_portfolio.empty) else None,
                    min_dollar_volume=min_dollar_volume
                )
                strat_cum_gross, strat_cum_net, qqq_cum, hybrid_tno = backend.run_backtest_isa_dynamic(
                    roundtrip_bps=roundtrip_bps,
                    min_dollar_volume=min_dollar_volume,
                    show_net=apply_costs
                )

            if new_raw is not None and not new_raw.empty:
                backend.save_current_portfolio(new_raw)
                st.session_state.latest_portfolio = new_raw

            # --- Tabs ---
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
                ["📊 Rebalancing Plan",
                 "✅ New Portfolio",
                 "📈 Performance",
                 "💸 Costs & Liquidity",
                 "📉 Market Volatility",
                 "📡 Regime & Live"]
            )

            # ----------------
            # Tab 1: Plan
            # ----------------
            with tab1:
                st.subheader("Rebalancing Plan")
                st.info(decision_note)

                if new_raw is not None and not new_raw.empty:
                    signals = backend.diff_portfolios(
                        prev_portfolio if prev_portfolio is not None else st.session_state.get('latest_portfolio', backend.load_previous_portfolio()),
                        new_raw, tol
                    )
                    if not any(signals.values()):
                        st.success("✅ No major rebalancing needed!")
                    else:
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            if signals['sell']:
                                st.error("🔴 Sell Completely")
                                for t in signals['sell']:
                                    st.markdown(f"- **{t}**")
                        with c2:
                            if signals['buy']:
                                st.success("🟢 New Buys")
                                for t in signals['buy']:
                                    w = float(new_raw.at[t, 'Weight'])
                                    st.markdown(f"- **{t}** (Target: {w:.2%})")
                        with c3:
                            if signals['rebalance']:
                                st.info("🔄 Rebalance")
                                for t, old_w, new_w in signals['rebalance']:
                                    st.markdown(f"- **{t}**: {old_w:.2%} → **{new_w:.2%}**")

                # Explainability
                try:
                    with st.expander("🔎 What changed and why?", expanded=False):
                        uni = backend.get_nasdaq_100_plus_tickers()
                        params_for_explain = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"] if mode != "Classic 90/10 (sliders)" else {
                            'mr_lb': 21, 'mr_ma': 200
                        }
                        start_explain = (datetime.today() - relativedelta(months=14)).strftime('%Y-%m-%d')
                        end_explain = datetime.today().strftime('%Y-%m-%d')
                        px_explain = backend.fetch_market_data(uni, start_explain, end_explain)
                        if px_explain.empty:
                            st.info("Could not fetch prices to build signal snapshot.")
                        else:
                            expl = backend.explain_portfolio_changes(
                                prev_portfolio if prev_portfolio is not None else st.session_state.get('latest_portfolio', backend.load_previous_portfolio()),
                                new_raw,
                                px_explain,
                                params_for_explain
                            )
                            if expl.empty:
                                st.write("No material changes to explain.")
                            else:
                                st.dataframe(expl, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not build explanation: {e}")

            # ----------------
            # Tab 2: Portfolio
            # ----------------
            with tab2:
                st.subheader("New Target Portfolio")
                if new_display is not None and not new_display.empty:
                    st.dataframe(new_display, use_container_width=True)
                    st.subheader("Portfolio Weights")
                    st.bar_chart(new_raw['Weight'])
                else:
                    st.warning("No portfolio to display.")

            # ----------------
            # Tab 3: Performance
            # ----------------
            with tab3:
                st.subheader("Backtest vs. QQQ (Since 2018)")
                if strat_cum_gross is not None and qqq_cum is not None:
                    rows = []
                    strat_g = strat_cum_gross.pct_change().dropna()
                    rows.append(backend.kpi_row("Strategy (Gross)", strat_g))
                    if apply_costs and (strat_cum_net is not None):
                        strat_n = strat_cum_net.pct_change().dropna()
                        rows.append(backend.kpi_row("Strategy (Net)", strat_n))
                    qqq_r = qqq_cum.pct_change().dropna()
                    rows.append(backend.kpi_row("QQQ Benchmark", qqq_r))
                    kpi_df = pd.DataFrame(
                        rows,
                        columns=["Model","Freq","CAGR","Sharpe","Sortino","Calmar","MaxDD","Trades/yr","Turnover/yr","Equity Multiple"]
                    )
                    st.table(kpi_df)

                    fig, ax = plt.subplots(figsize=(10,5))
                    ax.plot(strat_cum_gross.index, strat_cum_gross.values, label='Strategy (Gross)')
                    if apply_costs and strat_cum_net is not None:
                        ax.plot(strat_cum_net.index, strat_cum_net.values, label='Strategy (Net)', linestyle='-.')
                    ax.plot(qqq_cum.index, qqq_cum.values, label='QQQ Benchmark', linestyle='--')
                    ax.set_ylabel("Cumulative Growth")
                    ax.set_yscale('log')
                    ax.legend()
                    ax.grid(True, which="both", ls="--")
                    st.pyplot(fig)

                    if hybrid_tno is not None:
                        st.caption("Estimated monthly turnover (fraction of portfolio traded):")
                        st.line_chart(hybrid_tno.rename("Turnover"))
                else:
                    st.warning("Could not generate backtest.")

            # ----------------
            # Tab 4: Costs & Liquidity
            # ----------------
            with tab4:
                st.subheader("Costs & Liquidity Reality Check")
                st.caption("Estimates based on median $ volume and your round-trip cost setting. Not investment advice.")

                notional = st.number_input(
                    "Assumed portfolio notional (USD)", min_value=10_000, value=100_000, step=10_000
                )

                try:
                    uni = backend.get_nasdaq_100_plus_tickers()
                    start = (datetime.today() - relativedelta(months=14)).strftime('%Y-%m-%d')
                    end   = datetime.today().strftime('%Y-%m-%d')
                    close, vol = backend.fetch_price_volume(uni, start, end)
                    if not close.empty and not vol.empty:
                        available = [t for t in uni if t in close.columns]
                        keep = backend.filter_by_liquidity(close[available], vol[available], min_dollar_volume) if (min_dollar_volume > 0 and available) else available
                        st.write(f"**Liquidity filter:** {len(keep)}/{len(close.columns)} tickers pass (≥ ${min_dollar_volume:,.0f} median $ volume).")
                    else:
                        st.info("Couldn’t fetch price/volume to compute liquidity.")
                except Exception as e:
                    st.warning(f"Liquidity check failed: {e}")

                if (new_raw is not None and not new_raw.empty) and (not close.empty and not vol.empty):
                    try:
                        med_dollar = backend.median_dollar_volume(
                            close[new_raw.index.intersection(close.columns)],
                            vol[new_raw.index.intersection(vol.columns)],
                            window=60
                        ).rename("Median_$Vol(60d)")
                        df = pd.DataFrame(med_dollar)
                        df["Weight"] = new_raw['Weight'].reindex(df.index)
                        df["$Position"] = df["Weight"] * notional
                        df["Participation(%)"] = np.where(df["Median_$Vol(60d)"]>0,
                                                          (df["$Position"] / df["Median_$Vol(60d)"])*100, np.nan)
                        st.dataframe(df.sort_values("Weight", ascending=False), use_container_width=True)
                        st.caption("Participation(%) = $Position / Median Dollar Volume. Lower is better for execution.")
                    except Exception as e:
                        st.warning(f"Couldn’t compute per-ticker liquidity table: {e}")
                else:
                    st.info("Generate a portfolio first to see per-ticker liquidity.")

            # ----------------
            # Tab 5: Volatility
            # ----------------
            with tab5:
                st.subheader("QQQ 10-Day Rolling Volatility")
                try:
                    qqq_vol_data = yf.download("QQQ", period="60d", auto_adjust=True, progress=False)
                    qqq_close = qqq_vol_data['Close']
                    vol_series = qqq_close.pct_change().rolling(window=10).std().dropna()
                    current_vol = vol_series.iloc[-1]
