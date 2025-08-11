# app.py
import traceback
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import backend  # all logic lives here

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(layout="wide", page_title="Hybrid Momentum Portfolio (ISA-Dynamic)")

st.title("🚀 Hybrid Momentum (ISA Dynamic) — Monthly Execution")
st.caption("Composite momentum + mean reversion, with stickiness, sector caps, and monthly lock.")

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("⚙️ Execution Settings")

# Universe choice
universe_choice = st.sidebar.selectbox(
    "Universe",
    options=["Hybrid Top150", "NASDAQ100+", "S&P500 (All)"],
    index=["Hybrid Top150", "NASDAQ100+", "S&P500 (All)"].index(
        st.session_state.get("universe", "Hybrid Top150")
    )
)
st.session_state["universe"] = universe_choice

# ISA preset defaults
preset = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]

# Stickiness & sector cap (overrides)
stickiness_days = st.sidebar.slider("Stickiness (days in top cohort)", 3, 15, preset.get("stability_days", 7), 1)
sector_cap = st.sidebar.slider(
    "Sector Cap (max % per sector)",
    0.10, 0.50, 
    preset.get("sector_cap", 0.30), 
    0.05, 
    format="%.0f%%"
)

st.session_state["stickiness_days"] = stickiness_days
st.session_state["sector_cap"] = sector_cap

# Trading cost & liquidity
roundtrip_bps = st.sidebar.slider("Round-trip cost (bps)", 0, 100, backend.ROUNDTRIP_BPS_DEFAULT, 5)
min_dollar_volume = st.sidebar.number_input("Min 60d median $ volume (optional)", min_value=0, value=0, step=100000)

# Net toggle
show_net = st.sidebar.checkbox("Show net of costs", value=True)

# Rebalance tolerance for plan
tol = st.sidebar.slider("Rebalance tolerance (abs Δ weight)", 0.005, 0.05, 0.01, 0.005, format="%.3f")

# Prev portfolio (for plan & monthly lock)
prev_portfolio = backend.load_previous_portfolio()

# ---------------------------
# Go button
# ---------------------------
go = st.button("Generate Portfolio & Backtest", type="primary", use_container_width=True)

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Rebalancing Plan", "✅ Current Portfolio", "📈 Performance", "🧭 Regime", "🔎 Changes"]
)

# Placeholders to reuse below
live_disp = None
live_raw = None
decision = "—"

strategy_cum_gross = None
strategy_cum_net = None
qqq_cum = None
hybrid_tno = None

# ===========================
# Generate
# ===========================
if go:
    with st.spinner("Building monthly-locked portfolio and running backtest…"):
        try:
            # ---- Live portfolio (monthly lock + stability + trigger + sector caps)
            live_disp, live_raw, decision = backend.generate_live_portfolio_isa_monthly(
                preset=preset,
                prev_portfolio=prev_portfolio,
                min_dollar_volume=min_dollar_volume
            )
        except Exception as e:
            st.error(f"Portfolio generation failed: {e}")
            st.code(traceback.format_exc())

        try:
            # ---- ISA Dynamic backtest (Hybrid150 default) from 2017-07-01
            strategy_cum_gross, strategy_cum_net, qqq_cum, hybrid_tno = backend.run_backtest_isa_dynamic(
                roundtrip_bps=roundtrip_bps,
                min_dollar_volume=min_dollar_volume,
                show_net=show_net,
                start_date="2017-07-01",
                universe_choice=universe_choice,
                top_n=preset["mom_topn"],
                name_cap=preset["mom_cap"],
                sector_cap=sector_cap,
                stickiness_days=stickiness_days,
                mr_topn=preset["mr_topn"],
                mom_weight=preset["mom_w"],
                mr_weight=preset["mr_w"],
            )
        except Exception as e:
            st.warning(f"Backtest failed: {e}")
            st.code(traceback.format_exc())

        # save to session for Save button / persistence
        if live_raw is not None and not live_raw.empty:
            st.session_state.latest_portfolio = live_raw.copy()

# ---------------------------
# Tab 1: Rebalancing Plan
# ---------------------------
with tab1:
    st.subheader("📊 Rebalancing Plan (vs last saved)")
    if live_raw is None or live_raw.empty:
        st.info("Click Generate to produce a live portfolio.")
    else:
        signals = backend.diff_portfolios(prev_portfolio, live_raw, tol)
        if not any([signals["sell"], signals["buy"], signals["rebalance"]]):
            st.success("✅ No major rebalancing needed!")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("### 🔴 Sell Completely")
                if signals["sell"]:
                    for t in signals["sell"]:
                        st.write(f"- **{t}**")
                else:
                    st.write("None")
            with c2:
                st.markdown("### 🟢 New Buys")
                if signals["buy"]:
                    for t in signals["buy"]:
                        tgt = float(live_raw.loc[t, "Weight"]) if t in live_raw.index else 0.0
                        st.write(f"- **{t}** — target {tgt:.2%}")
                else:
                    st.write("None")
            with c3:
                st.markdown("### 🔄 Rebalance (≥ {tol:.1%})")
                if signals["rebalance"]:
                    for t, old_w, new_w in signals["rebalance"]:
                        st.write(f"- **{t}**: {old_w:.2%} → **{new_w:.2%}**")
                else:
                    st.write("None")

# ---------------------------
# Tab 2: Current Portfolio
# ---------------------------
with tab2:
    st.subheader("✅ Current Portfolio (Monthly-Locked)")
    if live_disp is None or live_disp.empty:
        st.warning(decision if isinstance(decision, str) else "—")
    else:
        st.caption(decision)
        st.dataframe(live_disp, use_container_width=True)

        # Bar chart
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            live_sorted = live_raw.sort_values("Weight", ascending=False)
            ax.bar(live_sorted.index, live_sorted["Weight"].values)
            ax.set_ylabel("Weight")
            ax.set_xticklabels(live_sorted.index, rotation=45, ha="right")
            st.pyplot(fig)
        except Exception:
            pass

        # Save
        if st.button("💾 Save this portfolio for next month"):
            backend.save_portfolio_to_gist(live_raw)
            st.success("Saved to Gist.")

# ---------------------------
# Tab 3: Performance
# ---------------------------
with tab3:
    st.subheader("📈 Backtest (since 2017-07-01)")
    if strategy_cum_gross is None or qqq_cum is None:
        st.info("Click Generate to see backtest results.")
    else:
        st.markdown("#### Key Performance (monthly series inferred)")

        # Build KPI rows (use backend.kpi_row)
        rows = []
        rows.append(
            backend.kpi_row(
                "Strategy (Gross)",
                strategy_cum_gross.pct_change(),
                turnover_series=hybrid_tno,
                avg_trade_size=0.02  # estimate trades/yr from turnover
            )
        )
        if strategy_cum_net is not None and show_net:
            rows.append(
                backend.kpi_row(
                    "Strategy (Net of costs)",
                    strategy_cum_net.pct_change(),
                    turnover_series=hybrid_tno,
                    avg_trade_size=0.02
                )
            )
        rows.append(
            backend.kpi_row(
                "QQQ Benchmark",
                qqq_cum.pct_change()
            )
        )

        kdf = pd.DataFrame(
            rows,
            columns=["Model", "Freq", "CAGR", "Sharpe", "Sortino", "Calmar", "MaxDD", "Trades/yr", "Turnover/yr", "Equity x"]
        )
        st.dataframe(kdf, use_container_width=True)

        # Equity curves
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(strategy_cum_gross.index, strategy_cum_gross.values, label="Strategy (Gross)")
        if strategy_cum_net is not None and show_net:
            ax.plot(strategy_cum_net.index, strategy_cum_net.values, label="Strategy (Net)", linestyle=":")
        ax.plot(qqq_cum.index, qqq_cum.values, label="QQQ", linestyle="--")
        ax.set_yscale("log")
        ax.set_ylabel("Cumulative Growth (log)")
        ax.grid(True, ls="--", alpha=0.6)
        ax.legend()
        st.pyplot(fig)

        st.caption("Trades/yr estimated from turnover assuming ~2% avg single-leg trade (configurable in code).")

# ---------------------------
# Tab 4: Regime
# ---------------------------
with tab4:
    st.subheader("🧭 Market Regime")
    try:
        label, metrics = backend.get_market_regime()
        c1, c2, c3 = st.columns(3)
        c1.metric("Regime", label)
        c2.metric("% Universe >200DMA", f"{metrics.get('universe_above_200dma', np.nan)*100:.1f}%")
        c3.metric("QQQ above 200DMA", "Yes" if metrics.get("qqq_above_200dma", 0.0) >= 1.0 else "No")

        c4, c5 = st.columns(2)
        c4.metric("QQQ 10D Vol", f"{metrics.get('qqq_vol_10d', np.nan)*100:.2f}%")
        c5.metric("QQQ 50DMA slope (10D)", f"{metrics.get('qqq_50dma_slope_10d', np.nan)*100:.2f}%")

        st.markdown(
            "**Breadth (share of tickers with positive 6-month return):** "
            f"{metrics.get('breadth_pos_6m', np.nan)*100:.1f}%"
        )

        # Opinionated guidance (informational)
        breadth = float(metrics.get("breadth_pos_6m", np.nan))
        vol10   = float(metrics.get("qqq_vol_10d", np.nan))
        qqq_abv = bool(metrics.get("qqq_above_200dma", 0.0) >= 1.0)

        target_equity = 1.0
        headline = "Risk-On — full equity allocation recommended."
        box = st.success

        if ((not qqq_abv and (breadth < 0.35)) or (vol10 > 0.035 and not qqq_abv)):
            target_equity = 0.0
            headline = "Extreme Risk-Off — consider 100% cash."
            box = st.error
        elif (not qqq_abv) or (breadth < 0.45):
            target_equity = 0.50
            headline = "Risk-Off — scale to ~50% equity / 50% cash."
            box = st.warning

        box(
            f"**Regime advice:** {headline}  \n"
            f"**Targets:** equity **{target_equity*100:.0f}%**, cash **{(1-target_equity)*100:.0f}%**.  \n"
            f"_Context — Breadth (6m>0): {breadth:.0%} • 10-day vol: {vol10:.2%} • QQQ >200DMA: {'Yes' if qqq_abv else 'No'}_"
        )
        st.caption("Note: ISA Dynamic already scales via stickiness/trigger; treat this as a sanity check.")
    except Exception as e:
        st.error(f"Failed to load regime data: {e}")
        st.code(traceback.format_exc())

# ---------------------------
# Tab 5: Changes (Explainability)
# ---------------------------
with tab5:
    st.subheader("🔎 What changed and why?")
    if live_raw is None or live_raw.empty:
        st.info("Generate first to see changes.")
    else:
        try:
            # Pull a small recent window for signals
            uni_tickers, _, _ = backend.get_universe(universe_choice)
            end = datetime.today().strftime("%Y-%m-%d")
            start = (datetime.today() - relativedelta(months=12)).strftime("%Y-%m-%d")
            px = backend.fetch_market_data(uni_tickers, start, end)

            # Explain vs last saved (fallback to live if no saved)
            base_df = backend.load_previous_portfolio()
            if base_df is None or base_df.empty:
                base_df = live_raw

            expl = backend.explain_portfolio_changes(base_df, live_raw, px, backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"])
            if expl is None or expl.empty:
                st.info("No changes to explain.")
            else:
                # Tidy display
                show = expl.copy()
                for col in show.columns:
                    if "Score" in col:
                        show[col] = show[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
                    if "Return" in col:
                        show[col] = show[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
                st.dataframe(show, use_container_width=True)
        except Exception as e:
            st.error(f"Explainability failed: {e}")
            st.code(traceback.format_exc())
