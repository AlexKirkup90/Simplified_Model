# app.py
import traceback
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import backend  # our production module

st.set_page_config(layout="wide", page_title="Hybrid Momentum Portfolio")

# ======================
# Sidebar Controls
# ======================
st.sidebar.header("⚙️ Strategy Settings")

universe_name = st.sidebar.selectbox(
    "Universe",
    ["NASDAQ100+", "Hybrid Top150", "S&P500"],
    index=1,
    help="Choose where the model can pick stocks from."
)

min_dollar_volume = st.sidebar.number_input(
    "Liquidity floor (median $ volume, last 60d)",
    min_value=0.0, value=0.0, step=1_000_000.0,
    help="Optional: drop illiquid names (0 = off). Ignored for Hybrid Top150 which uses top-N by liquidity."
)

roundtrip_bps = st.sidebar.number_input(
    "Rebalance cost (round-trip, bps)",
    min_value=0, max_value=200, value=0, step=5,
    help="Applied to turnover in backtests when 'Show net' is on."
)

show_net = st.sidebar.checkbox("Show net of costs", value=False)

st.sidebar.header("Momentum Sleeve")
momentum_window = st.sidebar.slider("Momentum Lookback (Months)", 3, 18, 15, 1)
top_n = st.sidebar.slider("Number of Stocks (Momentum)", 5, 20, 8, 1)
cap = st.sidebar.slider("Max Weight Cap per Stock", 0.10, 0.50, 0.25, 0.01, format="%.2f")

st.sidebar.header("ISA Dynamic Weights")
mom_w = st.sidebar.slider("Weight: Momentum", 0.50, 0.95, 0.85, 0.05)
mr_w = 1.0 - mom_w
st.sidebar.caption(f"Mean-Reversion weight auto-set to **{mr_w:.0%}**.")

st.sidebar.header("Rebalancing")
tol = st.sidebar.slider("Rebalancing Tolerance (Δ weight)", 0.005, 0.05, 0.01, 0.005, format="%.3f")

st.title("🚀 Hybrid Momentum Portfolio Manager")

st.markdown(
    f"**Universe:** `{universe_name}` &nbsp;|&nbsp; **Liquidity floor:** "
    f"{'off' if min_dollar_volume==0 else f'${min_dollar_volume:,.0f}'} &nbsp;|&nbsp; "
    f"**Costs:** {roundtrip_bps} bps (round-trip) {'(shown)' if show_net else '(hidden)'}"
)

# ======================
# Action button
# ======================
if st.button("Generate Portfolio & Rebalancing Plan", type="primary", use_container_width=True):

    with st.spinner("Crunching numbers..."):
        try:
            # ---- Step 1: Live portfolio (ISA dynamic with monthly lock)
            prev_portfolio = backend.load_previous_portfolio()

            preset = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"].copy()
            preset["mom_w"] = mom_w
            preset["mr_w"]  = mr_w

            live_disp, live_raw, decision = backend.generate_live_portfolio_isa_monthly(
                preset=preset,
                prev_portfolio=prev_portfolio,
                universe_name=universe_name,
                min_dollar_volume=min_dollar_volume,
                hybrid_top_n=150
            )

            # ---- Step 2: Backtest (Classic 90/10) + ISA (optional)
            strategy_cum_gross, strategy_cum_net, qqq_cum, hybrid_tno = backend.run_backtest_for_app(
                momentum_window=momentum_window,
                top_n=top_n,
                cap=cap,
                universe_name=universe_name,
                roundtrip_bps=roundtrip_bps,
                min_dollar_volume=min_dollar_volume,
                hybrid_top_n=150,
                show_net=show_net
            )

            # ---- Tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["📊 Rebalancing Plan", "✅ Current Portfolio", "📈 Performance", "🧭 Regime", "🧩 Changes"]
            )

            # =======================
            # Tab 1: Rebalancing Plan
            # =======================
            with tab1:
                st.subheader("Rebalancing Plan")
                if live_raw is None or live_raw.empty:
                    st.warning("No portfolio available.")
                else:
                    signals = backend.diff_portfolios(prev_portfolio, live_raw, tol)
                    if not any(s for s in signals.values()):
                        st.success("✅ No major rebalancing needed!")
                    else:
                        cols = st.columns(3)
                        with cols[0]:
                            st.markdown("### 🔴 Sell Completely")
                            if signals['sell']:
                                for ticker in signals['sell']: st.markdown(f"- **{ticker}**")
                            else:
                                st.caption("None")
                        with cols[1]:
                            st.markdown("### 🟢 New Buys")
                            if signals['buy']:
                                for ticker in signals['buy']:
                                    weight = live_raw.at[ticker, 'Weight']
                                    st.markdown(f"- **{ticker}** (Target: {weight:.2%})")
                            else:
                                st.caption("None")
                        with cols[2]:
                            st.markdown("### 🔄 Rebalance")
                            if signals['rebalance']:
                                for ticker, old_w, new_w in signals['rebalance']:
                                    st.markdown(f"- **{ticker}**: {old_w:.2%} → **{new_w:.2%}**")
                            else:
                                st.caption("None")

            # =======================
            # Tab 2: Current Portfolio
            # =======================
            with tab2:
                st.subheader("Current Portfolio (Monthly-Locked)")
                if live_disp is None or live_disp.empty:
                    st.warning(decision)
                else:
                    st.caption(decision)
                    st.dataframe(live_disp, use_container_width=True)

                    # Preview next rebalance (does NOT save)
                    with st.expander("🔎 Preview next rebalance (does NOT trade or save)", expanded=False):
                        try:
                            univ = backend.get_universe(universe_name)
                            end = datetime.today().strftime("%Y-%m-%d")
                            lookback_months = max(backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]["mom_lb"], 12) + 8
                            start = (datetime.today() - relativedelta(months=lookback_months)).strftime("%Y-%m-%d")

                            close, vol, _ = backend.get_universe_prices(
                                universe_name, start, end,
                                min_dollar_volume=min_dollar_volume, hybrid_top_n=150
                            )
                            if close.empty:
                                st.info("Could not fetch price/volume for preview.")
                            else:
                                preview_params = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"].copy()
                                preview_params.update({"mom_w": mom_w, "mr_w": mr_w})

                                preview_w = backend._build_isa_weights(close, preview_params)
                                if preview_w is None or preview_w.empty:
                                    st.info("No preview weights available (signals too weak or filters too strict).")
                                else:
                                    preview_df = pd.DataFrame({"Weight": preview_w}).sort_values("Weight", ascending=False)
                                    preview_fmt = preview_df.copy()
                                    preview_fmt["Weight"] = preview_fmt["Weight"].map("{:.2%}".format)

                                    st.markdown("**Proposed target weights if you rebalanced today (preview only):**")
                                    st.dataframe(preview_fmt, use_container_width=True)

                                    base_df = live_raw if (live_raw is not None and not live_raw.empty) else backend.load_previous_portfolio()
                                    if base_df is not None and not base_df.empty:
                                        changes = backend.diff_portfolios(base_df, preview_df, tol=0.01)
                                        st.markdown("**Changes (preview):**")
                                        c1, c2, c3 = st.columns(3)
                                        with c1:
                                            st.markdown("🟥 **Sells**")
                                            if changes["sell"]:
                                                for t in changes["sell"]: st.write(f"- **{t}**")
                                            else:
                                                st.write("None")
                                        with c2:
                                            st.markdown("🟩 **Buys**")
                                            if changes["buy"]:
                                                for t in changes["buy"]:
                                                    tgt = float(preview_df.loc[t, "Weight"]) if t in preview_df.index else 0.0
                                                    st.write(f"- **{t}** — target {tgt:.2%}")
                                            else:
                                                st.write("None")
                                        with c3:
                                            st.markdown("🔄 **Rebalances (±≥1%)**")
                                            if changes["rebalance"]:
                                                for t, old_w, new_w in changes["rebalance"]:
                                                    st.write(f"- **{t}**: {old_w:.2%} → **{new_w:.2%}**")
                                            else:
                                                st.write("None")
                        except Exception as e:
                            st.error(f"Preview failed: {e}")

                    # Mini bar chart
                    try:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        live_raw_sorted = live_raw.sort_values("Weight", ascending=False)
                        ax.bar(live_raw_sorted.index, live_raw_sorted["Weight"].values)
                        ax.set_ylabel("Weight")
                        ax.set_xticklabels(live_raw_sorted.index, rotation=45, ha="right")
                        st.pyplot(fig)
                    except Exception:
                        pass

                    if st.button("💾 Save this portfolio to Gist"):
                        backend.save_portfolio_to_gist(live_raw)
                        st.success("Saved to Gist.")

            # =======================
            # Tab 3: Performance
            # =======================
            with tab3:
                st.subheader("📈 Backtest (since 2018)")
                if strategy_cum_gross is None or qqq_cum is None:
                    st.info("Generate to see backtest results.")
                else:
                    st.markdown("#### Key Performance (monthly series inferred)")
                    krows = []
                    krows.append(
                        backend.kpi_row(
                            "Strategy (Gross)",
                            strategy_cum_gross.pct_change(),
                            turnover_series=hybrid_tno,
                            avg_trade_size=0.02
                        )
                    )
                    if strategy_cum_net is not None and show_net:
                        krows.append(
                            backend.kpi_row(
                                "Strategy (Net of costs)",
                                strategy_cum_net.pct_change(),
                                turnover_series=hybrid_tno,
                                avg_trade_size=0.02
                            )
                        )
                    krows.append(
                        backend.kpi_row(
                            "QQQ Benchmark",
                            qqq_cum.pct_change()
                        )
                    )
                    kdf = pd.DataFrame(
                        krows,
                        columns=["Model","Freq","CAGR","Sharpe","Sortino","Calmar","MaxDD","Trades/yr","Turnover/yr","Equity x"]
                    )
                    st.dataframe(kdf, use_container_width=True)

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

            # =======================
            # Tab 4: Regime
            # =======================
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
                    st.markdown("**Breadth (share of tickers with positive 6-month return):** "
                                f"{metrics.get('breadth_pos_6m', np.nan)*100:.1f}%")

                    # Simple advice box
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
                    st.caption("Note: The ISA Dynamic preset already scales exposure automatically; use this as a sanity check for monthly decisions.")
                except Exception as e:
                    st.error(f"Failed to load regime data: {e}")

            # =======================
            # Tab 5: Changes (Explainability)
            # =======================
            with tab5:
                st.subheader("🧩 What changed and why?")
                try:
                    # For signal snapshot we need daily prices on current universe
                    lookback_months = max(backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]["mom_lb"], 12) + 8
                    end = datetime.today().strftime("%Y-%m-%d")
                    start = (datetime.today() - relativedelta(months=lookback_months)).strftime("%Y-%m-%d")
                    close, vol, _ = backend.get_universe_prices(
                        universe_name, start, end,
                        min_dollar_volume=min_dollar_volume, hybrid_top_n=150
                    )

                    prev_df = backend.load_previous_portfolio()
                    expl = backend.explain_portfolio_changes(prev_df, live_raw, close, backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"])
                    if expl is None or expl.empty:
                        st.info("No changes to explain.")
                    else:
                        st.dataframe(expl, use_container_width=True)
                except Exception as e:
                    st.error(f"Explainability failed: {e}")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.code(traceback.format_exc())
