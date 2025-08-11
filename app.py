# app.py
import traceback
from datetime import datetime
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import backend  # uses your latest backend.py

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(layout="wide", page_title="Hybrid Momentum Portfolio (ISA Dynamic)")

st.title("🚀 Hybrid Momentum Portfolio Manager (Monthly-Locked ISA)")

st.markdown(
    "This app builds a **monthly-locked**, stability-aware hybrid (Momentum + Mean-Reversion) "
    "portfolio with optional liquidity screens and cost-aware backtests."
)

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("⚙️ Strategy Settings")

# Universe choice (backend currently fetches NASDAQ100+; shown here for future extension)
universe_choice = st.sidebar.selectbox(
    "Universe",
    ["NASDAQ100+", "S&P500 (All)", "Hybrid Top150"],
    index=0,
    help="Current backend uses NASDAQ100+ in live/BT. Other universes will be wired later."
)

# Cost & liquidity
roundtrip_bps = st.sidebar.number_input(
    "Roundtrip cost (bps)", min_value=0, max_value=200, value=30, step=5,
    help="Estimated all-in trading cost per roundtrip, in basis points."
)
min_dollar_volume = st.sidebar.number_input(
    "Min median $ volume (60d)", min_value=0, value=0, step=1000000,
    help="Liquidity filter. 0 disables the filter."
)

# Net-of-costs toggle for charts/table
show_net = st.sidebar.checkbox("Show Net-of-costs line in Performance", value=True)

# Rebalance tolerance for plan (±1% default)
tol = st.sidebar.slider("Rebalance tolerance (±)", 0.0, 0.05, 0.01, 0.005, format="%.3f")

# Action
go = st.sidebar.button("Generate Portfolio & Backtest", type="primary", use_container_width=True)

# Keep latest portfolio in session
if "latest_portfolio" not in st.session_state:
    st.session_state.latest_portfolio = None

# ---------------------------
# Helper: draw simple bar chart
# ---------------------------
def plot_weights_bar(df_weights: pd.DataFrame, title: str):
    try:
        if df_weights is None or df_weights.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 4))
        ordered = df_weights.sort_values("Weight", ascending=False)
        ax.bar(ordered.index, ordered["Weight"].values)
        ax.set_title(title)
        ax.set_ylabel("Weight")
        ax.set_xticklabels(ordered.index, rotation=45, ha="right")
        st.pyplot(fig)
    except Exception:
        pass

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Rebalancing Plan", "✅ Current Portfolio", "📈 Performance", "🧭 Regime", "🔎 Changes"]
)

# Defaults (filled after Generate)
prev_portfolio = backend.load_previous_portfolio()
live_disp = None
live_raw = None
decision = "Click Generate to build portfolio."
strategy_cum_gross = None
strategy_cum_net = None
qqq_cum = None
hybrid_tno = None

# ===========================
# Generate
# ===========================
if go:
    with st.spinner("Building monthly-locked portfolio and running backtest…"):
        # ---- Load preset
        try:
            preset = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]
        except Exception:
            st.error("ISA preset not found in backend. Please update backend.STRATEGY_PRESETS.")
            st.stop()

        # ---- Live portfolio (monthly lock + stability + trigger)
        try:
            live_disp, live_raw, decision = backend.generate_live_portfolio_isa_monthly(
                preset=preset,
                prev_portfolio=prev_portfolio,
                min_dollar_volume=min_dollar_volume
            )
        except Exception as e:
            live_disp, live_raw, decision = None, None, "Portfolio generation failed."
            st.error(f"Portfolio generation failed: {e}")
            st.code(traceback.format_exc())

        # ---- Backtest (ISA Dynamic, hybrid rules, 2017-07-01 start)
        try:
            strategy_cum_gross, strategy_cum_net, qqq_cum, hybrid_tno = backend.run_backtest_isa_dynamic(
                roundtrip_bps=roundtrip_bps,
                min_dollar_volume=min_dollar_volume,
                show_net=show_net,
                start_date="2017-07-01"
            )
        except Exception as e:
            strategy_cum_gross = strategy_cum_net = qqq_cum = hybrid_tno = None
            st.warning(f"Backtest failed: {e}")

        # ---- Persist latest portfolio to session (for Save / Diff)
        if live_raw is not None and not live_raw.empty:
            st.session_state.latest_portfolio = live_raw.copy()
# ===========================
# Tab 1 — Rebalancing Plan
# ===========================
with tab1:
    st.subheader("📊 Rebalancing Plan (vs last saved)")
    if live_raw is None or live_raw.empty:
        st.info("Click Generate to build the portfolio.")
    else:
        # Diff vs previous saved (or empty if none)
        try:
            signals = backend.diff_portfolios(prev_portfolio, live_raw, tol=tol)
            if not any([signals["sell"], signals["buy"], signals["rebalance"]]):
                st.success("✅ No major rebalancing needed!")
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("🔴 **Sells**")
                    if signals["sell"]:
                        for t in signals["sell"]:
                            st.write(f"- **{t}**")
                    else:
                        st.write("None")
                with c2:
                    st.markdown("🟢 **Buys**")
                    if signals["buy"]:
                        for t in signals["buy"]:
                            w = float(live_raw.loc[t, "Weight"]) if t in live_raw.index else 0.0
                            st.write(f"- **{t}** — target {w:.2%}")
                    else:
                        st.write("None")
                with c3:
                    st.markdown("🔄 **Rebalances (±≥{:.1f}%)**".format(tol*100))
                    if signals["rebalance"]:
                        for t, w_old, w_new in signals["rebalance"]:
                            st.write(f"- **{t}**: {w_old:.2%} → **{w_new:.2%}**")
                    else:
                        st.write("None")
        except Exception as e:
            st.error(f"Plan diff failed: {e}")

        st.divider()
        st.caption("Decision engine: " + (decision or ""))
        if st.button("💾 Save this portfolio for next month", use_container_width=True):
            try:
                backend.save_portfolio_to_gist(live_raw)
                backend.save_current_portfolio(live_raw)
                st.success("Saved to Gist (if configured) and local CSV.")
            except Exception as e:
                st.error(f"Save failed: {e}")

# ===========================
# Tab 2 — Current Portfolio
# ===========================
with tab2:
    st.subheader("✅ Current Portfolio (Monthly-Locked)")
    if live_disp is None or live_disp.empty:
        st.warning(decision)
    else:
        st.caption(decision)
        st.dataframe(live_disp, use_container_width=True)
        plot_weights_bar(live_raw, "Current Weights")

        # ---- Preview next rebalance (does NOT trade or save)
        with st.expander("🔎 Preview next rebalance (does NOT trade or save)", expanded=False):
            try:
                # 1) Pull universe prices for preview horizon
                univ = backend.get_nasdaq_100_plus_tickers()
                end = datetime.today().strftime("%Y-%m-%d")
                lookback_months = max(backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]["mom_lb"], 12) + 8
                start = (datetime.today() - relativedelta(months=lookback_months)).strftime("%Y-%m-%d")

                close, vol = backend.fetch_price_volume(univ, start, end)
                if close.empty:
                    st.warning("Could not fetch price/volume for preview.")
                else:
                    # 2) Optional liquidity filter
                    _min_dollar = float(min_dollar_volume) if min_dollar_volume else 0.0
                    avail = [t for t in univ if t in close.columns]
                    if _min_dollar > 0 and avail:
                        keep = backend.filter_by_liquidity(close[avail], vol[avail], _min_dollar)
                        close = close[keep] if keep else close[[]]

                    if close.empty:
                        st.info("No tickers remain after liquidity filtering.")
                    else:
                        # 3) Build proposed ISA weights using SAME rules
                        preset = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]
                        preview_w = backend._build_isa_weights(close, preset)
                        if preview_w is None or preview_w.empty:
                            st.info("No preview weights available (signals too weak or filters too strict).")
                        else:
                            preview_df = pd.DataFrame({"Weight": preview_w}).sort_values("Weight", ascending=False)
                            preview_fmt = preview_df.copy()
                            preview_fmt["Weight"] = preview_fmt["Weight"].map("{:.2%}".format)

                            st.markdown("**Proposed weights if you rebalanced today (preview only):**")
                            st.dataframe(preview_fmt, use_container_width=True)

                            # 4) Diff vs current live (or last saved)
                            base_df = live_raw if (live_raw is not None and not live_raw.empty) else backend.load_previous_portfolio()
                            if base_df is not None and not base_df.empty:
                                changes = backend.diff_portfolios(base_df, preview_df, tol=0.01)
                                st.markdown("**Changes (preview):**")
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.markdown("🟥 **Sells**")
                                    if changes["sell"]:
                                        for t in changes["sell"]:
                                            st.write(f"- **{t}**")
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

# ===========================
# Tab 3 — Performance
# ===========================
with tab3:
    st.subheader("📈 Backtest (since 2018)")
    try:
        if strategy_cum_gross is None or qqq_cum is None:
            st.info("Click Generate to see backtest results.")
        else:
            st.markdown("#### Key Performance (monthly series inferred)")

            krows = []
            # Gross
            krows.append(
                backend.kpi_row(
                    "Strategy (Gross)",
                    strategy_cum_gross.pct_change(),
                    turnover_series=hybrid_tno,
                    avg_trade_size=0.02
                )
            )
            # Net
            if show_net and strategy_cum_net is not None:
                krows.append(
                    backend.kpi_row(
                        "Strategy (Net of costs)",
                        strategy_cum_net.pct_change(),
                        turnover_series=hybrid_tno,
                        avg_trade_size=0.02
                    )
                )
            # Benchmark
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
            st.caption("Trades/yr is estimated from turnover assuming ~2% average per single-leg trade (adjust in code).")

            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(strategy_cum_gross.index, strategy_cum_gross.values, label="Strategy (Gross)")
            if show_net and strategy_cum_net is not None:
                ax.plot(strategy_cum_net.index, strategy_cum_net.values, label="Strategy (Net)", linestyle=":")
            ax.plot(qqq_cum.index, qqq_cum.values, label="QQQ", linestyle="--")
            ax.set_yscale("log")
            ax.set_ylabel("Cumulative Growth (log)")
            ax.grid(True, ls="--", alpha=0.6)
            ax.legend()
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error loading performance tab: {e}")

# ===========================
# Tab 4 — Regime
# ===========================
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

        # Simple advice block
        breadth = float(metrics.get("breadth_pos_6m", np.nan))
        vol10   = float(metrics.get("qqq_vol_10d", np.nan))
        qqq_abv = bool(metrics.get("qqq_above_200dma", 0.0) >= 1.0)

        target_equity = 1.0
        headline = "Risk-On — full equity allocation recommended."
        box = st.success

        # Extreme risk-off → cash
        if ((not qqq_abv and (breadth < 0.35)) or (vol10 > 0.035 and not qqq_abv)):
            target_equity = 0.0
            headline = "Extreme Risk-Off — consider 100% cash."
            box = st.error
        # Risk-off → reduce
        elif (not qqq_abv) or (breadth < 0.45):
            target_equity = 0.50
            headline = "Risk-Off — scale to ~50% equity / 50% cash."
            box = st.warning

        box(
            f"**Regime advice:** {headline}  \n"
            f"**Targets:** equity **{target_equity*100:.0f}%**, cash **{(1-target_equity)*100:.0f}%**.  \n"
            f"_Context — Breadth (6m>0): {breadth:.0%} • 10-day vol: {vol10:.2%} • QQQ >200DMA: {'Yes' if qqq_abv else 'No'}_"
        )
        st.caption("Note: The ISA Dynamic preset already scales exposure automatically; use this as a sanity check.")
    except Exception as e:
        st.error(f"Failed to load regime data: {e}")

# ===========================
# Tab 5 — Changes (Explainability)
# ===========================
with tab5:
    st.subheader("🔎 What changed and why?")
    try:
        if live_raw is None or live_raw.empty:
            st.info("Generate a portfolio to see changes.")
        else:
            # Pull prices for only the tickers we need to explain
            tickers = list(live_raw.index)
            if prev_portfolio is not None and not prev_portfolio.empty and "Weight" in prev_portfolio.columns:
                tickers = sorted(set(tickers) | set(prev_portfolio.index.tolist()))
            end = datetime.today().strftime("%Y-%m-%d")
            start = (datetime.today() - relativedelta(months=18)).strftime("%Y-%m-%d")
            # Fetch Close only
            px = backend.fetch_market_data(tickers, start, end)
            if px.empty:
                st.warning("No price data available to compute explanations.")
            else:
                params = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]
                expl = backend.explain_portfolio_changes(prev_portfolio, live_raw, px, params)
                if expl is None or expl.empty:
                    st.info("No material changes detected.")
                else:
                    st.dataframe(expl, use_container_width=True)
    except Exception as e:
        st.error(f"Explainability failed: {e}")

# ===========================
# Footer / Help
# ===========================
st.caption("Tip: This is monthly-locked to avoid noise and fees inside an ISA. Use the Preview expander to see what the next rebalance would look like without committing.")
