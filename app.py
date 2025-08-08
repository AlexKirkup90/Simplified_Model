# app.py
import traceback
from datetime import datetime
from dateutil.relativedelta import relativedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import backend  # our backend module

st.set_page_config(page_title="Hybrid Momentum Portfolio (ISA-ready)", layout="wide")

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.selectbox(
    "Strategy Mode",
    ["ISA Dynamic (0.75) — Monthly Lock (Regime-Aware)", "Classic 90/10 (no lock)"],
    index=0,
)

# Common
tol = st.sidebar.slider("Rebalance tolerance (abs Δ weight)", 0.005, 0.05, 0.01, 0.005, format="%.3f")
min_dollar_volume = st.sidebar.number_input(
    "Min median $ volume (60d)", min_value=0.0, value=0.0, step=1_000_000.0, help="Liquidity filter"
)
roundtrip_bps = st.sidebar.slider(
    "Round-trip trading cost (bps)", 0, 100, 20, 1, help="Used in backtests only"
)
show_net = st.sidebar.checkbox("Show net of costs in backtest", value=True)

# Classic-only sliders
if "Classic" in mode:
    momentum_window = st.sidebar.slider("Momentum lookback (months)", 3, 12, 6, 1)
    top_n = st.sidebar.slider("Number of momentum names", 5, 20, 15, 1)
    cap = st.sidebar.slider("Max weight cap per stock", 0.10, 0.50, 0.25, 0.01, format="%.2f")

# App title
st.title("🚀 Hybrid Momentum Portfolio Manager (ISA-friendly)")

if "Classic" in mode:
    st.caption("**Mode:** Classic 90/10 hybrid (no lock, no regime scaling). For research & exploration.")
else:
    st.caption("**Mode:** ISA Dynamic (0.75) — Monthly lock + stability + regime-aware exposure/trigger.")

# =========================
# Buttons
# =========================
go = st.button("🔁 Generate / Refresh", type="primary", use_container_width=True)

# =========================
# Main execution
# =========================
live_disp = live_raw = None
decision = "Click **Generate / Refresh** to build the latest portfolio."

strategy_cum_gross = strategy_cum_net = qqq_cum = hybrid_tno = None

if go:
    with st.spinner("Building portfolio and running diagnostics..."):
        try:
            prev_port = backend.load_previous_portfolio()

            if "Classic" in mode:
                # Build live classic portfolio
                live_disp, live_raw = backend.generate_live_portfolio_classic(
                    momentum_window, top_n, cap, min_dollar_volume=min_dollar_volume
                )
                decision = "Classic mode has no monthly lock or regime overlay."

                # Backtest (since 2018)
                strategy_cum_gross, strategy_cum_net, qqq_cum, hybrid_tno = backend.run_backtest_for_app(
                    momentum_window, top_n, cap,
                    roundtrip_bps=roundtrip_bps, min_dollar_volume=min_dollar_volume, show_net=show_net
                )

            else:
                # ISA Dynamic (monthly lock + regime-aware)
                preset = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]
                live_disp, live_raw, decision = backend.generate_live_portfolio_isa_monthly(
                    preset, prev_port, min_dollar_volume=min_dollar_volume
                )

                # Backtest (since 2018)
                strategy_cum_gross, strategy_cum_net, qqq_cum, hybrid_tno = backend.run_backtest_isa_dynamic(
                    roundtrip_bps=roundtrip_bps, min_dollar_volume=min_dollar_volume, show_net=show_net
                )

            st.session_state.latest_portfolio = live_raw if live_raw is not None else pd.DataFrame(columns=["Weight"])
            st.session_state.latest_decision = decision

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.code(traceback.format_exc())

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Plan", "✅ Current Portfolio", "📈 Performance", "🧭 Regime", "🔍 Changes", "💸 Costs/Liquidity"]
)

# ---- Tab 1: Plan (diffs vs last saved) ----
with tab1:
    st.subheader("📊 Rebalancing Plan")

    prev_port = backend.load_previous_portfolio()
    if "latest_portfolio" in st.session_state:
        candidate = st.session_state.latest_portfolio
    else:
        candidate = None

    if candidate is None or candidate.empty:
        st.info("Generate a portfolio to see the rebalancing plan.")
    else:
        diffs = backend.diff_portfolios(prev_port, candidate, tol=tol)
        if not any([diffs["sell"], diffs["buy"], diffs["rebalance"]]):
            st.success("✅ No rebalancing needed.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("### 🔴 Sells")
                if diffs["sell"]:
                    for t in diffs["sell"]:
                        st.write(f"- **{t}**")
                else:
                    st.write("None")
            with c2:
                st.markdown("### 🟢 Buys")
                if diffs["buy"]:
                    for t in diffs["buy"]:
                        tgt = float(candidate.loc[t, "Weight"]) if t in candidate.index else 0.0
                        st.write(f"- **{t}** — target {tgt:.2%}")
                else:
                    st.write("None")
            with c3:
                st.markdown("### 🔄 Rebalances")
                if diffs["rebalance"]:
                    for t, old_w, new_w in diffs["rebalance"]:
                        st.write(f"- **{t}**: {old_w:.2%} → **{new_w:.2%}**")
                else:
                    st.write("None")

    st.divider()
    # Save actions
    if "latest_portfolio" in st.session_state and not st.session_state.latest_portfolio.empty:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save to Gist"):
                backend.save_portfolio_to_gist(st.session_state.latest_portfolio)
                backend.save_current_portfolio(st.session_state.latest_portfolio)
                st.success("Saved.")
        with c2:
            if st.button("📝 Log today’s 1-day live vs QQQ"):
                try:
                    res = backend.record_live_snapshot(st.session_state.latest_portfolio, note=mode)
                    if res.get("ok", False):
                        st.success(
                            f"Logged. 1d strat: {res['strat_ret']:.2%} vs QQQ: {res['qqq_ret']:.2%}  — {res['rows']} rows total."
                        )
                    else:
                        st.error(res.get("msg", "Failed to log."))
                except Exception as e:
                    st.error(f"Log failed: {e}")

# ---- Tab 2: Current Portfolio ----
with tab2:
    st.subheader("✅ Current Portfolio")
    decision_text = st.session_state.get("latest_decision", decision)
    st.caption(decision_text)

    if live_disp is None or live_disp.empty:
        st.warning("No portfolio to show yet.")
    else:
        # Regime/exposure tag
        try:
            regime_label, _rm = backend.get_market_regime()
            eff_expo = float(live_raw["Weight"].sum()) if (live_raw is not None and not live_raw.empty) else 0.0
            st.caption(f"Regime: **{regime_label}**  •  Effective equity exposure: **{eff_expo*100:.1f}%**")
        except Exception:
            pass

        st.dataframe(live_disp, use_container_width=True)

        # Quick bar chart
        try:
            fig, ax = plt.subplots(figsize=(10, 4))
            sr = live_raw.sort_values("Weight", ascending=False)["Weight"]
            ax.bar(sr.index, sr.values)
            ax.set_ylabel("Weight")
            ax.set_xticklabels(sr.index, rotation=45, ha="right")
            st.pyplot(fig)
        except Exception:
            pass

        # --- Preview next rebalance (regime-aware) ---
        with st.expander("🔎 Preview next rebalance (does NOT trade or save)"):
            try:
                univ = backend.get_nasdaq_100_plus_tickers()
                end = datetime.today().strftime("%Y-%m-%d")
                lookback_months = 24
                start = (datetime.today() - relativedelta(months=lookback_months)).strftime("%Y-%m-%d")
                close, vol = backend.fetch_price_volume(univ, start, end)
                if close.empty:
                    st.warning("Could not fetch price/volume for preview.")
                else:
                    # Liquidity filter
                    available = [t for t in univ if t in close.columns]
                    if min_dollar_volume > 0 and available:
                        keep = backend.filter_by_liquidity(close[available], vol[available], min_dollar_volume)
                        if keep:
                            close = close[keep]
                        else:
                            close = close[[]]

                    # Build regime-aware preview
                    preset = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]
                    preview_w, preview_trigger, preview_regime, _ = backend.build_isa_dynamic_with_regime(close, preset)

                    if preview_w is None or preview_w.empty:
                        st.info("No preview weights available (signals too weak or filters too strict).")
                    else:
                        preview_df = pd.DataFrame({"Weight": preview_w}).sort_values("Weight", ascending=False)
                        preview_fmt = preview_df.copy()
                        preview_fmt["Weight"] = preview_fmt["Weight"].map("{:.2%}".format)

                        st.caption(f"Preview regime: **{preview_regime}**  •  Trigger used: **{preview_trigger:.2f}**")
                        st.dataframe(preview_fmt, use_container_width=True)

                        # Diff vs current
                        base_df = live_raw if (live_raw is not None and not live_raw.empty) else backend.load_previous_portfolio()
                        if base_df is not None and not base_df.empty:
                            changes = backend.diff_portfolios(base_df, preview_df, tol=0.01)
                            st.markdown("**Preview changes vs current:**")
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.markdown("🟥 **Sells**")
                                st.write("None" if not changes["sell"] else "\n".join([f"- **{t}**" for t in changes["sell"]]))
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
                                    for t, ow, nw in changes["rebalance"]:
                                        st.write(f"- **{t}**: {ow:.2%} → **{nw:.2%}**")
                                else:
                                    st.write("None")
            except Exception as e:
                st.error(f"Preview failed: {e}")

# ---- Tab 3: Performance ----
with tab3:
    st.subheader("📈 Backtest (since 2018)")
    if strategy_cum_gross is None or qqq_cum is None:
        st.info("Generate to see backtest results.")
    else:
        # KPIs (gross & net)
        st.markdown("#### Key Performance (monthly series inferred)")
        krows = []
        krows.append(backend.kpi_row("Strategy (Gross)", strategy_cum_gross.pct_change()))
        if strategy_cum_net is not None and show_net:
            krows.append(backend.kpi_row("Strategy (Net of costs)", strategy_cum_net.pct_change()))
        krows.append(backend.kpi_row("QQQ Benchmark", qqq_cum.pct_change()))
        kdf = pd.DataFrame(krows, columns=["Model","Freq","CAGR","Sharpe","Sortino","Calmar","MaxDD","Trades/yr","Turnover/yr","Equity x"])
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

# ---- Tab 4: Regime ----
with tab4:
    st.subheader("🧭 Market Regime")
    try:
        # Build regime block using backend metrics
        univ = backend.get_nasdaq_100_plus_tickers()
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - relativedelta(months=12)).strftime("%Y-%m-%d")
        px = backend.fetch_market_data(univ, start, end)

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
        # === Regime Advice (simple, opinionated rules) ===
try:
    label, m = backend.get_market_regime()
    breadth = float(m.get("breadth_pos_6m", np.nan))      # share > 0 means 0..1
    vol10   = float(m.get("qqq_vol_10d", np.nan))          # daily vol (std)
    qqq_abv = bool(m.get("qqq_above_200dma", 0.0) >= 1.0)  # True/False

    # Default recommendations (you can tweak thresholds)
    target_equity = 1.0
    headline = "Risk-On — full equity allocation recommended."
    box = st.success

    # Extreme risk-off: trend weak + breadth poor OR volatility spike while below 200DMA
    if ((not qqq_abv and (breadth < 0.35)) or (vol10 > 0.035 and not qqq_abv)):
        target_equity = 0.0
        headline = "Extreme Risk-Off — consider 100% cash."
        box = st.error

    # Plain risk-off: either below 200DMA or weak breadth
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
except Exception as _e:
    st.info(f"Could not compute regime advice: {_e}")
    except Exception as e:
        st.error(f"Failed to load regime data: {e}")

# ---- Tab 5: Changes (Explainability) ----
with tab5:
    st.subheader("🔍 What changed and why?")
    try:
        prev_df = backend.load_previous_portfolio()
        curr_df = st.session_state.get("latest_portfolio", None)

        if curr_df is None or curr_df.empty:
            st.info("Generate the portfolio to view changes.")
        else:
            # Fetch prices for just the relevant tickers (fast)
            tickers = sorted(set(curr_df.index) | (set(prev_df.index) if prev_df is not None else set()))
            if tickers:
                end = datetime.today().strftime("%Y-%m-%d")
                start = (datetime.today() - relativedelta(months=18)).strftime("%Y-%m-%d")
                px = backend.fetch_market_data(tickers, start, end)
                params = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]

                expl = backend.explain_portfolio_changes(prev_df, curr_df, px, params)
                if expl is None or expl.empty:
                    st.info("No material changes to explain.")
                else:
                    st.dataframe(expl, use_container_width=True)
            else:
                st.info("No tickers found to explain.")
    except Exception as e:
        st.error(f"Explainability failed: {e}")

# ---- Tab 6: Costs & Liquidity ----
with tab6:
    st.subheader("💸 Costs & Liquidity (Backtest diagnostics)")
    if hybrid_tno is None or strategy_cum_gross is None:
        st.info("Generate to see cost/liquidity diagnostics.")
    else:
        # Turnover summary by year
        tno = pd.Series(hybrid_tno).copy()
        tno.index = pd.to_datetime(tno.index)
        yearly_turnover = tno.groupby(tno.index.year).sum()
        st.markdown("**Turnover (Σ |Δw| / 2) per year:**")
        st.bar_chart(yearly_turnover)

        # Live portfolio liquidity snapshot
        if live_raw is not None and not live_raw.empty:
            try:
                univ = list(live_raw.index)
                end = datetime.today().strftime("%Y-%m-%d")
                start = (datetime.today() - relativedelta(days=120)).strftime("%Y-%m-%d")
                close, vol = backend.fetch_price_volume(univ, start, end)
                med_dollar = backend.median_dollar_volume(close, vol, window=60).reindex(live_raw.index)
                snap = pd.DataFrame({
                    "Weight": live_raw["Weight"],
                    "Median $Vol (60d)": med_dollar
                }).sort_values("Weight", ascending=False)
                st.markdown("**Live holdings — median dollar volume (60d):**")
                st.dataframe(snap, use_container_width=True)
            except Exception as e:
                st.warning(f"Liquidity snapshot failed: {e}")
