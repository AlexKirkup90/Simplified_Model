# app.py — ISA Dynamic Portfolio (Monthly-Locked with Countdown)
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

import backend

# =========================
# Page Config
# =========================
st.set_page_config(layout="wide", page_title="ISA Dynamic Portfolio (Monthly)")

# =========================
# Date helpers (UI)
# =========================
def first_business_day(dt: datetime) -> pd.Timestamp:
    start = pd.Timestamp(dt.date()).replace(day=1)
    return pd.date_range(start=start, periods=1, freq="BMS")[0]

def next_rebalance_day(dt: datetime) -> pd.Timestamp:
    # next month’s first business day
    nxt = dt + relativedelta(months=1)
    return first_business_day(nxt)

def is_rebalance_day(dt: datetime) -> bool:
    return pd.Timestamp(dt.date()) == first_business_day(dt)

def human_countdown(target_ts: pd.Timestamp) -> str:
    now = pd.Timestamp(datetime.now())
    delta = target_ts - now
    if delta.total_seconds() <= 0:
        return "Today"
    days = delta.days
    hours = int((delta.total_seconds() - days * 86400) // 3600)
    mins = int((delta.total_seconds() - days * 86400 - hours * 3600) // 60)
    parts = []
    if days > 0: parts.append(f"{days}d")
    if hours > 0: parts.append(f"{hours}h")
    if mins > 0: parts.append(f"{mins}m")
    return " ".join(parts) if parts else "Soon"

# =========================
# Sidebar controls
# =========================
st.sidebar.header("⚙️ ISA Dynamic Settings")
preset = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"].copy()

# (Keep these in sync with backend preset defaults)
trigger = st.sidebar.slider("Rebalance Trigger (Health)", 0.50, 1.00, float(preset["trigger"]), 0.05)
mom_w   = st.sidebar.slider("Momentum Weight", 0.0, 1.0, float(preset["mom_w"]), 0.05)
mr_w    = round(1.0 - mom_w, 2)
st.sidebar.caption(f"MR Weight auto: **{mr_w:.2f}**")

min_dollar_volume = st.sidebar.number_input("Min Median $ Volume (60d)", min_value=0.0, value=0.0, step=1e6, format="%.0f")
show_net = st.sidebar.checkbox("Show Net of Costs in Backtest", value=False)
roundtrip_bps = st.sidebar.slider("Roundtrip Cost (bps)", 0, 100, 10, 1)

st.title("📈 ISA Dynamic Portfolio — Monthly Execution")

# =========================
# Monthly lock status + countdown
# =========================
today = datetime.today()
rebalance_today = backend.is_rebalance_today(today.date(), None)  # backend uses price index if available internally
nxt_reb = next_rebalance_day(today)
colA, colB = st.columns(2)
with colA:
    if rebalance_today:
        st.success(f"Rebalance Day — {today.strftime('%Y-%m-%d')}")
    else:
        st.info(f"Holding portfolio — next rebalance: **{nxt_reb.date()}**")
with colB:
    st.metric("⏳ Time until next rebalance", human_countdown(nxt_reb))

# =========================
# Generate/Load portfolio (monthly-locked inside backend)
# =========================
prev_port = backend.load_previous_portfolio()
try:
    # The function may return either (display_df, raw_df, decision)
    # or ((display_df, raw_df), decision). Handle both defensively.
    result = backend.generate_live_portfolio_isa_monthly(
        preset={
            "mom_lb": preset["mom_lb"], "mom_topn": preset["mom_topn"], "mom_cap": preset["mom_cap"],
            "mr_lb": preset["mr_lb"], "mr_topn": preset["mr_topn"], "mr_ma": preset["mr_ma"],
            "mom_w": mom_w, "mr_w": mr_w,
            "trigger": trigger,
            "stability_days": preset.get("stability_days", 5),
        },
        prev_portfolio=prev_port,
        min_dollar_volume=min_dollar_volume
    )
    # Unpack robustly
    if isinstance(result, tuple) and len(result) == 3:
        live_disp, live_raw, decision = result
    elif isinstance(result, tuple) and len(result) == 2:
        pair, decision = result
        live_disp, live_raw = pair
    else:
        live_disp, live_raw, decision = None, None, "Unexpected return shape from backend."
except Exception as e:
    live_disp, live_raw, decision = None, None, f"Error building portfolio: {e}"

# Save if we actually rebalanced (backend decides via monthly lock + trigger)
if live_raw is not None and not live_raw.empty:
    backend.save_current_portfolio(live_raw)

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Performance", "💼 Current Portfolio", "💰 Cost / Liquidity", "🧭 Regime", "🔄 Changes"
])

# ---- Tab 1: Performance ----
with tab1:
    st.subheader("Backtest vs QQQ (Since 2018)")
    strat_cum_gross, strat_cum_net, qqq_cum, tno = backend.run_backtest_isa_dynamic(
        roundtrip_bps=roundtrip_bps,
        min_dollar_volume=min_dollar_volume,
        show_net=show_net
    )
    if strat_cum_gross is None:
        st.warning("Could not run backtest (missing data).")
    else:
        # Returns series from cum curves
        strat_rets = strat_cum_gross.pct_change().fillna(0.0) if not show_net else strat_cum_net.pct_change().fillna(0.0)
        bench_rets = qqq_cum.pct_change().fillna(0.0)

        # KPIs
        cols = st.columns(2)
        with cols[0]:
            k = backend.kpi_row("ISA Dynamic", strat_rets, trade_log=None, turnover_series=tno)
            st.markdown("**ISA Dynamic (selected)**")
            st.write(dict(zip(
                ["Model","Freq","CAGR","Sharpe","Sortino","Calmar","MaxDD","Trades/yr","Turnover/yr","Equity×"],
                k
            )))
        with cols[1]:
            k = backend.kpi_row("QQQ", bench_rets)
            st.markdown("**QQQ Benchmark**")
            st.write(dict(zip(
                ["Model","Freq","CAGR","Sharpe","Sortino","Calmar","MaxDD","Trades/yr","Turnover/yr","Equity×"],
                k
            )))

        # Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        label = "ISA Dynamic (Net)" if show_net else "ISA Dynamic (Gross)"
        ax.plot(strat_cum_net.index if show_net else strat_cum_gross.index,
                strat_cum_net.values if show_net else strat_cum_gross.values, label=label)
        ax.plot(qqq_cum.index, qqq_cum.values, label="QQQ", linestyle="--")
        ax.set_yscale("log")
        ax.set_ylabel("Cumulative Growth (log)")
        ax.grid(True, which="both", ls="--")
        ax.legend()
        st.pyplot(fig)

# ---- Tab 2: Current Portfolio ----
with tab2:
    st.subheader("Current Portfolio (Monthly-Locked)")
    if live_disp is None or live_disp.empty:
        st.warning(decision)
    else:
        st.caption(decision)
        st.dataframe(live_disp, use_container_width=True)

        with st.expander("🔎 Preview next rebalance (does NOT trade or save)"):
            try:
                # 1) Pull universe prices (last ~max lookback + buffer)
                univ = backend.get_nasdaq_100_plus_tickers()
                end = datetime.today().strftime("%Y-%m-%d")
                lookback_months = max(backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]["mom_lb"], 12) + 8
                start = (datetime.today() - relativedelta(months=lookback_months)).strftime("%Y-%m-%d")

                close, vol = backend.fetch_price_volume(univ, start, end)
                if close.empty:
                    st.warning("Could not fetch price/volume for preview.")
                else:
                    # 2) Optional liquidity filter
                    try:
                        _min_dollar = float(min_dollar_volume)  # from sidebar
                    except Exception:
                        _min_dollar = 0.0

                    available = [t for t in univ if t in close.columns]
                    if _min_dollar > 0 and available:
                        keep = backend.filter_by_liquidity(close[available], vol[available], _min_dollar)
                        if keep:
                            close = close[keep]
                        else:
                            close = close[[]]

                    if close.empty:
                        st.info("No tickers remain after liquidity filtering.")
                    else:
                        # 3) Build proposed ISA weights
                        preview_params = {
                            "mom_lb":   backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]["mom_lb"],
                            "mom_topn": backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]["mom_topn"],
                            "mom_cap":  backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]["mom_cap"],
                            "mr_lb":    backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]["mr_lb"],
                            "mr_topn":  backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]["mr_topn"],
                            "mr_ma":    backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]["mr_ma"],
                            "mom_w":    float(mom_w) if "mom_w" in locals() else backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]["mom_w"],
                            "mr_w":     float(mr_w)  if "mr_w"  in locals() else (1.0 - backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]["mom_w"]),
                            "trigger":  float(trigger) if "trigger" in locals() else backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]["trigger"],
                            "stability_days": backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"].get("stability_days", 5),
                        }

                        preview_w = backend._build_isa_weights(close, preview_params)
                        if preview_w is None or preview_w.empty:
                            st.info("No preview weights available.")
                        else:
                            preview_df = pd.DataFrame({"Weight": preview_w}).sort_values("Weight", ascending=False)
                            preview_fmt = preview_df.copy()
                            preview_fmt["Weight"] = preview_fmt["Weight"].map("{:.2%}".format)

                            st.markdown("**Proposed target weights if you rebalanced today (preview only):**")
                            st.dataframe(preview_fmt, use_container_width=True)

                            # 4) Changes vs CURRENT portfolio
                            base_df = None
                            if 'live_raw' in locals() and live_raw is not None and not live_raw.empty:
                                base_df = live_raw
                                st.caption("Diff vs current on-screen portfolio.")
                            else:
                                base_df = backend.load_previous_portfolio()
                                st.caption("Diff vs last saved portfolio.")

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

# ---- Tab 3: Cost / Liquidity ----
with tab3:
    st.subheader("Estimated Liquidity (Median $ Volume, 60d)")
    try:
        if live_raw is None or live_raw.empty:
            st.info("No live portfolio to analyze.")
        else:
            # fetch recent month for constituents only (faster)
            end = datetime.today().strftime("%Y-%m-%d")
            start = (datetime.today() - relativedelta(months=3)).strftime("%Y-%m-%d")
            tickers = list(live_raw.index)
            close, vol = backend.fetch_price_volume(tickers, start, end)
            if close.empty:
                st.warning("Could not fetch recent price/volume for constituents.")
            else:
                med = backend.median_dollar_volume(close, vol, window=60)
                liq_df = pd.DataFrame({"Median_$Vol_60d": med}).sort_values("Median_$Vol_60d", ascending=False)
                st.dataframe(liq_df.style.format({"Median_$Vol_60d": "£{:,.0f}"}), use_container_width=True)
    except Exception as e:
        st.error(f"Liquidity calc failed: {e}")

# ---- Tab 4: Regime ----
with tab4:
    st.subheader("Market Regime (Breadth, Vol, Trend)")

    try:
        # Fetch regime metrics from backend
        regime_data = backend.get_market_regime()
        if not regime_data:
            st.warning("No regime data available.")
        else:
            # Extract values
            breadth = regime_data.get("universe_above_200dma", 0)
            qqq_above = regime_data.get("qqq_above_200dma", 0)
            vol_10d = regime_data.get("qqq_vol_10d", 0)
            breadth_pos_6m = regime_data.get("breadth_pos_6m", 0)
            slope_50dma = regime_data.get("qqq_50dma_slope_10d", 0)

            # Classification logic
            if breadth > 0.6 and slope_50dma > 0 and vol_10d < 0.02:
                regime = "Bull"
                colour = "🟢"
                summary = (
                    "Conditions are bullish — broad participation, positive trend, "
                    "and low volatility. Environment supportive for taking risk."
                )
            elif breadth > 0.4 and slope_50dma >= 0:
                regime = "Caution"
                colour = "🟡"
                summary = (
                    "Mixed signals — trend remains positive but breadth or volatility "
                    "is not ideal. Maintain positions but avoid aggressive buying."
                )
            else:
                regime = "Bear"
                colour = "🔴"
                summary = (
                    "Defensive conditions — weak breadth or negative trend with higher volatility. "
                    "Reduce risk and protect capital."
                )

            # Display
            st.markdown(f"### {colour} {regime} Regime")
            st.write(summary)

            with st.expander("View raw regime metrics"):
                st.json(regime_data)

            # Optional: breadth gauge
            st.progress(min(max(breadth, 0), 1))
            st.caption(f"Breadth: {breadth:.1%} of universe above 200DMA")

            # Snapshot
            note = st.text_input("Add a note to live snapshot (optional)")
            if st.button("Record Live Snapshot"):
                backend.save_regime_snapshot(regime_data, note)
                st.success("Snapshot saved.")

    except Exception as e:
        st.error(f"Failed to load regime data: {e}")

# ---- Tab 5: Changes ----
with tab5:
    st.subheader("Changes vs Last Saved Portfolio")
    prev_port = backend.load_previous_portfolio()
    if (prev_port is None or prev_port.empty) or (live_raw is None or live_raw.empty):
        st.info("Need both a previous and current portfolio to compute changes.")
    else:
        changes = backend.diff_portfolios(prev_port, live_raw, tol=0.01)
        # Render changes
        cols = st.columns(3)
        with cols[0]:
            st.markdown("### 🟥 Sells")
            if changes["sell"]:
                for t in changes["sell"]:
                    st.write(f"- **{t}**")
            else:
                st.write("None")
        with cols[1]:
            st.markdown("### 🟩 Buys")
            if changes["buy"]:
                for t in changes["buy"]:
                    tgt = float(live_raw.loc[t, "Weight"]) if t in live_raw.index else 0.0
                    st.write(f"- **{t}** — target {tgt:.2%}")
            else:
                st.write("None")
        with cols[2]:
            st.markdown("### 🔄 Rebalances (±≥1%)")
            if changes["rebalance"]:
                for t, old_w, new_w in changes["rebalance"]:
                    st.write(f"- **{t}**: {old_w:.2%} → **{new_w:.2%}**")
            else:
                st.write("None")

# =========================
# Live snapshot logging (optional button)
# =========================
st.divider()
if live_raw is not None and not live_raw.empty:
    note = st.text_input("Add a note to live snapshot (optional)", "")
    if st.button("📝 Record Live Snapshot"):
        res = backend.record_live_snapshot(live_raw, note=note)
        if res.get("ok", False):
            st.success(f"Recorded snapshot for {res.get('rows', '?')} day(s).")
        else:
            st.error(res.get("msg", "Failed to log snapshot."))
