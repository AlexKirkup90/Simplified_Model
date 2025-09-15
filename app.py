# app.py - Enhanced Hybrid Momentum Portfolio (ISA-Dynamic) with Strategy Health Monitoring
import traceback
from datetime import date
from dateutil.relativedelta import relativedelta

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import backend  # all logic lives here

# Default average trade size as a fraction
AVG_TRADE_SIZE_DEFAULT = backend.AVG_TRADE_SIZE_DEFAULT

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(layout="wide", page_title="Hybrid Momentum Portfolio (ISA-Dynamic)")

st.title("🚀 Hybrid Momentum (ISA Dynamic) — Monthly Execution")
st.caption("Enhanced composite momentum + mean reversion, with stickiness, sector caps, monthly lock, and health monitoring.")

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

# Enhanced features toggle
use_enhanced_features = st.sidebar.checkbox(
    "🔬 Use Enhanced Features",
    value=True,
    help="Enables volatility-adjusted caps, regime awareness, and signal decay"
)

# Optional parameter optimization
auto_optimize = st.sidebar.checkbox(
    "📈 Auto-optimize parameters",
    value=False,
    help="Skip presets and search for Sharpe-maximizing parameters when checked",
)
st.sidebar.caption(
    "Presets use fixed parameters and skip optimization unless 'Auto-optimize parameters' is enabled."
)

# Monte Carlo seed input
st.session_state.setdefault("mc_seed", 42)
mc_seed_input = st.sidebar.text_input(
    "Monte Carlo seed (blank for random)",
    value=str(st.session_state["mc_seed"]),
    help="Controls the random seed for Monte Carlo projections",
)
raw = mc_seed_input.strip()
try:
    st.session_state["mc_seed"] = int(raw) if raw else None
except ValueError:
    st.session_state["mc_seed"] = None
    st.sidebar.warning("Monte Carlo seed must be an integer (blank for random).")

# Initialize parameters from presets or prior assessment
st.session_state.setdefault("stickiness_days", preset.get("stability_days", 7))
st.session_state.setdefault("name_cap", preset.get("mom_cap", 0.25))
st.session_state.setdefault("sector_cap", preset.get("sector_cap", 0.30))
st.session_state.setdefault("roundtrip_bps", backend.ROUNDTRIP_BPS_DEFAULT)
st.session_state.setdefault("min_dollar_volume", 0)
st.session_state.setdefault("min_profitability", 0.0)
st.session_state.setdefault("max_leverage", 2.0)
st.session_state.setdefault("show_net", True)
st.session_state.setdefault("vix_ts_threshold", backend.VIX_TS_THRESHOLD_DEFAULT)
st.session_state.setdefault("hy_oas_threshold", backend.HY_OAS_THRESHOLD_DEFAULT)
st.session_state["use_enhanced_features"] = use_enhanced_features
st.session_state["auto_optimize"] = auto_optimize

# Derived constants
AVG_TRADE_SIZE = AVG_TRADE_SIZE_DEFAULT
tol = 0.01
show_net = st.session_state["show_net"]

# Prev portfolio (for plan & monthly lock)
prev_portfolio = backend.load_previous_portfolio()

# ---------------------------
# Market assessment
# ---------------------------
assess = st.button("Assess Market Conditions", key="assess_market")
if assess:
    with st.spinner("Assessing market conditions…"):
        assessment = backend.assess_market_conditions()
    st.write("### Market Metrics")
    st.json(assessment["metrics"])
    st.write("### Recommended Universe")
    st.write(assessment.get("universe", st.session_state.get("universe")))
    st.write("### Recommended Settings")
    st.json(assessment["settings"])
    for k, v in assessment["settings"].items():
        st.session_state[k] = v
    # ensure selected universe persists
    st.session_state["universe"] = assessment.get("universe", st.session_state.get("universe"))

    # -------------------------
    # Assessment log & accuracy
    # -------------------------
    with st.expander("📄 Assessment History & Accuracy"):
        log = backend.load_assess_log()
        summary = backend.evaluate_assessment_accuracy(log)

        hist = summary.get("history", pd.DataFrame())
        if not hist.empty:
            # merge regime label from metrics
            hist = hist.merge(log[["date", "metrics"]], on="date", how="left")

            def _regime(x):
                try:
                    return json.loads(x).get("regime", "n/a")
                except Exception:
                    return "n/a"

            hist["regime"] = hist["metrics"].apply(_regime)
            hist = hist.drop(columns=["metrics"])
            st.write("#### Log")
            st.dataframe(hist.sort_values("date", ascending=False), use_container_width=True)
        else:
            st.write("No assessment history yet.")

        st.write("#### Summary Statistics")
        c1, c2 = st.columns(2)
        hit_rate = summary.get("hit_rate", np.nan)
        avg_alpha = summary.get("avg_alpha", np.nan)
        c1.metric("Hit Rate", f"{hit_rate:.1%}" if pd.notna(hit_rate) else "n/a")
        c2.metric("Avg Alpha", f"{avg_alpha:.2%}" if pd.notna(avg_alpha) else "n/a")
        confusion = summary.get("confusion_matrix", pd.DataFrame())
        if not confusion.empty:
            st.write("Confusion Matrix")
            st.dataframe(confusion, use_container_width=True)

        # Highlight parameter adjustments from recalibration
        prev_map = st.session_state.get("param_map_defaults", backend.PARAM_MAP_DEFAULTS)
        backend.update_parameter_mapping(log)
        new_map = st.session_state.get("param_map_defaults", prev_map)
        adjustments = {k: new_map[k] for k in new_map if prev_map.get(k) != new_map.get(k)}
        if adjustments:
            st.write("#### Parameter Adjustments")
            st.json(adjustments)

# ---------------------------
# Go button
# ---------------------------
go = st.button("Generate Portfolio & Backtest", type="primary", use_container_width=True)

# ---------------------------
# Enhanced Tabs
# ---------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["📊 Rebalancing Plan", "✅ Current Portfolio", "📈 Performance", "🧭 Regime", "🔎 Changes", "🏥 Strategy Health", "🧪 Trust Checks"]
)

# Placeholders to reuse below
live_disp = None
live_raw = None
decision = "—"

strategy_cum_gross = None
strategy_cum_net = None
qqq_cum = None
hybrid_tno = None
mc_results = st.session_state.get("mc_results")

# ===========================
# Generate
# ===========================
if go:
    with st.spinner("Building enhanced monthly-locked portfolio and running backtest…"):
        try:
            # ---- Live portfolio (monthly lock + stability + trigger + sector caps + enhancements)
            live_disp, live_raw, decision = backend.generate_live_portfolio_isa_monthly(
                preset=preset,
                prev_portfolio=prev_portfolio,
                min_dollar_volume=st.session_state.get("min_dollar_volume", 0),
                as_of=date.today(),
                use_enhanced_features=st.session_state.get("use_enhanced_features", True),
            )
        except Exception as e:
            st.error(f"Portfolio generation failed: {e}")
            st.code(traceback.format_exc())

        try:
            # ---- Enhanced ISA Dynamic backtest (Hybrid150 default) from 2017-07-01
            strategy_cum_gross, strategy_cum_net, qqq_cum, hybrid_tno = backend.run_backtest_isa_dynamic(
                roundtrip_bps=st.session_state.get("roundtrip_bps", backend.ROUNDTRIP_BPS_DEFAULT),
                min_dollar_volume=st.session_state.get("min_dollar_volume", 0),
                show_net=st.session_state.get("show_net", True),
                start_date="2017-07-01",
                universe_choice=st.session_state.get("universe", "Hybrid Top150"),
                top_n=None if auto_optimize else preset["mom_topn"],
                name_cap=None if auto_optimize else float(
                    st.session_state.get("name_cap", preset.get("mom_cap", 0.25))
                ),
                sector_cap=None if auto_optimize else float(
                    st.session_state.get("sector_cap", preset.get("sector_cap", 0.30))
                ),
                stickiness_days=int(st.session_state.get("stickiness_days", preset.get("stability_days", 7))),
                mr_topn=preset["mr_topn"],
                mom_weight=None if auto_optimize else preset["mom_w"],
                mr_weight=None if auto_optimize else preset["mr_w"],
                use_enhanced_features=st.session_state.get("use_enhanced_features", True),
            )
        except Exception as e:
            st.warning(f"Backtest failed: {e}")
            st.code(traceback.format_exc())

        # save to session for Save button / persistence
        if live_raw is not None and not live_raw.empty:
            st.session_state.latest_portfolio = live_raw.copy()

        # Automatically run Monte Carlo projections using historical returns
        try:
            base_cum = strategy_cum_net if (strategy_cum_net is not None and show_net) else strategy_cum_gross
            if base_cum is not None:
                mc_returns = base_cum.pct_change().dropna()
                st.session_state.mc_results = backend.run_monte_carlo_projections(
                    mc_returns,
                    confidence_levels=[5, 25, 50, 75, 95],
                    seed=st.session_state.get("mc_seed")
                )
            else:
                st.session_state.mc_results = None
        except Exception as e:
            st.warning(f"Monte Carlo simulation failed: {e}")
            st.session_state.mc_results = {"error": str(e)}

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
                st.markdown(f"### 🔄 Rebalance (≥ {tol:.1%})")
                if signals["rebalance"]:
                    for t, old_w, new_w in signals["rebalance"]:
                        st.write(f"- **{t}**: {old_w:.2%} → **{new_w:.2%}**")
                else:
                    st.write("None")

        # Build rebalancing plan for download
        picks_rows = []
        for t in signals["sell"]:
            old_w = float(prev_portfolio.loc[t, "Weight"]) if prev_portfolio is not None and t in prev_portfolio.index else 0.0
            picks_rows.append({"Action": "Sell", "Ticker": t, "OldWeight": old_w, "NewWeight": 0.0})
        for t in signals["buy"]:
            new_w = float(live_raw.loc[t, "Weight"]) if t in live_raw.index else 0.0
            picks_rows.append({"Action": "Buy", "Ticker": t, "OldWeight": 0.0, "NewWeight": new_w})
        for t, old_w, new_w in signals["rebalance"]:
            picks_rows.append({"Action": "Rebalance", "Ticker": t, "OldWeight": old_w, "NewWeight": new_w})
        picks = pd.DataFrame(picks_rows)
        plan_csv = picks.to_csv(index=False).encode("utf-8")
        st.download_button("Download Plan", plan_csv, "rebalancing_plan.csv", "text/csv")

# ---------------------------
# Tab 2: Current Portfolio
# ---------------------------
with tab2:
    st.subheader("✅ Current Portfolio (Monthly-Locked)")

    # Determine if today is a rebalance day using the latest price index
    price_index = st.session_state.get("latest_price_index")
    is_rebalance_day = backend.is_rebalance_today(date.today(), price_index) if price_index is not None else False

    # -------------------------
    # Top panel: decision + table
    # -------------------------
    if live_disp is None or live_disp.empty:
        st.warning(decision if isinstance(decision, str) else "—")
        weights = None
    else:
        # Decision banner
        st.info(decision)

        # Feature badge
        if use_enhanced_features:
            st.success("🔬 Enhanced features active: Volatility-adjusted caps, regime awareness, signal decay")
        else:
            st.info("📊 Using standard features only")

        # Non-rebalance day notice
        if not is_rebalance_day:
            st.info("Preview only – portfolio not saved")

        st.dataframe(live_disp, use_container_width=True)

        # Extract current weights safely
        try:
            if "Weight" in live_raw.columns:
                weights = live_raw["Weight"].astype(float)
            else:
                # fallback if the column is named differently or the table is the source
                col_name = [c for c in live_disp.columns if c.lower() in ("weight", "weights")]
                weights = live_disp[col_name[0]].astype(float) if col_name else None
        except Exception:
            weights = None

        # -------------------------
        # Constraints & concentration metrics
        # -------------------------
        if weights is not None and len(weights) > 0:
            base_map = backend.get_sector_map(list(weights.index))
            sectors_map = backend.get_enhanced_sector_map(
                list(weights.index), base_map=base_map
            )
            group_caps = backend.build_group_caps(sectors_map)
            violations = backend.check_constraint_violations(
                weights,
                sectors_map,
                name_cap=float(
                    st.session_state.get("name_cap", preset.get("mom_cap", 0.25))
                ),
                sector_cap=float(
                    st.session_state.get("sector_cap", preset.get("sector_cap", 0.30))
                ),
                group_caps=group_caps,
            )

            max_w = weights.max()
            total_exp = weights.sum()
            c1, c2 = st.columns(2)
            c1.metric("Max Weight", f"{max_w:.2%}")
            c2.metric(
                "Total Equity Exposure",
                f"{total_exp:.2%}",
                help="Sector/name cap trimming leaves residual cash until final exposure scaling.",
            )

            if violations:
                st.markdown("**Constraint Violations**")
                st.table(pd.DataFrame({"Issue": violations}))
            else:
                st.success("No constraint violations")

        # Gated save action – only visible on rebalance days
        if is_rebalance_day and st.button("💾 Save Portfolio"):
            backend.save_portfolio_if_rebalance(live_raw, price_index)

    # -------------------------
    # Sector totals (current holdings)
    # -------------------------
    try:
        if weights is not None and len(weights) > 0:
            # Enhanced sector map with safe fallback to "Other"
            base_map = backend.get_sector_map(list(weights.index))
            enh_map = backend.get_enhanced_sector_map(
                list(weights.index), base_map=base_map
            )
            enh_map = {t: enh_map.get(t, "Other") for t in weights.index}

            sector_series = pd.Series(enh_map).reindex(weights.index)
            sector_totals = (
                weights.groupby(sector_series).sum().sort_values(ascending=False)
            )

            label = "Sector totals (post-caps)" if is_rebalance_day else "Sector totals (current holdings)"
            st.markdown(f"**{label}:**")
            st.dataframe(
                sector_totals.map(lambda x: f"{x:.2%}"),
                use_container_width=True
            )
    except Exception:
        pass

# ---------------------------
# Tab 3: Enhanced Performance
# ---------------------------
with tab3:
    st.subheader("📈 Backtest (since 2017-07-01)")
    if strategy_cum_gross is None or qqq_cum is None:
        st.info("Click Generate to see backtest results.")
    else:
        st.markdown("#### Key Performance (monthly series inferred)")

        # --- KPI table ---
        rows = []
        rows.append(
            backend.kpi_row(
                "Strategy (Gross)",
                strategy_cum_gross.pct_change(),
                turnover_series=hybrid_tno,
                avg_trade_size=AVG_TRADE_SIZE
            )
        )
        if strategy_cum_net is not None and show_net:
            rows.append(
                backend.kpi_row(
                    "Strategy (Net of costs)",
                    strategy_cum_net.pct_change(),
                    turnover_series=hybrid_tno,
                    avg_trade_size=AVG_TRADE_SIZE
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
            columns=["Model", "Freq", "CAGR", "Sharpe", "Sortino", "Calmar", "MaxDD", "Trades/yr", "Turnover/yr (0.5×L1)", "Equity x"]
        )
        st.dataframe(kdf, use_container_width=True)

        # --- Enhanced Equity curves ---
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curves
        axes[0,0].plot(strategy_cum_gross.index, strategy_cum_gross.values, label="Strategy (Gross)", linewidth=2)
        if strategy_cum_net is not None and show_net:
            axes[0,0].plot(strategy_cum_net.index, strategy_cum_net.values, label="Strategy (Net)", linestyle=":", linewidth=2)
        axes[0,0].plot(qqq_cum.index, qqq_cum.values, label="QQQ", linestyle="--", alpha=0.8)
        axes[0,0].set_yscale("log")
        axes[0,0].set_ylabel("Cumulative Growth (log)")
        axes[0,0].set_title("Equity Curves")
        axes[0,0].grid(True, ls="--", alpha=0.6)
        axes[0,0].legend()
        
        # Drawdowns
        base_series = strategy_cum_net if (strategy_cum_net is not None and show_net) else strategy_cum_gross
        dd_strategy = backend.drawdown(base_series)
        dd_qqq = backend.drawdown(qqq_cum)
        
        axes[0,1].fill_between(dd_strategy.index, dd_strategy.values, 0, alpha=0.3, color='red', label='Strategy')
        axes[0,1].fill_between(dd_qqq.index, dd_qqq.values, 0, alpha=0.3, color='green', label='QQQ')
        axes[0,1].set_ylabel("Drawdown")
        axes[0,1].set_title("Drawdown Comparison")
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Rolling Sharpe (12-month)
        if len(base_series.pct_change().dropna()) >= 12:
            rolling_returns = base_series.pct_change().dropna()
            rolling_sharpe = rolling_returns.rolling(12).apply(
                lambda x: (x.mean() * 12) / (x.std() * np.sqrt(12) + 1e-9)
            ).dropna()
            
            axes[1,0].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='purple')
            axes[1,0].axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, label='Sharpe = 1.0')
            axes[1,0].set_ylabel("Rolling 12M Sharpe")
            axes[1,0].set_title("Strategy Consistency")
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Turnover analysis
        if hybrid_tno is not None and not hybrid_tno.empty:
            axes[1,1].plot(hybrid_tno.index, hybrid_tno.values, color='brown', alpha=0.7)
            axes[1,1].set_ylabel("Monthly Turnover (0.5×L1)")
            axes[1,1].set_title("Trading Activity")
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

        st.caption(
            f"Trades/yr estimated from turnover (0.5×L1) assuming ~{AVG_TRADE_SIZE*100:.1f}% average single-leg trade."
        )

        # ========= Monthly Net Returns =========
        st.subheader("📅 Monthly Net Returns (%)")

        # Choose net if available (and user ticked 'show net'); else use gross
        base_cum = strategy_cum_net if (strategy_cum_net is not None and show_net) else strategy_cum_gross

        monthly_net = base_cum.pct_change().dropna() * 100  # %
        monthly_net = monthly_net[monthly_net.index >= pd.Timestamp("2017-07-01")]

        monthly_net_df = pd.DataFrame(monthly_net, columns=["Net Return (%)"])

        # Robust index formatting: handle Datetime/Period or pre-formatted strings
        idx = monthly_net_df.index
        if isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex)):
            monthly_net_df.index = idx.strftime("%Y-%m")
        else:
            parsed = pd.to_datetime(idx, errors="coerce")
            if parsed.notna().all():
                monthly_net_df.index = parsed.strftime("%Y-%m")
            else:
                monthly_net_df.index = pd.Index(idx.astype(str))

        st.dataframe(monthly_net_df.round(2), use_container_width=True)

        csv_bytes = monthly_net_df.round(4).to_csv().encode("utf-8")
        st.download_button(
            "⬇️ Download monthly net returns (CSV)",
            data=csv_bytes,
            file_name="monthly_net_returns.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Quick stats
        m = (monthly_net / 100.0).astype(float)  # decimal
        if len(m) > 0:
            ann_cagr = (1 + m).prod() ** (12 / len(m)) - 1
            ann_vol  = m.std() * (12 ** 0.5)
            sharpe   = (m.mean() * 12) / (m.std() * (12 ** 0.5) + 1e-9)
            hit_rate = (m > 0).mean()

            st.markdown(
                f"**Since 2017-07:** "
                f"CAGR: **{ann_cagr:.2%}** • Vol: **{ann_vol:.2%}** • Sharpe: **{sharpe:.2f}** • Hit-rate: **{hit_rate:.1%}**"
            )

        # ========= 12-Month Monte Carlo Projection =========
        st.subheader("🔮 12-Month Monte Carlo Projection")
        mc_results = st.session_state.get("mc_results")
        if not mc_results or "error" in mc_results:
            st.info("Monte Carlo projections unavailable.")
        else:
            percentiles = mc_results["percentiles"]
            pct_table = pd.DataFrame({
                "Percentile": [f"{int(k[1:])}th" for k in sorted(percentiles.keys(), key=lambda x: int(x[1:]))],
                "Return": [f"{percentiles[k]:.1%}" for k in sorted(percentiles.keys(), key=lambda x: int(x[1:]))]
            })
            st.table(pct_table)

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            scenarios = mc_results["scenarios"]
            ax2.hist(scenarios * 100.0, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            for level, pct_val in sorted(percentiles.items(), key=lambda x: int(x[0][1:])):
                ax2.axvline(pct_val * 100, linestyle="--", linewidth=1, label=f"{level[1:]}th: {pct_val:.1%}")
            ax2.axvline(0, color='red', linestyle='-', alpha=0.5, label='Break-even')
            ax2.set_xlabel("12-month return (%)")
            ax2.set_ylabel("Probability Density")
            ax2.set_title("Monte Carlo Return Distribution")
            ax2.legend()
            ax2.grid(True, ls="--", alpha=0.5)
            st.pyplot(fig2)


# ---------------------------
# Tab 4: Enhanced Regime
# ---------------------------
with tab4:
    st.subheader("🧭 Enhanced Market Regime Analysis")
    try:
        label, metrics = backend.get_market_regime()
        
        # Main regime display
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Regime", label)
        c2.metric("% Universe >200DMA", f"{metrics.get('universe_above_200dma', np.nan)*100:.1f}%")
        c3.metric("QQQ above 200DMA", "Yes" if metrics.get('qqq_above_200dma', 0.0) >= 1.0 else "No")

        c4, c5, c6 = st.columns(3)
        c4.metric("QQQ 10D Vol", f"{metrics.get('qqq_vol_10d', np.nan)*100:.2f}%")
        c5.metric("QQQ 50DMA slope (10D)", f"{metrics.get('qqq_50dma_slope_10d', np.nan)*100:.2f}%")
        c6.metric("6M Breadth", f"{metrics.get('breadth_pos_6m', np.nan)*100:.1f}%",
                 help="Percentage of stocks with positive 6-month returns")

        c7, c8 = st.columns(2)
        c7.metric("VIX 3M/1M", f"{metrics.get('vix_term_structure', np.nan):.2f}",
                 help="Ratio of 3M to 1M VIX; <1 can signal stress")
        c8.metric("HY OAS (%)", f"{metrics.get('hy_oas', np.nan):.2f}%",
                 help="High-yield option-adjusted spread")

        # Enhanced regime guidance
        breadth = float(metrics.get("breadth_pos_6m", np.nan))
        vol10   = float(metrics.get("qqq_vol_10d", np.nan))
        qqq_abv = bool(metrics.get("qqq_above_200dma", 0.0) >= 1.0)

        target_equity = 1.0
        headline = "Risk-On — full equity allocation recommended."
        box_func = st.success

        if ((not qqq_abv and (breadth < 0.35)) or (vol10 > 0.045 and not qqq_abv)):
            target_equity = 0.0
            headline = "Extreme Risk-Off — consider 100% cash."
            box_func = st.error
        elif (not qqq_abv) or (breadth < 0.45) or (vol10 > 0.035):
            target_equity = 0.50
            headline = "Risk-Off — scale to ~50% equity / 50% cash."
            box_func = st.warning
        elif breadth > 0.65 and vol10 < 0.025:
            target_equity = 1.1
            headline = "Strong Risk-On — consider modest leverage (110%)."
            box_func = st.success

        box_func(
            f"**Enhanced Regime Advice:** {headline}  \n"
            f"**Targets:** equity **{target_equity*100:.0f}%**, cash **{max(0, 100-target_equity*100):.0f}%**.  \n"
            f"_Context — Breadth (6m>0): {breadth:.0%} • 10-day vol: {vol10:.2%} • QQQ >200DMA: {'Yes' if qqq_abv else 'No'}_"
        )
        
        # Regime history visualization
        if use_enhanced_features:
            st.subheader("📊 Regime History")
            st.info("Enhanced regime tracking shows how market conditions evolve over time")
            
            # You could add a chart showing regime changes over time here
            # For now, just show the current status
        
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
            # --- figure out the two portfolios we're comparing ---
            base_df = backend.load_previous_portfolio()
            if base_df is None or base_df.empty:
                base_df = live_raw  # fallback: compare to itself (no-op)

            # --- build a price frame that DEFINITELY includes all tickers we need ---
            union_tickers = sorted(set(base_df.index) | set(live_raw.index))
            if union_tickers:
                end   = date.today().strftime("%Y-%m-%d")
                start = (date.today() - relativedelta(months=14)).strftime("%Y-%m-%d")
                px_union = backend.fetch_market_data(union_tickers, start, end)
            else:
                px_union = pd.DataFrame()

            # --- explain changes using the union price frame (robust to missing cols) ---
            expl = backend.explain_portfolio_changes(
                base_df,
                live_raw,
                px_union,  # <- robust frame built for the exact tickers
                backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]
            )

            if expl is None or expl.empty:
                st.info("No changes to explain.")
            else:
                # Enhanced formatting for display
                show = expl.copy()
                for col in show.columns:
                    if "Score" in col:
                        show[col] = show[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
                    if "Return" in col:
                        show[col] = show[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
                
                st.dataframe(show, use_container_width=True)
                
                # Enhanced insights
                if len(expl) > 0:
                    buys = expl[expl['Action'] == 'Buy']
                    sells = expl[expl['Action'] == 'Sell']
                    
                    if len(buys) > 0:
                        avg_mom_rank_buys = buys['Mom Rank'].mean()
                        st.info(f"📈 **New Positions:** Average momentum rank {avg_mom_rank_buys:.1f} - targeting higher momentum stocks")
                    
                    if len(sells) > 0:
                        st.info(f"📉 **Exits:** Removed {len(sells)} positions - likely due to momentum deterioration or stickiness requirements")

        except Exception as e:
            st.error(f"Explainability failed: {e}")
            st.code(traceback.format_exc())

# ---------------------------
# NEW Tab 6: Strategy Health Monitor
# ---------------------------
with tab6:
    st.subheader("🏥 Strategy Health Monitor")
    
    if strategy_cum_net is None and strategy_cum_gross is None:
        st.info("Generate backtest first to see strategy health metrics.")
    else:
        # Use net if available, otherwise gross
        perf_series = strategy_cum_net if strategy_cum_net is not None else strategy_cum_gross
        returns_series = perf_series.pct_change().dropna()
        
        # Get QQQ for comparison
        qqq_returns = qqq_cum.pct_change().dropna() if qqq_cum is not None else None
        
        # Calculate health metrics
        health_metrics = backend.get_strategy_health_metrics(returns_series, qqq_returns)
        
        # Display health dashboard
        st.markdown("#### 📊 Health Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_dd = health_metrics.get('current_drawdown', 0)
            dd_color = "inverse" if current_dd > -0.10 else "off"
            st.metric("Current Drawdown", f"{current_dd:.1%}", 
                     delta=None, delta_color=dd_color)
        
        with col2:
            recent_ret = health_metrics.get('recent_3m_return', 0)
            ret_color = "normal" if recent_ret > 0 else "inverse"
            st.metric("Recent 3M Avg Return", f"{recent_ret:.2%}", 
                     delta=None, delta_color=ret_color)
        
        with col3:
            recent_sharpe = health_metrics.get('recent_3m_sharpe', 0)
            sharpe_color = "normal" if recent_sharpe > 1.0 else "off"
            st.metric("Recent 3M Sharpe", f"{recent_sharpe:.2f}",
                     delta=None, delta_color=sharpe_color)
        
        with col4:
            correlation = health_metrics.get('benchmark_correlation', 0)
            corr_color = "inverse" if correlation > 0.85 else "normal"
            st.metric("QQQ Correlation", f"{correlation:.2f}",
                     delta=None, delta_color=corr_color,
                     help="High correlation reduces diversification benefit")
        
        # Health diagnostics
        st.markdown("#### 🔍 Health Diagnostics")
        
        issues = backend.diagnose_strategy_issues(returns_series, hybrid_tno)
        
        if issues and issues[0] != "No significant issues detected":
            st.markdown("**⚠️ Detected Issues:**")
            for issue in issues:
                st.warning(f"• {issue}")
        else:
            st.success("✅ No significant issues detected")
        
        # Detailed health analysis
        if len(returns_series) >= 12:
            st.markdown("#### 📈 Performance Trends")
            
            # Rolling 6-month performance
            if len(returns_series) >= 6:
                rolling_6m = returns_series.rolling(6).apply(
                    lambda x: ((1 + x).prod() ** (12/6)) - 1
                ).dropna()
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Rolling CAGR
                ax1.plot(rolling_6m.index, rolling_6m * 100, linewidth=2, color='blue')
                ax1.axhline(y=0, color='red', linestyle='-', alpha=0.3)
                ax1.axhline(y=15, color='green', linestyle='--', alpha=0.5, label='15% Target')
                ax1.set_ylabel("6-Month Rolling CAGR (%)")
                ax1.set_title("Strategy Performance Trend")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Hit rate trend
                rolling_hit_rate = returns_series.rolling(6).apply(lambda x: (x > 0).mean()).dropna()
                ax2.plot(rolling_hit_rate.index, rolling_hit_rate * 100, linewidth=2, color='green')
                ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% Baseline')
                ax2.set_ylabel("6-Month Hit Rate (%)")
                ax2.set_xlabel("Date")
                ax2.set_title("Strategy Consistency Trend")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
        
        # Strategy recommendations
        st.markdown("#### 💡 Health-Based Recommendations")
        
        current_dd = health_metrics.get('current_drawdown', 0)
        recent_perf = health_metrics.get('recent_3m_return', 0)
        vol_regime = health_metrics.get('vol_regime_ratio', 1.0)
        
        if current_dd < -0.20:
            st.error("🔴 **High Alert:** Large drawdown detected. Consider reducing position sizes or pausing new investments.")
        elif current_dd < -0.10:
            st.warning("🟡 **Caution:** Moderate drawdown. Monitor closely and ensure risk controls are working.")
        elif recent_perf > 0.02:  # >2% monthly average
            st.success("🟢 **Strong Performance:** Strategy performing well. Consider if position sizing is optimal.")
        else:
            st.info("🔵 **Normal:** Strategy within normal performance range. Continue monitoring.")
        
        # Enhancement recommendations
        if use_enhanced_features:
            st.info("🔬 **Enhanced Features Active:** Strategy using volatility-adjusted caps and regime awareness.")
        else:
            st.info("📊 **Standard Mode:** Consider enabling enhanced features for improved risk management.")

# ---------------------------
# Tab 7: Trust Checks (Signal • Construction • Health)
# ---------------------------
with tab7:
    st.subheader("🧪 Trust Checks")

    # Pull latest metrics / weights / turnover from earlier steps or session
    metrics = st.session_state.get("latest_metrics")  # optional: set this when you call assess_market_conditions()
    if metrics is None:
        try:
            # Fallback: compute quickly from current universe
            label, metrics = backend.get_market_regime()
            metrics["regime"] = label
        except Exception:
            metrics = {}

    weights_df = None
    try:
        # Prefer current generated portfolio
        if live_raw is not None and not live_raw.empty:
            weights_df = live_raw
        elif "latest_portfolio" in st.session_state:
            weights_df = st.session_state["latest_portfolio"]
    except Exception:
        pass

    turnover_series = None
    try:
        turnover_series = hybrid_tno
        # Also stash for later refresh
        st.session_state["hybrid_tno"] = hybrid_tno
    except Exception:
        turnover_series = st.session_state.get("hybrid_tno")

    name_cap   = float(st.session_state.get("name_cap", preset.get("mom_cap", 0.25)))
    sector_cap = float(st.session_state.get("sector_cap", preset.get("sector_cap", 0.30)))

    report = backend.run_trust_checks(
        weights_df=weights_df,
        metrics=metrics,
        turnover_series=turnover_series,
        name_cap=name_cap,
        sector_cap=sector_cap,
    )

    # ======= Summary tiles =======
    c1, c2, c3 = st.columns(3)
    c1.metric("Signal Alignment", "Pass" if report["signal"]["ok"] else "Check")
    c2.metric("Construction", "Pass" if report["construction"]["ok"] else "Check")
    c3.metric("Health", "Pass" if report["health"]["ok"] else "Check")

    st.markdown(f"**Composite Score:** `{report['score']}/3` (3 = strong trust)")

    # ======= Detail: Signal =======
    with st.expander("🔎 Signal Alignment – details"):
        st.write(f"Regime label: **{report['signal']['reason']}**")
        checks = report["signal"]["checks"]
        if checks:
            st.table(
                pd.DataFrame([checks]).T.rename(columns={0: "OK?"})
            )
        else:
            st.info("No metrics available.")

    # ======= Detail: Construction =======
    with st.expander("🏗️ Portfolio Construction – details"):
        st.json(report["construction"]["stats"])
        issues = report["construction"]["issues"]
        if issues:
            st.warning("Issues:")
            for it in issues:
                st.write(f"- {it}")
        else:
            st.success("No construction issues detected.")

    # ======= Detail: Health =======
    with st.expander("🏥 Health – details"):
        st.json(report["health"]["stats"])
        issues = report["health"]["issues"]
        if issues:
            st.warning("Issues:")
            for it in issues:
                st.write(f"- {it}")
        else:
            st.success("No health issues detected.")
