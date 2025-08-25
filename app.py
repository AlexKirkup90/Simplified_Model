# app.py - Enhanced Hybrid Momentum Portfolio (ISA-Dynamic) with Strategy Health Monitoring
import traceback
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import sys, os
sys.path.append(os.path.dirname(__file__))
import backend  # all logic lives here

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

# Stickiness & sector cap (overrides)
stickiness_days = st.sidebar.slider(
    "Stickiness (days in top cohort)",
    3, 15,
    preset.get("stability_days", 7),
    1
)

# --- Caps (UI in %; backend uses fractions) ---
name_cap_pct = st.sidebar.slider(
    "Single-name cap (%)",
    min_value=20, max_value=50, value=int(preset.get("mom_cap", 0.25) * 100),
    step=5
)
sector_cap_pct = st.sidebar.slider(
    "Sector Cap (max % per sector)",
    min_value=10, max_value=50, value=int(preset.get("sector_cap", 0.30) * 100),
    step=5
)

# Convert to fractions for backend and stash in session
name_cap = name_cap_pct / 100.0
sector_cap = sector_cap_pct / 100.0
st.session_state["name_cap"] = name_cap
st.session_state["sector_cap"] = sector_cap

# Optional helper labels
st.sidebar.caption(f"Single-name cap: **{name_cap_pct}%**")
st.sidebar.caption(f"Sector cap: **{sector_cap_pct}%**")

# Persist to session so backend reads the same values
st.session_state["stickiness_days"] = int(stickiness_days)
st.session_state["use_enhanced_features"] = use_enhanced_features

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
# Enhanced Tabs
# ---------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📊 Rebalancing Plan", "✅ Current Portfolio", "📈 Performance", "🧭 Regime", "🔎 Changes", "🏥 Strategy Health"]
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
    with st.spinner("Building enhanced monthly-locked portfolio and running backtest…"):
        try:
            # ---- Live portfolio (monthly lock + stability + trigger + sector caps + enhancements)
            live_disp, live_raw, decision = backend.generate_live_portfolio_isa_monthly(
                preset=preset,
                prev_portfolio=prev_portfolio,
                min_dollar_volume=min_dollar_volume
            )
        except Exception as e:
            st.error(f"Portfolio generation failed: {e}")
            st.code(traceback.format_exc())

        try:
            # ---- Enhanced ISA Dynamic backtest (Hybrid150 default) from 2017-07-01
            strategy_cum_gross, strategy_cum_net, qqq_cum, hybrid_tno = backend.run_backtest_isa_dynamic(
                roundtrip_bps=roundtrip_bps,
                min_dollar_volume=min_dollar_volume,
                show_net=show_net,
                start_date="2017-07-01",
                universe_choice=universe_choice,
                top_n=preset["mom_topn"],
                name_cap=float(name_cap),
                sector_cap=sector_cap,
                stickiness_days=stickiness_days,
                mr_topn=preset["mr_topn"],
                mom_weight=preset["mom_w"],
                mr_weight=preset["mr_w"],
                use_enhanced_features=use_enhanced_features
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
                st.markdown(f"### 🔄 Rebalance (≥ {tol:.1%})")
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
        # Enhanced decision display with regime context
        st.info(decision)
        
        # Show enhancement status
        if use_enhanced_features:
            st.success("🔬 Enhanced features active: Volatility-adjusted caps, regime awareness, signal decay")
        else:
            st.info("📊 Using standard features only")
        
        st.dataframe(live_disp, use_container_width=True)

        # Enhanced bar chart with more details
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            live_sorted = live_raw.sort_values("Weight", ascending=False)
            
            # Weight distribution
            bars = ax1.bar(live_sorted.index, live_sorted["Weight"].values)
            ax1.set_ylabel("Weight")
            ax1.set_title("Portfolio Weights")
            ax1.tick_params(axis='x', rotation=45)
            
            # Add cap line
            ax1.axhline(y=name_cap, color='red', linestyle='--', alpha=0.7, label=f'Name Cap ({name_cap:.0%})')
            ax1.legend()
            
            # Position concentration
            cumsum = live_sorted["Weight"].cumsum()
            ax2.plot(range(1, len(cumsum) + 1), cumsum.values, marker='o')
            ax2.set_xlabel("Number of Positions")
            ax2.set_ylabel("Cumulative Weight")
            ax2.set_title("Position Concentration")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        except Exception:
            pass

        # Enhanced save with snapshot recording
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save this portfolio for next month"):
                backend.save_portfolio_to_gist(live_raw)
                st.success("Saved to Gist.")
        with col2:
            if st.button("📸 Record live snapshot"):
                result = backend.record_live_snapshot(live_raw, note="Manual snapshot")
                if result.get("ok"):
                    st.success(f"✅ Snapshot recorded! Strategy: {result['strat_ret']:.2%}, QQQ: {result['qqq_ret']:.2%}")
                else:
                    st.error(f"❌ Snapshot failed: {result.get('msg', 'Unknown error')}")

# ---------------------------
# Tab 3: Enhanced Performance
# ---------------------------
with tab3:
    st.subheader("📈 Backtest (since 2017-07-01)")
    if strategy_cum_gross is None or qqq_cum is None:
        st.info("Click Generate to see backtest results.")
    else:
        st.markdown("#### Key Performance (monthly series inferred)")

        AVG_TRADE_SIZE = 0.02  # estimate trades/yr from turnover

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
            columns=["Model", "Freq", "CAGR", "Sharpe", "Sortino", "Calmar", "MaxDD", "Trades/yr", "Turnover/yr", "Equity x"]
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
            axes[1,1].set_ylabel("Monthly Turnover")
            axes[1,1].set_title("Trading Activity")
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

        st.caption("Trades/yr estimated from turnover assuming ~2% average single-leg trade (adjust in code with AVG_TRADE_SIZE).")

        # ========= Monthly Net Returns =========
        st.subheader("📅 Monthly Net Returns (%)")

        # Choose net if available (and user ticked 'show net'); else use gross
        base_cum = strategy_cum_net if (strategy_cum_net is not None and show_net) else strategy_cum_gross

        monthly_net = base_cum.pct_change().dropna() * 100  # %
        monthly_net = monthly_net[monthly_net.index >= pd.Timestamp("2017-07-01")]

        monthly_net_df = pd.DataFrame(monthly_net, columns=["Net Return (%)"])
        monthly_net_df.index = monthly_net_df.index.strftime("%Y-%m")
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

        # ========= Enhanced 12-Month Monte Carlo (bootstrap) =========
        st.subheader("🔮 Enhanced 12-Month Monte Carlo (net returns)")
        if len(m) < 6:
            st.info("Not enough monthly history to run the simulation.")
        else:
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                n_trials = st.slider("Simulations", 1000, 10000, 5000, 500)
            with col_b:
                block = st.slider("Block size (months)", 1, 6, 3, 1,
                                  help="Use 1 for IID bootstrap; >1 keeps short-term clustering.")
            with col_c:
                seed = st.number_input("Random seed", value=42, step=1)
            with col_d:
                confidence_levels = st.multiselect("Confidence levels", [5, 10, 25, 50, 75, 90, 95], default=[10, 50, 90])

            # Enhanced Monte Carlo using backend function
            mc_results = backend.run_monte_carlo_projections(
                m, n_scenarios=n_trials, horizon_months=12, 
                confidence_levels=confidence_levels, block_size=block
            )
            
            if "error" not in mc_results:
                percentiles = mc_results['percentiles']
                
                # Display key statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Median Return", f"{percentiles.get('p50', 0):.1%}")
                with col2:
                    st.metric("Probability Positive", f"{mc_results['prob_positive']:.1%}")
                with col3:
                    st.metric("Probability >10%", f"{mc_results['prob_beat_10pct']:.1%}")
                
                # Display confidence intervals
                st.markdown("**12-month projected returns:**")
                for level in sorted(confidence_levels):
                    pct_key = f'p{level}'
                    if pct_key in percentiles:
                        st.write(f"• {level}th percentile: **{percentiles[pct_key]:.1%}**")

                # Enhanced histogram
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                scenarios = mc_results['scenarios']
                ax2.hist(scenarios * 100.0, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
                
                # Add percentile lines
                for level in confidence_levels:
                    if level in [10, 50, 90]:  # Show main percentiles
                        pct_val = percentiles.get(f'p{level}', 0)
                        ax2.axvline(pct_val*100, linestyle="--", linewidth=2, 
                                   label=f'{level}th pct: {pct_val:.1%}')
                
                ax2.axvline(0, color='red', linestyle='-', alpha=0.5, label='Break-even')
                ax2.set_xlabel("12-month return (%)")
                ax2.set_ylabel("Probability Density")
                ax2.set_title("Monte Carlo Return Distribution")
                ax2.legend()
                ax2.grid(True, ls="--", alpha=0.5)
                st.pyplot(fig2)

                      # === Enhanced TL;DR summary ===
                st.markdown("##### TL;DR for the next 12 months")

                start_amount = st.number_input(
                    "Show results for a starting amount (£)",
                    min_value=100, max_value=1_000_000, step=100, value=1000
                )

                def money(x: float) -> str:
                    return f"£{x:,.0f}"

                # Safe pulls with sensible fallbacks
                scenarios = mc_results.get('scenarios', np.array([]))
                median_ret = float(percentiles.get('p50', mc_results.get('mean_return', 0.0)))
                p10_ret    = float(percentiles.get('p10', np.percentile(scenarios, 10))) if scenarios.size else 0.0
                p90_ret    = float(percentiles.get('p90', np.percentile(scenarios, 90))) if scenarios.size else 0.0
                p05_ret    = float(percentiles.get('p5',  np.percentile(scenarios, 5)))  if scenarios.size else 0.0
                downside   = float(mc_results.get('downside_risk', 0.0))
                prob_pos   = float(mc_results.get('prob_positive', 0.0))

                st.markdown(f"""
**Expected Outcomes for £{start_amount:,}:**
- **Median outcome:** **{median_ret*100:.1f}%** → **{money(start_amount*(1+median_ret))}**  
- **Typical range (10th–90th pct):** **{p10_ret*100:.1f}%** to **{p90_ret*100:.1f}%**  
  → **{money(start_amount*(1+p10_ret))}** to **{money(start_amount*(1+p90_ret))}**  
- **Downside scenario (5th pct):** **{p05_ret*100:.1f}%** → **{money(start_amount*(1+p05_ret))}**  
- **Chance of positive year:** **{prob_pos*100:.1f}%**  
- **Average loss in bad scenarios:** **{downside*100:.1f}%**  
""")

                st.info(
                    f"In plain English: the distribution is skewed positive. "
                    f"Most paths are up (median ≈ {median_ret*100:.1f}%), "
                    f"but there’s still real downside risk (~{prob_pos*100:.1f}% chance of a positive year). "
                    f"Size positions accordingly."
                )

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
                end   = datetime.today().strftime("%Y-%m-%d")
                start = (datetime.today() - relativedelta(months=14)).strftime("%Y-%m-%d")
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

print("\n✅ Enhanced app.py created with Strategy Health monitoring!")
