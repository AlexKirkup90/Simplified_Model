import streamlit as st
import backend
import traceback
from datetime import date
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Momentum Portfolio V3")

# --- V3: Sidebar Controls ---
st.sidebar.header("⚙️ V3 Strategy Parameters")
momentum_window = st.sidebar.slider("Momentum Lookback (Months)", 3, 12, 6, 1)
top_n = st.sidebar.slider("Number of Stocks in Portfolio", 5, 20, 15, 1)
cap = st.sidebar.slider("Max Weight Cap per Stock", 0.1, 0.5, 0.25, 0.01, format="%.2f")
tol = st.sidebar.slider("Rebalancing Tolerance", 0.005, 0.05, 0.01, 0.005, format="%.3f")

# --- App UI ---
st.title("🚀 Momentum Portfolio Manager V3")
st.markdown("This application generates a portfolio and rebalancing plan based on the parameters you set in the sidebar. It also provides an on-the-fly backtest to visualize the historical performance of your selected strategy.")

# --- Main Logic ---
if st.button("Generate Portfolio & Rebalancing Plan", type="primary", use_container_width=True):
    
    with st.spinner("Generating portfolio and running backtest..."):
        try:
            # --- Step 1: Generate Live Portfolio & Load Previous ---
            prev_portfolio = backend.load_previous_portfolio()
            new_portfolio_display, new_portfolio_raw = backend.generate_live_portfolio(momentum_window, top_n, cap)
            
            # --- Step 2: Run On-the-Fly Backtest ---
            strategy_perf, qqq_perf = backend.run_backtest_for_app(momentum_window, top_n, cap)

            # --- Step 3: Create Tabbed Layout ---
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Rebalancing Plan", "✅ New Portfolio", "📈 Performance Chart", "📉 Market Volatility"])

            if new_portfolio_raw is not None and not new_portfolio_raw.empty:
                with tab1: # Rebalancing Plan
                    st.subheader("Rebalancing Plan")
                    signals = backend.diff_portfolios(prev_portfolio, new_portfolio_raw, tol)
                    if not any(s for s in signals.values()):
                         st.success("✅ No major rebalancing needed!")
                    else:
                        cols = st.columns(3)
                        # Display signals...
                        with cols[0]:
                            if signals['sell']:
                                st.error("🔴 Sell Completely")
                                for ticker in signals['sell']: st.markdown(f"- **{ticker}**")
                        with cols[1]:
                            if signals['buy']:
                                st.success("🟢 New Buys")
                                for ticker in signals['buy']:
                                    weight = new_portfolio_raw.at[ticker, 'Weight']
                                    st.markdown(f"- **{ticker}** (Target: {weight:.2%})")
                        with cols[2]:
                            if signals['rebalance']:
                                st.info("🔄 Rebalance")
                                for ticker, old_w, new_w in signals['rebalance']:
                                    st.markdown(f"- **{ticker}**: {old_w:.2%} → **{new_w:.2%}**")

                with tab2: # New Portfolio
                    st.subheader("New Target Portfolio")
                    st.dataframe(new_portfolio_display, use_container_width=True)
                    st.subheader("Portfolio Weights Visualized")
                    st.bar_chart(new_portfolio_raw['Weight'])

                with tab3: # Performance Chart
                    st.subheader(f"Backtest: Your Strategy vs. QQQ (Since 2018)")
                    if strategy_perf is not None and qqq_perf is not None:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(strategy_perf.index, strategy_perf.values, label='Your Strategy')
                        ax.plot(qqq_perf.index, qqq_perf.values, label='QQQ Benchmark', linestyle='--')
                        ax.set_ylabel("Cumulative Growth")
                        ax.set_yscale('log')
                        ax.legend()
                        ax.grid(True, which="both", ls="--")
                        st.pyplot(fig)
                    else:
                        st.warning("Could not generate backtest chart.")

                with tab4: # Market Volatility
                    st.subheader("QQQ 10-Day Rolling Volatility")
                    try:
                        qqq_vol_data = yf.download("QQQ", period="60d", auto_adjust=True)['Close']
                        vol_series = qqq_vol_data.pct_change().rolling(window=10).std().dropna()
                        current_vol = vol_series.iloc[-1]
                        if current_vol > 0.025:
                            st.warning(f"High market volatility detected: {current_vol:.2%}")
                        else:
                            st.success(f"Market volatility is normal: {current_vol:.2%}")
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(vol_series.index, vol_series.values, label='10-Day Volatility')
                        ax.axhline(y=0.025, color='r', linestyle='--', label='High Volatility Threshold (2.5%)')
                        ax.legend()
                        ax.grid(True, which="both", ls="--")
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Could not fetch volatility data: {e}")

                # V3: Add explicit save button to sidebar
                st.session_state.latest_portfolio = new_portfolio_raw
            else:
                st.error("Portfolio generation failed. Please see messages above.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.code(traceback.format_exc())

# V3: Logic for the save button
if 'latest_portfolio' in st.session_state and not st.session_state.latest_portfolio.empty:
    st.sidebar.header("💾 Save Portfolio")
    if st.sidebar.button("Save this portfolio for next month"):
        backend.save_portfolio_to_gist(st.session_state.latest_portfolio)
