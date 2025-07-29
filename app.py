import streamlit as st
import backend
import traceback
from datetime import date
import numpy as np

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Momentum Portfolio V2")

# --- V2: Sidebar Controls ---
st.sidebar.header("⚙️ V2 Strategy Parameters")
momentum_window = st.sidebar.slider("Momentum Lookback (Months)", min_value=3, max_value=12, value=6, step=1)
top_n = st.sidebar.slider("Number of Stocks in Portfolio", min_value=5, max_value=20, value=10, step=1)
cap = st.sidebar.slider("Max Weight Cap per Stock", min_value=0.1, max_value=0.5, value=0.25, step=0.01, format="%.2f")
tol = st.sidebar.slider("Rebalancing Tolerance", min_value=0.005, max_value=0.05, value=0.01, step=0.005, format="%.3f")

# --- App UI ---
st.title("🚀 Momentum Portfolio Manager V2")
st.markdown("This application generates a portfolio and rebalancing plan based on the parameters you set in the sidebar.")

# --- Monthly Reminder ---
if 'last_run' not in st.session_state:
    st.session_state.last_run = None
today = date.today()
if today.day <= 5 and st.session_state.last_run != today:
    st.info("🔔 It's the start of the month—time for your portfolio review!")

# --- Main Logic ---
if st.button("Generate Portfolio & Rebalancing Plan", type="primary", use_container_width=True):
    st.session_state.last_run = today
    
    with st.spinner("Generating portfolio..."):
        try:
            prev_portfolio = backend.load_previous_portfolio()
            new_portfolio_display, new_portfolio_raw = backend.generate_live_portfolio(momentum_window, top_n, cap)

            if new_portfolio_raw is not None and not new_portfolio_raw.empty:
                st.subheader("📊 Rebalancing Plan")
                signals = backend.diff_portfolios(prev_portfolio, new_portfolio_raw, tol)

                if not any(s for s in signals.values()):
                    avg_shift = 0
                    if 'rebalance' in signals and signals['rebalance']:
                         avg_shift = np.mean([abs(new - old) for _, old, new in signals['rebalance']])
                    st.success(f"✅ No major rebalancing needed! Average weight shift was only {avg_shift:.2%}.")
                else:
                    cols = st.columns(3)
                    # Display rebalancing plan...
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
                
                st.subheader("✅ New Target Portfolio")
                st.dataframe(new_portfolio_display, use_container_width=True)
                
                # V2: Add explicit save button
                st.session_state.latest_portfolio = new_portfolio_raw

            else:
                st.error("Portfolio generation failed. Please see messages above.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.code(traceback.format_exc())

# V2: Logic for the save button
if 'latest_portfolio' in st.session_state and not st.session_state.latest_portfolio.empty:
    st.sidebar.header("💾 Save Portfolio")
    if st.sidebar.button("Save this portfolio for next month's comparison"):
        backend.save_portfolio_to_gist(st.session_state.latest_portfolio)
