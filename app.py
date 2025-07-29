import streamlit as st
import backend
import traceback
from datetime import date

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Optimal Momentum Portfolio Manager"
)

# --- App UI ---
st.title("🚀 Optimal Momentum Portfolio Manager")
st.markdown("This application generates a new momentum portfolio and shows you the exact trades needed to rebalance from your previous portfolio.")

# --- Monthly Reminder ---
if 'last_run' not in st.session_state:
    st.session_state.last_run = None
today = date.today()
if today.day <= 5 and st.session_state.last_run != today:
    st.info("🔔 It's the start of the month—time for your portfolio review!")

# --- Main Logic ---
if st.button("Generate Portfolio & Rebalancing Plan", type="primary", use_container_width=True):
    st.session_state.last_run = today
    
    with st.spinner("Generating portfolio... This may take a moment."):
        try:
            prev_portfolio = backend.load_previous_portfolio()
            new_portfolio_display, new_portfolio_raw = backend.generate_live_portfolio()

            if new_portfolio_raw is not None and not new_portfolio_raw.empty:
                st.subheader("📊 Rebalancing Plan")
                signals = backend.diff_portfolios(prev_portfolio, new_portfolio_raw)

                if not any(signals.values()):
                    st.success("✅ No changes needed—your portfolio is up to date!")
                else:
                    cols = st.columns(3)
                    with cols[0]:
                        if signals['sell']:
                            st.error("🔴 Sell Completely")
                            for ticker in signals['sell']:
                                st.markdown(f"- **{ticker}**")
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
            else:
                st.error("Portfolio generation failed. Please see messages above for details.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.code(traceback.format_exc())
