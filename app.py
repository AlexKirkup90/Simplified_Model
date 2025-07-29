import streamlit as st
import backend  # Our final backend.py file
import traceback

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Momentum Portfolio Generator"
)

# --- App UI ---
st.title("🚀 Long-Only Momentum Portfolio Generator")
st.markdown("""
This application generates a portfolio based on our validated **long-only momentum** strategy, which proved to be the most robust model in backtesting.
- **Universe:** NASDAQ 100+
- **Strategy:** Each month, it identifies the top 10 stocks with the highest 6-month price momentum.
- **Allocation:** The model recommends an equal weight for each stock in the portfolio.
Click the button below to get the recommended portfolio for the upcoming month.
""")

# --- Button and Backend Logic ---
if st.button("Generate Live Portfolio", type="primary", use_container_width=True):
    with st.spinner("Generating portfolio... This may take a moment."):
        try:
            # Call the backend function to get the live portfolio
            portfolio_df = backend.generate_live_portfolio()

            # Display the result if successful
            if portfolio_df is not None and not portfolio_df.empty:
                st.subheader("✅ Recommended Portfolio Allocation")
                st.dataframe(portfolio_df, use_container_width=True)
                st.success("Portfolio generation complete. These are the recommended holdings for the next month.")
            else:
                # The backend will show specific warnings/errors. This is a fallback.
                st.error("Portfolio generation failed. Please see messages above for details.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.code(traceback.format_exc())
