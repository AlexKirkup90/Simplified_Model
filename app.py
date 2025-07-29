import streamlit as st
import backend  # Our final backend.py file
import traceback

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Optimal Momentum Portfolio Generator"
)

# --- App UI ---
st.title("🚀 Optimal Momentum Portfolio Generator")
st.markdown("""
This application generates a portfolio based on the **Momentum-Score (25% Cap)** strategy, which was the winning model from our rigorous backtesting process.

- **Strategy:** Ranks NASDAQ 100+ stocks by 6-month momentum and weights the top 10 based on their score, with a 25% cap on any single stock.
- **Result:** This approach aims to "let winners run" while maintaining a prudent level of diversification.

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
