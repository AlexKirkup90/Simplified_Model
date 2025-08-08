import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import backend
from datetime import datetime

# --- App Config ---
st.set_page_config(page_title="ISA Dynamic Portfolio", layout="wide")
st.title("📈 ISA Dynamic Portfolio — Monthly Rebalancing")

# --- Load Data ---
st.sidebar.header("Portfolio Settings")
trigger_threshold = st.sidebar.slider("Trigger Threshold", 0.5, 1.0, 0.75, 0.05)
mom_lb = st.sidebar.number_input("Momentum Lookback (days)", 10, 60, 15)
mom_topn = st.sidebar.number_input("Momentum Top N", 1, 20, 8)
mom_cap = st.sidebar.slider("Momentum Cap (%)", 5, 50, 25) / 100
mr_lb = st.sidebar.number_input("Mean Reversion Lookback (days)", 5, 60, 21)
mr_topn = st.sidebar.number_input("Mean Reversion Top N", 1, 10, 3)
mr_ma = st.sidebar.number_input("Mean Reversion Long MA (days)", 50, 300, 200)
mom_w = st.sidebar.slider("Momentum Weight (%)", 0, 100, 85) / 100
mr_w = 1 - mom_w

# Load universe prices from backend
univ_prices = backend.load_price_data()

# Run monthly-locked model
isa_rets, isa_tno, isa_trades, isa_portfolio = backend.run_dynamic_with_log(
    univ_prices,
    mom_lb=mom_lb,
    mom_topn=mom_topn,
    mom_cap=mom_cap,
    mr_lb=mr_lb,
    mr_topn=mr_topn,
    mr_ma=mr_ma,
    mom_w=mom_w,
    mr_w=mr_w,
    trigger_threshold=trigger_threshold,
    rebalance_freq="M"  # 🔒 Monthly
)

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Performance", "💼 Current Portfolio", "💰 Cost / Liquidity", "📜 Trade Log"]
)

# --- Tab 1: Performance ---
with tab1:
    eq_curve = backend.equity_curve(isa_rets)
    dd_series = backend.drawdown_series(eq_curve)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(eq_curve, label="ISA Dynamic Portfolio", color='blue')
    axes[0].set_ylabel("Equity (log)")
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(dd_series, color='red')
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True)
    st.pyplot(fig)

    # Performance table
    table = PrettyTable()
    table.field_names = ["Model", "Freq", "CAGR", "Sharpe", "Sortino", "Calmar", "MaxDD", "Trades/yr", "Turnover/yr", "Equity Multiple"]
    backend.perf_summary(
        "ISA Dynamic", isa_rets, isa_trades, isa_tno, freq="Monthly (12py)", table=table
    )
    st.text(table)

# --- Tab 2: Current Portfolio ---
with tab2:
    st.subheader("Current Holdings")
    st.dataframe(isa_portfolio)

    # Portfolio diff from last month
    prev_portfolio = backend.load_previous_portfolio()
    if prev_portfolio is not None:
        diff_df = backend.diff_portfolios(prev_portfolio, isa_portfolio)
        st.subheader("Changes vs Last Month")
        st.dataframe(diff_df)

    backend.save_current_portfolio(isa_portfolio)

# --- Tab 3: Cost / Liquidity ---
with tab3:
    st.subheader("Cost & Liquidity Impact")
    cost_df = backend.estimate_cost_liquidity(isa_portfolio)
    st.dataframe(cost_df)

# --- Tab 4: Trade Log ---
with tab4:
    st.subheader("Trade Log (Monthly Rebalance Points)")
    st.dataframe(isa_trades)

# --- Auto-log results ---
backend.log_run(
    model_name="ISA Dynamic Monthly",
    params={
        "mom_lb": mom_lb, "mom_topn": mom_topn, "mom_cap": mom_cap,
        "mr_lb": mr_lb, "mr_topn": mr_topn, "mr_ma": mr_ma,
        "mom_w": mom_w, "mr_w": mr_w,
        "trigger_threshold": trigger_threshold
    },
    portfolio=isa_portfolio
)
