# app.py
import streamlit as st
import traceback
import yfinance as yf
import matplotlib.pyplot as plt

import backend  # our production module above

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Hybrid Momentum Portfolio")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Strategy Parameters (Momentum Sleeve)")
momentum_window = st.sidebar.slider("Momentum Lookback (Months)", 3, 12, 6, 1)
top_n = st.sidebar.slider("Number of Stocks in Portfolio", 5, 20, 15, 1)
cap = st.sidebar.slider("Max Weight Cap per Stock", 0.1, 0.5, 0.25, 0.01, format="%.2f")
tol = st.sidebar.slider("Rebalancing Tolerance", 0.005, 0.05, 0.01, 0.005, format="%.3f")

# --- App UI ---
st.title("🚀 Hybrid Momentum Portfolio Manager")
st.markdown("This application generates a portfolio based on a **90/10 Hybrid Strategy**: 90% allocated to a core Momentum strategy and 10% to a complementary Mean-Reversion strategy.")

# --- Main Logic ---
if st.button("Generate Portfolio & Rebalancing Plan", type="primary", use_container_width=True):

    with st.spinner("Generating portfolio and running backtest..."):
        try:
            # 1) Generate Live Portfolio & Load Previous
            prev_portfolio = backend.load_previous_portfolio()
            new_portfolio_display, new_portfolio_raw = backend.generate_live_portfolio(momentum_window, top_n, cap)

            # 2) Run On-the-Fly Backtest
            strategy_returns_cum, qqq_returns_cum = backend.run_backtest_for_app(momentum_window, top_n, cap)

            # 3) Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Rebalancing Plan", "✅ New Portfolio", "📈 Performance", "📉 Market Volatility"])

            if new_portfolio_raw is not None and not new_portfolio_raw.empty:
                with tab1:
                    st.subheader("Rebalancing Plan")
                    signals = backend.diff_portfolios(prev_portfolio, new_portfolio_raw, tol)
                    if not any(s for s in signals.values()):
                        st.success("✅ No major rebalancing needed!")
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

                with tab2:
                    st.subheader("New Target Portfolio (90/10 Hybrid)")
                    st.dataframe(new_portfolio_display, use_container_width=True)
                    st.subheader("Portfolio Weights Visualized")
                    st.bar_chart(new_portfolio_raw['Weight'])

                with tab3:
                    st.subheader(f"Backtest: Hybrid Strategy vs. QQQ (Since 2018)")
                    if strategy_returns_cum is not None and qqq_returns_cum is not None:
                        # KPIs (monthly pct_change)
                        strat_metrics = backend.get_performance_metrics(strategy_returns_cum.pct_change())
                        qqq_metrics = backend.get_performance_metrics(qqq_returns_cum.pct_change())

                        st.markdown("##### Key Performance Indicators")

                        cols = st.columns(2)
                        with cols[0]:
                            st.markdown("###### Hybrid Strategy")
                            st.metric("Annual Return", strat_metrics['Annual Return'])
                            st.metric("Sharpe Ratio", strat_metrics['Sharpe Ratio'])
                            st.metric("Sortino Ratio", strat_metrics['Sortino Ratio'], help="Measures risk-adjusted return, but only penalizes for downside volatility.")
                            st.metric("Calmar Ratio", strat_metrics['Calmar Ratio'], help="Annual Return / Max Drawdown. A high value is desirable.")
                            st.metric("Max Drawdown", strat_metrics['Max Drawdown'])

                        with cols[1]:
                            st.markdown("###### QQQ Benchmark")
                            st.metric("Annual Return", qqq_metrics['Annual Return'])
                            st.metric("Sharpe Ratio", qqq_metrics['Sharpe Ratio'])
                            st.metric("Sortino Ratio", qqq_metrics['Sortino Ratio'])
                            st.metric("Calmar Ratio", qqq_metrics['Calmar Ratio'])
                            st.metric("Max Drawdown", qqq_metrics['Max Drawdown'])

                        st.divider()

                        # Chart
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(strategy_returns_cum.index, strategy_returns_cum.values, label='Hybrid Strategy')
                        ax.plot(qqq_returns_cum.index, qqq_returns_cum.values, label='QQQ Benchmark', linestyle='--')
                        ax.set_ylabel("Cumulative Growth")
                        ax.set_yscale('log')
                        ax.legend()
                        ax.grid(True, which="both", ls="--")
                        st.pyplot(fig)
                    else:
                        st.warning("Could not generate backtest chart.")

                with tab4:
                    st.subheader("QQQ 10-Day Rolling Volatility")
                    try:
                        qqq_vol_data = yf.download("QQQ", period="60d", auto_adjust=True, progress=False)
                        qqq_close_prices = qqq_vol_data['Close']
                        vol_series = qqq_close_prices.pct_change().rolling(window=10).std().dropna()
                        current_vol = float(vol_series.iloc[-1])

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

                st.session_state.latest_portfolio = new_portfolio_raw
            else:
                st.error("Portfolio generation failed. Please check the parameters and try again.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.code(traceback.format_exc())

# Save button
if 'latest_portfolio' in st.session_state and not st.session_state.latest_portfolio.empty:
    st.sidebar.header("💾 Save Portfolio")
    if st.sidebar.button("Save this portfolio for next month"):
        backend.save_portfolio_to_gist(st.session_state.latest_portfolio)
