# backend.py
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ---- Shared strategy core ----
from strategy_core import (
    get_nasdaq_100_plus_tickers as _core_get_universe,
    fetch_market_data as _core_fetch_prices,
    cap_weights,
    build_momentum_weights,
    run_backtest_momentum,
    run_backtest_mean_reversion,
    combine_hybrid,
    cumulative_growth,
    get_performance_metrics,
)

# --- Gist API Constants for Portfolio Persistence ---
GIST_ID = st.secrets.get("GIST_ID")
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN")
GIST_API_URL = f"https://api.github.com/gists/{GIST_ID}" if GIST_ID else None
HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'} if GITHUB_TOKEN else {}
FILENAME = 'portfolio.json'

# --------------------------
# Cached wrappers (Streamlit)
# --------------------------
@st.cache_data(ttl=86400)
def get_nasdaq_100_plus_tickers() -> list:
    """Streamlit-cached universe fetcher."""
    # original extras list from your code
    extras = ['TSLA', 'SHOP', 'SNOW', 'PLTR', 'ETSY', 'RIVN', 'COIN']
    return _core_get_universe(extras=extras)

@st.cache_data(ttl=43200)
def fetch_market_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Streamlit-cached price fetcher."""
    try:
        # fetch a little earlier to ensure enough history for calculations (as before)
        fetch_start = (pd.to_datetime(start_date) - pd.DateOffset(months=14)).strftime('%Y-%m-%d')
        return _core_fetch_prices(tickers, start_date=fetch_start, end_date=end_date, price_field="Close")
    except Exception as e:
        st.error(f"Failed to download market data: {e}")
        return pd.DataFrame()

# --------------------------------
# Strategy + App integration logic
# --------------------------------
def generate_live_portfolio(momentum_window: int, top_n: int, cap: float):
    """Generates the live HYBRID portfolio but does not save it."""
    st.info("Fetching latest market data...")
    universe = get_nasdaq_100_plus_tickers()
    if not universe:
        return None, None

    start_date = (datetime.today() - relativedelta(months=momentum_window + 8)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    prices = fetch_market_data(universe, start_date, end_date)
    if prices.empty:
        return None, None

    # 1) Momentum (90%)
    st.info("Calculating momentum signals...")
    monthly_prices = prices.resample('ME').last()
    mom_weights, mom_scores = build_momentum_weights(
        monthly_prices=monthly_prices,
        lookback_m=momentum_window,
        top_n=top_n,
        cap=cap,
    )
    if mom_weights.empty:
        mom_weights = pd.Series(dtype=float)
    else:
        mom_weights = mom_weights * 0.90

    # 2) Mean-Reversion (10%) — same logic as before: 21d dip among names > 200d MA, equal weight top 5
    st.info("Calculating mean-reversion signals...")
    short_term_returns = prices.pct_change(21).iloc[-1]
    long_term_trend = prices.rolling(200).mean().iloc[-1]
    quality_stocks = prices.iloc[-1] > long_term_trend
    mr_candidates = short_term_returns[quality_stocks[quality_stocks].index].dropna()
    dip_stocks = mr_candidates.nsmallest(5)

    if dip_stocks.empty:
        mr_weights = pd.Series(dtype=float)
    else:
        mr_weights = pd.Series(1 / len(dip_stocks), index=dip_stocks.index) * 0.10

    # 3) Combine + normalize
    st.info("Combining strategies into final portfolio...")
    final_weights = mom_weights.add(mr_weights, fill_value=0)
    final_weights = final_weights / final_weights.sum() if final_weights.sum() > 0 else final_weights

    portfolio_df = pd.DataFrame({'Weight': final_weights}).sort_values('Weight', ascending=False)

    display_df = portfolio_df.copy()
    display_df['Weight'] = display_df['Weight'].map('{:.2%}'.format)

    return display_df, portfolio_df

def run_backtest_for_app(momentum_window: int, top_n: int, cap: float):
    """Runs a quick backtest for the HYBRID strategy to display in the app."""
    start_date = '2018-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')

    universe = get_nasdaq_100_plus_tickers()
    all_tickers = universe + ['QQQ']

    prices = fetch_market_data(all_tickers, start_date, end_date)
    if prices.empty or 'QQQ' not in prices.columns:
        return None, None

    # Only use tickers that downloaded successfully
    valid_universe = [t for t in universe if t in prices.columns]
    daily_prices = prices[valid_universe]
    qqq_prices = prices['QQQ']

    # Sleeve backtests (monthly series)
    mom_rets, _ = run_backtest_momentum(daily_prices, lookback_m=momentum_window, top_n=top_n, cap=cap)
    mr_rets, _ = run_backtest_mean_reversion(daily_prices)

    # Combine 90/10 and compute cumulative curves
    hybrid_rets = combine_hybrid(mom_rets, mr_rets, mom_weight=0.90, mr_weight=0.10)
    strategy_cumulative = cumulative_growth(hybrid_rets)

    qqq_cumulative = cumulative_growth(qqq_prices.resample('ME').last().pct_change())
    qqq_cumulative = qqq_cumulative.reindex(strategy_cumulative.index, method='ffill')

    return strategy_cumulative, qqq_cumulative

# --------------------------
# Gist persistence utilities
# --------------------------
def save_portfolio_to_gist(portfolio_df: pd.DataFrame):
    """Saves the provided portfolio DataFrame to the GitHub Gist."""
    if not GIST_API_URL or not HEADERS:
        st.sidebar.error("Gist credentials missing. Set GIST_ID and GITHUB_TOKEN in Streamlit secrets.")
        return
    try:
        json_content = portfolio_df.to_json(orient="index")
        data_to_save = {'files': {FILENAME: {'content': json_content}}}
        response = requests.patch(GIST_API_URL, headers=HEADERS, json=data_to_save)
        response.raise_for_status()
        st.sidebar.success("✅ Successfully saved portfolio to Gist.")
    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred while saving: {e}")

def load_previous_portfolio() -> pd.DataFrame:
    """Loads the last saved portfolio from the GitHub Gist."""
    if not GIST_API_URL or not HEADERS:
        return pd.DataFrame(columns=['Weight'])
    try:
        response = requests.get(GIST_API_URL, headers=HEADERS)
        response.raise_for_status()
        gist_content = response.json()['files'][FILENAME]['content']
        if not gist_content or gist_content == '{}':
            return pd.DataFrame(columns=['Weight'])
        return pd.read_json(gist_content, orient="index")
    except Exception:
        return pd.DataFrame(columns=['Weight'])

def diff_portfolios(prev_df: pd.DataFrame, curr_df: pd.DataFrame, tol: float) -> dict:
    """Compares two portfolios and generates buy, sell, and rebalance signals."""
    tickers_prev = set(prev_df.index)
    tickers_curr = set(curr_df.index)
    sells = tickers_prev - tickers_curr
    buys = tickers_curr - tickers_prev
    overlap = tickers_prev & tickers_curr
    rebalances = []
    for ticker in overlap:
        w_old = float(prev_df.at[ticker, 'Weight'])
        w_new = float(curr_df.at[ticker, 'Weight'])
        if abs(w_new - w_old) >= tol:
            rebalances.append((ticker, w_old, w_new))
    rebalances.sort(key=lambda x: abs(x[2] - x[1]), reverse=True)
    return {'sell': sorted(list(sells)), 'buy': sorted(list(buys)), 'rebalance': rebalances}
