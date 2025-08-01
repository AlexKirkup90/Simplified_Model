import pandas as pd
import numpy as np
import yfinance as yf
import requests
import streamlit as st
import quantstats as qs
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- Gist API Constants for Portfolio Persistence ---
GIST_ID = st.secrets.get("GIST_ID")
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN")
GIST_API_URL = f"https://api.github.com/gists/{GIST_ID}"
HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'}
FILENAME = 'portfolio.json'

# --- Data Fetching ---
@st.cache_data(ttl=86400)
def get_nasdaq_100_plus_tickers() -> list:
    """Fetches a list of NASDAQ-100 and other major tech tickers."""
    try:
        payload = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        nasdaq_100 = payload[4]['Ticker'].tolist()
        extras = ['TSLA', 'SHOP', 'SNOW', 'PLTR', 'ETSY', 'RIVN', 'COIN']
        if 'SQ' in extras: extras.remove('SQ') # SQ was acquired
        full_list = sorted(list(set(nasdaq_100 + extras)))
        return full_list
    except Exception as e:
        st.error(f"Failed to fetch ticker list: {e}")
        return []

@st.cache_data(ttl=43200)
def fetch_market_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches daily 'Close' price data for a list of tickers."""
    try:
        # Fetch data starting a bit earlier to ensure enough history for calculations
        fetch_start = (pd.to_datetime(start_date) - pd.DateOffset(months=14)).strftime('%Y-%m-%d')
        data = yf.download(tickers, start=fetch_start, end=end_date, auto_adjust=True, progress=False)['Close']
        return data.dropna(axis=1, how='all')
    except Exception as e:
        st.error(f"Failed to download market data: {e}")
        return pd.DataFrame()

# --- Core Logic & Analytics ---
def cap_weights(weights: pd.Series, cap: float = 0.25) -> pd.Series:
    """Applies an iterative 'waterfall' cap to portfolio weights."""
    w = weights.copy()
    for _ in range(100): # Safety break
        over_cap = w > cap
        if not over_cap.any(): return w
        excess_weight = (w[over_cap] - cap).sum()
        w[over_cap] = cap
        under_cap = ~over_cap
        if w[under_cap].sum() > 0:
            w[under_cap] += w[under_cap] / w[under_cap].sum() * excess_weight
    return w

def get_performance_metrics(returns: pd.Series) -> dict:
    """Calculates a dictionary of advanced performance metrics using quantstats."""
    if not isinstance(returns, pd.Series) or returns.empty or returns.isnull().all() or len(returns) < 2:
        return {'Annual Return': 'N/A', 'Sharpe Ratio': 'N/A', 'Max Drawdown': 'N/A', 'Sortino Ratio': 'N/A', 'Calmar Ratio': 'N/A'}
    
    # QuantStats requires returns as a simple Series, not percentages
    # Ensure monthly frequency is set for correct annualization
    returns.index = returns.index.to_period('M')
    
    return {
        'Annual Return': f"{qs.stats.cagr(returns, period='monthly'):.2%}",
        'Sharpe Ratio': f"{qs.stats.sharpe(returns, period='monthly'):.2f}",
        'Sortino Ratio': f"{qs.stats.sortino(returns, period='monthly'):.2f}",
        'Calmar Ratio': f"{qs.stats.calmar(returns):.2f}",
        'Max Drawdown': f"{qs.stats.max_drawdown(returns):.2%}"
    }

# --- Strategy Implementation Functions ---
def run_backtest_gross(daily_prices: pd.DataFrame, momentum_window: int = 6, top_n: int = 15, cap: float = 0.25) -> pd.Series:
    """Runs the primary MOMENTUM strategy."""
    monthly_prices = daily_prices.resample('ME').last()
    future_returns = monthly_prices.pct_change().shift(-1)
    momentum = monthly_prices.pct_change(periods=momentum_window).shift(1)
    portfolio_returns = pd.Series(index=momentum.index, dtype=float)

    for month in momentum.index:
        momentum_scores = momentum.loc[month].dropna()
        positive_momentum = momentum_scores[momentum_scores > 0]
        if positive_momentum.empty:
            portfolio_returns.loc[month] = 0
            continue
        
        top_performers = positive_momentum.nlargest(top_n)
        raw_weights = top_performers / top_performers.sum()
        capped_weights = cap_weights(raw_weights, cap=cap)
        final_weights = capped_weights / capped_weights.sum()
        
        valid_tickers = final_weights.index.intersection(future_returns.columns)
        month_return = (future_returns.loc[month, valid_tickers] * final_weights[valid_tickers]).sum()
        portfolio_returns.loc[month] = month_return
    return portfolio_returns.fillna(0)

def run_backtest_mean_reversion(daily_prices: pd.DataFrame, lookback_period_mr: int = 21, top_n_mr: int = 5) -> pd.Series:
    """Runs the complementary MEAN-REVERSION strategy."""
    monthly_prices = daily_prices.resample('ME').last()
    future_returns = monthly_prices.pct_change().shift(-1)
    
    short_term_returns = daily_prices.pct_change(periods=lookback_period_mr)
    monthly_short_term_returns = short_term_returns.resample('ME').last()
    
    long_term_trend = daily_prices.rolling(window=200).mean()
    monthly_long_term_trend = long_term_trend.resample('ME').last()
    
    portfolio_returns = pd.Series(index=monthly_prices.index, dtype=float)
    for month in monthly_prices.index:
        quality_filter = monthly_prices.loc[month] > monthly_long_term_trend.loc[month]
        quality_stocks = quality_filter[quality_filter].index
        if quality_stocks.empty:
            portfolio_returns.loc[month] = 0
            continue
            
        mr_candidates = monthly_short_term_returns.loc[month, quality_stocks].dropna()
        if mr_candidates.empty:
            portfolio_returns.loc[month] = 0
            continue
        
        dip_stocks = mr_candidates.nsmallest(top_n_mr)
        if dip_stocks.empty:
            portfolio_returns.loc[month] = 0
            continue
        
        final_weights = pd.Series(1/len(dip_stocks), index=dip_stocks.index)
        valid_tickers = final_weights.index.intersection(future_returns.columns)
        month_return = (future_returns.loc[month, valid_tickers] * final_weights[valid_tickers]).sum()
        portfolio_returns.loc[month] = month_return
    return portfolio_returns.fillna(0)

# --- Functions for Streamlit App ---
def generate_live_portfolio(momentum_window, top_n, cap):
    """Generates the live HYBRID portfolio but does not save it."""
    st.info("Fetching latest market data...")
    universe = get_nasdaq_100_plus_tickers()
    if not universe: return None, None

    start_date = (datetime.today() - relativedelta(months=momentum_window + 8)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    prices = fetch_market_data(universe, start_date, end_date)
    if prices.empty: return None, None
    
    # --- 1. Momentum Component (90% allocation) ---
    st.info("Calculating momentum signals...")
    monthly_prices_mom = prices.resample('ME').last()
    momentum = monthly_prices_mom.pct_change(periods=momentum_window).iloc[-1].dropna()
    positive_momentum = momentum[momentum > 0]
    top_performers = positive_momentum.nlargest(top_n)
    
    if top_performers.empty:
        mom_weights = pd.Series(dtype=float)
    else:
        raw_weights_mom = top_performers / top_performers.sum()
        mom_weights = cap_weights(raw_weights_mom, cap=cap) * 0.90
        
    # --- 2. Mean Reversion Component (10% allocation) ---
    st.info("Calculating mean-reversion signals...")
    short_term_returns = prices.pct_change(21).iloc[-1]
    long_term_trend = prices.rolling(200).mean().iloc[-1]
    quality_stocks = prices.iloc[-1] > long_term_trend
    
    mr_candidates = short_term_returns[quality_stocks[quality_stocks].index].dropna()
    dip_stocks = mr_candidates.nsmallest(5)
    
    if dip_stocks.empty:
        mr_weights = pd.Series(dtype=float)
    else:
        mr_weights = pd.Series(1/len(dip_stocks), index=dip_stocks.index) * 0.10
        
    # --- 3. Combine Portfolios ---
    st.info("Combining strategies into final portfolio...")
    final_weights = mom_weights.add(mr_weights, fill_value=0)
    final_weights = final_weights / final_weights.sum() # Re-normalize to 100%
    
    portfolio_df = pd.DataFrame({'Weight': final_weights}).sort_values('Weight', ascending=False)
    
    # Create display version
    display_df = portfolio_df.copy()
    display_df['Weight'] = display_df['Weight'].map('{:.2%}'.format)
    
    return display_df, portfolio_df

def run_backtest_for_app(momentum_window, top_n, cap):
    """Runs a quick backtest for the HYBRID strategy to display in the app."""
    start_date = '2018-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    universe = get_nasdaq_100_plus_tickers()
    all_tickers = universe + ['QQQ']
    
    prices = fetch_market_data(all_tickers, start_date, end_date)
    if prices.empty or 'QQQ' not in prices.columns: return None, None
        
    daily_prices = prices[universe]
    qqq_prices = prices['QQQ']

    # Run both strategies
    mom_returns = run_backtest_gross(daily_prices, momentum_window, top_n, cap)
    mr_returns = run_backtest_mean_reversion(daily_prices)
    
    # Combine returns for hybrid strategy
    hybrid_returns = (mom_returns * 0.90) + (mr_returns * 0.10)
    
    strategy_cumulative = (1 + hybrid_returns.fillna(0)).cumprod()
    qqq_cumulative = (1 + qqq_prices.resample('ME').last().pct_change()).cumprod()
    
    return strategy_cumulative, qqq_cumulative.reindex(strategy_cumulative.index, method='ffill')

# --- Gist Persistence (Unchanged) ---
def save_portfolio_to_gist(portfolio_df: pd.DataFrame):
    """Saves the provided portfolio DataFrame to the GitHub Gist."""
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
    try:
        response = requests.get(GIST_API_URL, headers=HEADERS)
        response.raise_for_status()
        gist_content = response.json()['files'][FILENAME]['content']
        if not gist_content or gist_content == '{}': return pd.DataFrame(columns=['Weight'])
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
        w_old = prev_df.at[ticker, 'Weight']
        w_new = curr_df.at[ticker, 'Weight']
        if abs(w_new - w_old) >= tol:
            rebalances.append((ticker, w_old, w_new))
    rebalances.sort(key=lambda x: abs(x[2] - x[1]), reverse=True)
    return {'sell': sorted(list(sells)), 'buy': sorted(list(buys)), 'rebalance': rebalances}
