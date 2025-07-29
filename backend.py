import pandas as pd
import numpy as np
import yfinance as yf
import requests
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json

# --- Gist API Constants & Headers ---
GIST_ID = st.secrets.get("GIST_ID")
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN")
GIST_API_URL = f"https://api.github.com/gists/{GIST_ID}"
HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'}
FILENAME = 'portfolio.json'

# --- Data Fetching ---
@st.cache_data(ttl=86400)
def get_nasdaq_100_plus_tickers():
    """Fetches a list of NASDAQ-100 and other major tech tickers."""
    try:
        payload = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        nasdaq_100 = payload[4]['Ticker'].tolist()
        extras = ['TSLA', 'SHOP', 'SNOW', 'PLTR', 'ETSY', 'RIVN', 'COIN']
        if 'SQ' in extras: extras.remove('SQ')
        full_list = sorted(list(set(nasdaq_100 + extras)))
        return full_list
    except Exception as e:
        st.error(f"Failed to fetch ticker list: {e}")
        return []

@st.cache_data(ttl=43200)
def fetch_market_data(tickers, start_date, end_date):
    """Fetches daily 'Close' price data for a list of tickers."""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']
        if isinstance(data, pd.DataFrame):
            return data.dropna(axis=1, how='all')
        return data
    except Exception as e:
        st.error(f"Failed to download market data: {e}")
        return pd.DataFrame()

# --- NEW: Iterative Weight Capping Function ---
def cap_weights(weights, cap=0.25):
    """
    Applies an iterative "waterfall" cap to portfolio weights.
    It caps the largest weights and redistributes the excess to the uncapped remainder,
    repeating until no weight exceeds the cap.
    """
    w = weights.copy()
    while True:
        over_cap = w > cap
        if not over_cap.any():
            break
        
        excess_weight = (w[over_cap] - cap).sum()
        w[over_cap] = cap
        
        under_cap = ~over_cap
        # Redistribute excess weight proportionally to the under-cap weights
        w[under_cap] += w[under_cap] / w[under_cap].sum() * excess_weight
    return w

# --- Live Portfolio Generation & Gist Persistence ---
def generate_live_portfolio(momentum_window=6, top_n=10, cap=0.25):
    """
    Generates the live portfolio, saves it to Gist, and returns display/raw dataframes.
    """
    st.info("Fetching the latest market data...")
    universe = get_nasdaq_100_plus_tickers()
    if not universe: return None, None

    start_date = (datetime.today() - relativedelta(months=momentum_window + 2)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    prices = fetch_market_data(universe, start_date, end_date)
    if prices.empty: return None, None

    st.info("Calculating momentum scores...")
    monthly_prices = prices.resample('ME').last()
    if len(monthly_prices) < momentum_window: return None, None

    momentum = monthly_prices.pct_change(periods=momentum_window)
    latest_momentum = momentum.iloc[-1].dropna()
    if latest_momentum.empty: return None, None

    st.info(f"Selecting the top {top_n} stocks...")
    positive_momentum = latest_momentum[latest_momentum > 0]
    top_performers = positive_momentum.nlargest(top_n)
    if top_performers.empty:
        st.warning("No stocks with positive momentum found. Recommending cash.")
        return pd.DataFrame(), pd.DataFrame()

    st.info("Calculating portfolio weights...")
    raw_weights = top_performers / top_performers.sum()
    final_weights = cap_weights(raw_weights, cap=cap)

    portfolio_df = pd.DataFrame({'Weight': final_weights})
    
    try:
        json_content = portfolio_df.to_json(orient="index")
        data_to_save = {'files': {FILENAME: {'content': json_content}}}
        response = requests.patch(GIST_API_URL, headers=HEADERS, json=data_to_save)
        response.raise_for_status() # This will raise an HTTPError for bad responses (4xx or 5xx)
        st.sidebar.success("Successfully saved portfolio to Gist.")
    except requests.exceptions.HTTPError as he:
        # ENHANCED: Provide detailed error logging
        st.sidebar.error(f"Gist save failed [{he.response.status_code}]: {he.response.text}")
    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred while saving to Gist: {e}")

    display_df = portfolio_df.copy()
    display_df['6-Month Momentum'] = top_performers
    display_df['Weight'] = display_df['Weight'].map('{:.2%}'.format)
    display_df['6-Month Momentum'] = display_df['6-Month Momentum'].map('{:.2%}'.format)
    
    return display_df, portfolio_df

# --- Portfolio Comparison Logic ---
def load_previous_portfolio():
    """Loads the last saved portfolio from the GitHub Gist."""
    try:
        response = requests.get(GIST_API_URL, headers=HEADERS)
        response.raise_for_status()
        gist_content = response.json()['files'][FILENAME]['content']
        if not gist_content or gist_content == '{}': return pd.DataFrame(columns=['Weight'])
        return pd.read_json(gist_content, orient="index")
    except requests.exceptions.HTTPError as he:
        st.sidebar.error(f"Gist load failed [{he.response.status_code}]: {he.response.text}")
        return pd.DataFrame(columns=['Weight'])
    except Exception as e:
        st.sidebar.warning(f"Could not load previous portfolio. (Error: {e})")
        return pd.DataFrame(columns=['Weight'])

def diff_portfolios(prev_df, curr_df, tol=0.01):
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
