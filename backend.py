import pandas as pd
import yfinance as yf
import requests
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- Settings & Data Fetching ---

@st.cache_data(ttl=86400) # Cache the list of tickers for 24 hours
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

@st.cache_data(ttl=43200) # Cache market data for 12 hours
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

# --- Live Portfolio Generation ---

def generate_live_portfolio(momentum_window=6, top_n=10, cap=0.25):
    """
    Calculates the live portfolio based on the Momentum-Score (25% Cap) strategy.
    """
    st.info("Fetching the latest market data...")
    
    # 1. Get the stock universe
    universe = get_nasdaq_100_plus_tickers()
    if not universe:
        return pd.DataFrame()

    # 2. Fetch recent price data (we need ~7 months for 6-month momentum)
    start_date = (datetime.today() - relativedelta(months=momentum_window + 1)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    prices = fetch_market_data(universe, start_date, end_date)
    if prices.empty:
        st.warning("Could not fetch price data for the universe.")
        return pd.DataFrame()

    # 3. Calculate latest momentum scores
    st.info("Calculating momentum scores...")
    monthly_prices = prices.resample('M').last()
    
    if len(monthly_prices) < momentum_window:
        st.warning(f"Not enough data for a {momentum_window}-month lookback.")
        return pd.DataFrame()

    momentum = monthly_prices.pct_change(periods=momentum_window)
    latest_momentum = momentum.iloc[-1].dropna()
    
    if latest_momentum.empty:
        st.warning("Could not calculate momentum scores.")
        return pd.DataFrame()

    # 4. Select top N stocks with positive momentum
    st.info(f"Selecting the top {top_n} stocks with positive momentum...")
    positive_momentum = latest_momentum[latest_momentum > 0]
    top_performers = positive_momentum.nlargest(top_n)

    if top_performers.empty:
        st.warning("No stocks with positive momentum found. Recommending cash.")
        return pd.DataFrame()

    # 5. Calculate weights using the Momentum-Score Capped method
    st.info("Calculating portfolio weights...")
    weights = top_performers / top_performers.sum()
    weights = weights.clip(upper=cap) # Apply the 25% cap
    weights = weights / weights.sum() # Re-normalize weights to sum to 1

    # 6. Format the output DataFrame
    portfolio_df = pd.DataFrame(weights, columns=['Weight'])
    portfolio_df['6-Month Momentum'] = top_performers
    
    portfolio_df = portfolio_df[['Weight', '6-Month Momentum']]
    portfolio_df['Weight'] = portfolio_df['Weight'].map('{:.2%}'.format)
    portfolio_df['6-Month Momentum'] = portfolio_df['6-Month Momentum'].map('{:.2%}'.format)
    
    return portfolio_df
