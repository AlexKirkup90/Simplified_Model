import pandas as pd
import numpy as np
import yfinance as yf
import warnings
from datetime import datetime
from dateutil.relativedelta import relativedelta

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================
# 1. Safe Data Fetching
# ==============================

def fetch_market_data_safe(tickers, start_date, end_date, chunk_size=20, max_retries=3):
    """
    Fetches market data in chunks with retry logic to avoid yfinance timeouts.
    Returns a DataFrame of adjusted close prices.
    """
    all_data = pd.DataFrame()
    tickers = list(set(tickers))  # remove duplicates if any

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Fetching {chunk} (Attempt {attempt})...")
                data = yf.download(chunk, start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']
                if isinstance(data, pd.Series):
                    data = data.to_frame()
                all_data = pd.concat([all_data, data], axis=1)
                break  # exit retry loop if successful
            except Exception as e:
                print(f"⚠️ Error fetching {chunk}: {e}")
                if attempt == max_retries:
                    print(f"❌ Failed to fetch {chunk} after {max_retries} attempts.")
    return all_data.dropna(axis=1, how='all')


# ==============================
# 2. Strategy Core Functions
# ==============================

def get_nasdaq_100_plus_tickers():
    """Fetches a list of NASDAQ-100 and other major tech tickers."""
    try:
        payload = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        nasdaq_100 = payload[4]['Ticker'].tolist()
        extras = ['TSLA', 'SHOP', 'SNOW', 'PLTR', 'ETSY', 'RIVN', 'COIN']
        if 'SQ' in extras:
            extras.remove('SQ')
        return sorted(list(set(nasdaq_100 + extras)))
    except Exception as e:
        print(f"Could not fetch ticker list: {e}")
        return []

def cap_weights(weights, cap=0.25, max_iterations=100):
    """Iterative 'waterfall' cap for portfolio weights."""
    w = weights.copy()
    for _ in range(max_iterations):
        over_cap = w > cap
        if not over_cap.any():
            return w
        excess_weight = (w[over_cap] - cap).sum()
        w[over_cap] = cap
        under_cap = ~over_cap
        if w[under_cap].sum() > 0:
            w[under_cap] += w[under_cap] / w[under_cap].sum() * excess_weight
        else:
            w += excess_weight / len(w)
    return w

def get_performance_metrics(returns, use_quantstats=False):
    """
    Calculates performance metrics for a MONTHLY returns series.
    Always returns the keys the app expects.
    """
    # Ensure we have a clean Series
    r = pd.Series(returns).dropna()
    if r.empty or len(r) < 2:
        return {
            'Annual Return': 'N/A',
            'Sharpe Ratio': 'N/A',
            'Sortino Ratio': 'N/A',
            'Calmar Ratio': 'N/A',
            'Max Drawdown': 'N/A',
        }

    # Annual return (CAGR from equity curve, not just mean*12)
    equity = (1 + r).cumprod()
    n = len(r)
    ann_return = equity.iloc[-1] ** (12 / n) - 1

    # Vol metrics (assumes monthly series -> 12 periods/yr)
    mean_m = r.mean()
    std_m = r.std()
    ann_vol = std_m * np.sqrt(12)

    # Sharpe (risk-free ~0)
    sharpe = (mean_m * 12) / (ann_vol + 1e-9)

    # Sortino (downside std only)
    downside = r.clip(upper=0)
    d_std_m = downside.std()
    sortino = (mean_m * 12) / (d_std_m * np.sqrt(12) + 1e-9) if d_std_m > 0 else np.nan

    # Max drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    mdd = drawdown.min()

    # Calmar
    calmar = (ann_return / abs(mdd)) if (mdd is not None and mdd != 0) else np.nan

    # Optional QS hook (ignored unless True)
    if use_quantstats:
        try:
            import quantstats as qs  # not required for the app path
            _ = qs  # just to avoid linter warnings if unused
        except Exception:
            pass

    def f_pct(x): 
        return "N/A" if pd.isna(x) else f"{x:.2%}"
    def f_num(x): 
        return "N/A" if pd.isna(x) else f"{x:.2f}"

    return {
        'Annual Return': f_pct(ann_return),
        'Sharpe Ratio': f_num(sharpe),
        'Sortino Ratio': f_num(sortino),
        'Calmar Ratio': f_num(calmar),
        'Max Drawdown': f_pct(mdd),
    }

def run_backtest_gross(daily_prices, momentum_window=6, top_n=15, cap=0.25):
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

def run_backtest_mean_reversion(daily_prices, lookback_period_mr=21, top_n_mr=5):
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

        final_weights = pd.Series(1 / len(dip_stocks), index=dip_stocks.index)

        valid_tickers = final_weights.index.intersection(future_returns.columns)
        month_return = (future_returns.loc[month, valid_tickers] * final_weights[valid_tickers]).sum()
        portfolio_returns.loc[month] = month_return

    return portfolio_returns.fillna(0)


# ==============================
# 3. Production Functions
# ==============================

def load_previous_portfolio():
    """Placeholder for loading saved portfolio."""
    return pd.DataFrame()

def save_portfolio_to_gist(portfolio_df):
    """Placeholder for saving portfolio."""
    print("Portfolio saved.")

def generate_live_portfolio(momentum_window, top_n, cap):
    """Generates a live portfolio using the current parameters."""
    END_DATE = datetime.today().strftime('%Y-%m-%d')
    START_DATE = (datetime.today() - relativedelta(years=20)).strftime('%Y-%m-%d')
    BENCHMARK_TICKER = 'SPY'

    tickers = get_nasdaq_100_plus_tickers()
    all_tickers = tickers + [BENCHMARK_TICKER]
    price_data = fetch_market_data_safe(all_tickers, START_DATE, END_DATE)

    if price_data.empty:
        return pd.DataFrame(), pd.DataFrame()

    strategy_prices = price_data[tickers].loc[START_DATE:END_DATE]
    momentum_returns = run_backtest_gross(strategy_prices, momentum_window, top_n, cap)
    mr_returns = run_backtest_mean_reversion(strategy_prices)

    hybrid_returns = (momentum_returns * 0.9) + (mr_returns * 0.1)

    latest_momentum = strategy_prices.pct_change(periods=momentum_window).iloc[-1]
    top_performers = latest_momentum.nlargest(top_n)
    raw_weights = top_performers / top_performers.sum()
    capped_weights = cap_weights(raw_weights, cap=cap)
    final_weights = capped_weights / capped_weights.sum()

    display_df = pd.DataFrame({'Ticker': final_weights.index, 'Weight': final_weights.values})
    display_df.set_index('Ticker', inplace=True)

    return display_df, display_df

def run_backtest_for_app(momentum_window, top_n, cap):
    """Runs a backtest and returns cumulative returns."""
    END_DATE = datetime.today().strftime('%Y-%m-%d')
    START_DATE = (datetime.today() - relativedelta(years=20)).strftime('%Y-%m-%d')
    BENCHMARK_TICKER = 'SPY'

    tickers = get_nasdaq_100_plus_tickers()
    all_tickers = tickers + [BENCHMARK_TICKER]
    price_data = fetch_market_data_safe(all_tickers, START_DATE, END_DATE)

    if price_data.empty:
        return None, None

    strategy_prices = price_data[tickers].loc[START_DATE:END_DATE]
    benchmark_prices = price_data[BENCHMARK_TICKER].loc[START_DATE:END_DATE]

    momentum_returns = run_backtest_gross(strategy_prices, momentum_window, top_n, cap)
    mr_returns = run_backtest_mean_reversion(strategy_prices)

    hybrid_returns = (momentum_returns * 0.9) + (mr_returns * 0.1)
    hybrid_cum = (1 + hybrid_returns).cumprod()
    benchmark_cum = (1 + benchmark_prices.resample('ME').last().pct_change().fillna(0)).cumprod()

    return hybrid_cum, benchmark_cum

def diff_portfolios(prev_df, new_df, tol):
    """Compares two portfolios and returns buy/sell/rebalance signals."""
    if prev_df is None or prev_df.empty:
        return {'buy': list(new_df.index), 'sell': [], 'rebalance': []}

    buy = [t for t in new_df.index if t not in prev_df.index]
    sell = [t for t in prev_df.index if t not in new_df.index]
    rebalance = []
    for ticker in set(prev_df.index).intersection(new_df.index):
        old_w, new_w = prev_df.at[ticker, 'Weight'], new_df.at[ticker, 'Weight']
        if abs(old_w - new_w) > tol:
            rebalance.append((ticker, old_w, new_w))

    return {'buy': buy, 'sell': sell, 'rebalance': rebalance}
