# backend.py
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

# =========================
# Gist API Constants
# =========================
GIST_ID = st.secrets.get("GIST_ID")
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN")
GIST_API_URL = f"https://api.github.com/gists/{GIST_ID}" if GIST_ID else None
HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'} if GITHUB_TOKEN else {}
FILENAME = 'portfolio.json'

# =========================
# Presets
# =========================
STRATEGY_PRESETS = {
    "ISA Dynamic (0.75)": {
        "mom_lb": 15, "mom_topn": 8, "mom_cap": 0.25,
        "mr_lb": 21,  "mr_topn": 3, "mr_ma": 200,
        "mom_w": 0.85, "mr_w": 0.15,
        "trigger": 0.75
    }
}

# =========================
# Cached Data Fetch
# =========================
@st.cache_data(ttl=86400)
def get_nasdaq_100_plus_tickers() -> list:
    """Fetch list of Nasdaq-100 + a few extra tech names."""
    try:
        payload = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        nasdaq_100 = payload[4]['Ticker'].tolist()
        extras = ['TSLA', 'SHOP', 'SNOW', 'PLTR', 'ETSY', 'RIVN', 'COIN']
        if 'SQ' in extras:
            extras.remove('SQ')
        full_list = sorted(list(set(nasdaq_100 + extras)))
        return full_list
    except Exception as e:
        st.error(f"Failed to fetch ticker list: {e}")
        return []

@st.cache_data(ttl=43200)
def fetch_market_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily Close prices for tickers (auto-adjusted)."""
    try:
        fetch_start = (pd.to_datetime(start_date) - pd.DateOffset(months=14)).strftime('%Y-%m-%d')
        data = yf.download(tickers, start=fetch_start, end=end_date, auto_adjust=True, progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data.dropna(axis=1, how='all')
    except Exception as e:
        st.error(f"Failed to download market data: {e}")
        return pd.DataFrame()

# =========================
# Utilities
# =========================
def cap_weights(weights: pd.Series, cap: float = 0.25) -> pd.Series:
    """Iterative 'waterfall' cap."""
    if weights.empty:
        return weights
    w = weights.copy().astype(float)
    for _ in range(100):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if w[under].sum() > 0:
            w[under] += (w[under] / w[under].sum()) * excess
        else:
            w += excess / len(w)
    return w

def _infer_periods_per_year(idx: pd.Index) -> float:
    """Robust inference of periods/year for frequency-aware KPIs."""
    idx = pd.DatetimeIndex(idx)
    if len(idx) < 3:
        return 12.0
    try:
        freq = pd.infer_freq(idx)
    except Exception:
        freq = None
    if freq:
        f = freq.upper()
        if f.startswith(('B','D')): return 252.0
        if f.startswith('W'):       return 52.0
        if f.startswith('M'):       return 12.0
        if f.startswith('Q'):       return 4.0
        if f.startswith(('A','Y')): return 1.0
    # fallback by median spacing
    deltas = np.diff(idx.view('i8')) / 1e9
    med_days = np.median(deltas) / 86400.0
    if med_days <= 2.5:  return 252.0
    if med_days <= 9:    return 52.0
    if med_days <= 45:   return 12.0
    if med_days <= 150:  return 4.0
    return 1.0

def _freq_label(py: float) -> str:
    if abs(py-252)<1: return "Daily (252py)"
    if abs(py-52)<1:  return "Weekly (52py)"
    if abs(py-12)<.5: return "Monthly (12py)"
    if abs(py-4)<.5:  return "Quarterly (4py)"
    if abs(py-1)<.2:  return "Yearly (1py)"
    return f"{py:.1f}py"

def equity_curve(returns: pd.Series) -> pd.Series:
    r = pd.Series(returns).fillna(0.0)
    return (1 + r).cumprod()

def drawdown(curve: pd.Series) -> pd.Series:
    return curve / curve.cummax() - 1

def kpi_row(name: str, rets: pd.Series, trade_log: pd.DataFrame | None = None, turnover_series: pd.Series | None = None) -> list:
    r = pd.Series(rets).dropna().astype(float)
    if r.empty:
        return [name, "-", "-", "-", "-", "-", "-", "-", "-", "-"]
    py = _infer_periods_per_year(r.index)
    eq = (1 + r).cumprod()
    n  = max(len(r), 1)
    ann_ret = eq.iloc[-1] ** (py / n) - 1
    mean_p, std_p = r.mean(), r.std()
    sharpe  = (mean_p * py) / (std_p * np.sqrt(py) + 1e-9)
    down_p  = r.clip(upper=0).std()
    sortino = (mean_p * py) / (down_p * np.sqrt(py) + 1e-9) if down_p > 0 else np.nan
    dd      = (eq/eq.cummax() - 1).min()
    calmar  = ann_ret / abs(dd) if dd != 0 else np.nan
    eq_mult = float(eq.iloc[-1])

    # Activity summaries — fallback to 0 if none
    tpy  = 0.0
    if trade_log is not None and len(trade_log) > 0 and 'date' in trade_log.columns:
        tl = trade_log.copy()
        tl['year'] = pd.to_datetime(tl['date']).dt.year
        grp = tl.groupby('year').size()
        if len(grp) > 0:
            tpy = float(grp.mean())

    topy = 0.0
    if turnover_series is not None and len(turnover_series) > 0:
        s = pd.Series(turnover_series).copy()
        s.index = pd.to_datetime(s.index)
        grp = s.groupby(s.index.year).sum()
        if len(grp) > 0:
            topy = float(grp.mean())

    return [
        name, _freq_label(py),
        f"{ann_ret*100:.2f}%",
        f"{sharpe:.2f}",
        f"{sortino:.2f}" if not np.isnan(sortino) else "N/A",
        f"{calmar:.2f}"  if not np.isnan(calmar)  else "N/A",
        f"{dd*100:.2f}%",
        f"{tpy:.1f}",
        f"{topy:.2f}",
        f"{eq_mult:.2f}x"
    ]

# =========================
# Regime & Live Tracking
# =========================

REGIME_MA = 200
LIVE_PERF_FILE = "live_perf.csv"

def _safe_series(obj):
    return obj.squeeze() if isinstance(obj, (pd.DataFrame,)) else obj

def get_benchmark_series(ticker: str, start: str, end: str) -> pd.Series:
    px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)['Close']
    px = _safe_series(px)
    return pd.Series(px).dropna()

def compute_regime_metrics(universe_prices_daily: pd.DataFrame) -> dict:
    """Simple, robust regime snapshot using daily data."""
    if universe_prices_daily.empty:
        return {}
    # QQQ anchors
    start = (universe_prices_daily.index.min() - pd.DateOffset(days=5)).strftime('%Y-%m-%d')
    end   = (universe_prices_daily.index.max() + pd.DateOffset(days=5)).strftime('%Y-%m-%d')
    qqq = get_benchmark_series("QQQ", start, end).reindex(universe_prices_daily.index).ffill().dropna()

    pct_above_ma = (universe_prices_daily.iloc[-1] >
                    universe_prices_daily.rolling(REGIME_MA).mean().iloc[-1]).mean()

    qqq_ma = qqq.rolling(REGIME_MA).mean()
    qqq_above_ma = float(qqq.iloc[-1] > qqq_ma.iloc[-1]) if len(qqq_ma.dropna()) else np.nan

    qqq_vol_10d = qqq.pct_change().rolling(10).std().iloc[-1]
    qqq_slope_50 = (qqq.rolling(50).mean().iloc[-1] / qqq.rolling(50).mean().iloc[-10] - 1) if len(qqq) > 60 else np.nan

    # Universe breadth (positive 6-month momentum)
    monthly = universe_prices_daily.resample('ME').last()
    pos_6m = (monthly.pct_change(6).iloc[-1] > 0).mean()

    return {
        "universe_above_200dma": float(pct_above_ma),   # 0–1
        "qqq_above_200dma": float(qqq_above_ma),       # 0/1
        "qqq_vol_10d": float(qqq_vol_10d),             # daily stdev
        "breadth_pos_6m": float(pos_6m),               # 0–1
        "qqq_50dma_slope_10d": float(qqq_slope_50) if pd.notna(qqq_slope_50) else np.nan
    }

# -------- Live paper tracker (very light) --------

def load_live_perf() -> pd.DataFrame:
    """Load live performance log from Gist (if configured) or return empty."""
    if not GIST_API_URL or not GITHUB_TOKEN:
        return pd.DataFrame(columns=['date','strat_ret','qqq_ret','note'])
    try:
        resp = requests.get(GIST_API_URL, headers=HEADERS); resp.raise_for_status()
        files = resp.json().get('files', {})
        content = files.get(LIVE_PERF_FILE, {}).get('content', '')
        if not content:
            return pd.DataFrame(columns=['date','strat_ret','qqq_ret','note'])
        df = pd.read_csv(pd.compat.StringIO(content))
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception:
        return pd.DataFrame(columns=['date','strat_ret','qqq_ret','note'])

def save_live_perf(df: pd.DataFrame):
    """Persist live performance log to Gist (no-op if not configured)."""
    if not GIST_API_URL or not GITHUB_TOKEN:
        return
    try:
        csv_str = df.to_csv(index=False)
        payload = {'files': {LIVE_PERF_FILE: {'content': csv_str}}}
        resp = requests.patch(GIST_API_URL, headers=HEADERS, json=payload)
        resp.raise_for_status()
    except Exception as e:
        st.sidebar.warning(f"Could not save live perf: {e}")

def calc_one_day_live_return(weights: pd.Series, daily_prices: pd.DataFrame) -> float:
    """
    Approx live 1-day return of portfolio using today's and yesterday's closes.
    weights: Series of target weights (sums to 1). daily_prices: Close df.
    """
    if weights is None or weights.empty or daily_prices.empty: return 0.0
    aligned = daily_prices[weights.index].dropna().iloc[-2:]  # last 2 days
    if len(aligned) < 2: return 0.0
    rets = aligned.pct_change().iloc[-1]
    return float((rets * weights).sum())

def record_live_snapshot(weights_df: pd.DataFrame, note: str = "") -> dict:
    """
    App helper: compute 1-day strat & QQQ returns from last two closes and save to Gist CSV.
    Returns a dict with the computed numbers for display.
    """
    try:
        universe = get_nasdaq_100_plus_tickers()
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - relativedelta(days=40)).strftime('%Y-%m-%d')
        px = fetch_market_data(universe + ['QQQ'], start_date, end_date)
        if px.empty or 'QQQ' not in px.columns:
            return {"ok": False, "msg": "No price data for live snapshot."}

        weights = weights_df['Weight'].astype(float)
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        strat_1d = calc_one_day_live_return(weights, px[weights.index])
        qqq_1d   = px['QQQ'].pct_change().iloc[-1]

        log = load_live_perf()
        new_row = pd.DataFrame([{
            'date': pd.to_datetime(px.index[-1]).normalize(),
            'strat_ret': strat_1d,
            'qqq_ret': float(qqq_1d),
            'note': note
        }])
        out = pd.concat([log, new_row], ignore_index=True).drop_duplicates(subset=['date'], keep='last')
        save_live_perf(out)
        return {"ok": True, "strat_ret": strat_1d, "qqq_ret": float(qqq_1d), "rows": len(out)}
    except Exception as e:
        return {"ok": False, "msg": str(e)}

def get_live_equity() -> pd.DataFrame:
    """
    Return cumulative equity of logged live returns for strategy vs QQQ.
    """
    log = load_live_perf().sort_values('date')
    if log.empty:
        return pd.DataFrame(columns=['date','strat_eq','qqq_eq'])
    df = log.copy()
    df['strat_eq'] = (1 + df['strat_ret'].fillna(0)).cumprod()
    df['qqq_eq']   = (1 + df['qqq_ret'].fillna(0)).cumprod()
    return df[['date','strat_eq','qqq_eq']]

# =========================
# Classic strategy (as before)
# =========================
def run_backtest_gross(daily_prices: pd.DataFrame, momentum_window: int = 6, top_n: int = 15, cap: float = 0.25) -> pd.Series:
    monthly_prices = daily_prices.resample('ME').last()
    future_returns = monthly_prices.pct_change().shift(-1)
    momentum = monthly_prices.pct_change(periods=momentum_window).shift(1)
    portfolio_returns = pd.Series(index=momentum.index, dtype=float)

    for month in momentum.index:
        momentum_scores = momentum.loc[month].dropna()
        positive_momentum = momentum_scores[momentum_scores > 0]
        if positive_momentum.empty:
            portfolio_returns.loc[month] = 0.0
            continue
        top_performers = positive_momentum.nlargest(top_n)
        raw_weights = top_performers / top_performers.sum()
        capped_weights = cap_weights(raw_weights, cap=cap)
        final_weights = capped_weights / capped_weights.sum()
        valid = final_weights.index.intersection(future_returns.columns)
        month_return = (future_returns.loc[month, valid] * final_weights[valid]).sum()
        portfolio_returns.loc[month] = float(month_return)
    return portfolio_returns.fillna(0.0)

def run_backtest_mean_reversion(daily_prices: pd.DataFrame, lookback_period_mr: int = 21, top_n_mr: int = 5, long_ma_days: int = 200) -> pd.Series:
    monthly_prices = daily_prices.resample('ME').last()
    future_returns = monthly_prices.pct_change().shift(-1)
    short_term_returns = daily_prices.pct_change(periods=lookback_period_mr)
    monthly_short = short_term_returns.resample('ME').last()
    long_trend = daily_prices.rolling(window=long_ma_days).mean().resample('ME').last()

    portfolio_returns = pd.Series(index=monthly_prices.index, dtype=float)
    for month in monthly_prices.index:
        quality = monthly_prices.loc[month] > long_trend.loc[month]
        quality_stocks = quality[quality].index
        if len(quality_stocks) == 0:
            portfolio_returns.loc[month] = 0.0
            continue
        mr_candidates = monthly_short.loc[month, quality_stocks].dropna()
        if mr_candidates.empty:
            portfolio_returns.loc[month] = 0.0
            continue
        dip_stocks = mr_candidates.nsmallest(top_n_mr)
        if dip_stocks.empty:
            portfolio_returns.loc[month] = 0.0
            continue
        weights = pd.Series(1/len(dip_stocks), index=dip_stocks.index)
        valid = weights.index.intersection(future_returns.columns)
        month_return = (future_returns.loc[month, valid] * weights[valid]).sum()
        portfolio_returns.loc[month] = float(month_return)
    return portfolio_returns.fillna(0.0)

# =========================
# Hybrid combiner (weights)
# =========================
def combine_hybrid(mom_rets: pd.Series, mr_rets: pd.Series, mom_tno: pd.Series | None = None, mr_tno: pd.Series | None = None,
                   mom_w: float = 0.9, mr_w: float = 0.1):
    mom = pd.Series(mom_rets).reindex(mr_rets.index, fill_value=0.0)
    mr  = pd.Series(mr_rets).reindex(mom.index,    fill_value=0.0)
    combo = (mom * mom_w) + (mr * mr_w)
    # simple turnover aggregation (if provided as monthly turnover fraction)
    if mom_tno is None and mr_tno is None:
        tno = None
    else:
        mtn = pd.Series(mom_tno).reindex(combo.index, fill_value=0.0) if mom_tno is not None else pd.Series(0.0, index=combo.index)
        rtn = pd.Series(mr_tno).reindex(combo.index,  fill_value=0.0) if mr_tno  is not None else pd.Series(0.0, index=combo.index)
        tno = (mtn * mom_w) + (rtn * mr_w)
    return combo, tno

# =========================
# ISA Dynamic (live) helpers
# =========================
def _momentum_scores_monthly(monthly_prices: pd.DataFrame, lookback_m: int) -> pd.Series:
    mom = monthly_prices.pct_change(periods=lookback_m).iloc[-1]
    return mom.dropna()

def _mr_scores_daily_to_monthly(daily_prices: pd.DataFrame, lookback_days: int, long_ma_days: int) -> pd.Series:
    st_returns = daily_prices.pct_change(lookback_days).iloc[-1]
    ltrend = daily_prices.rolling(long_ma_days).mean().iloc[-1]
    quality = daily_prices.iloc[-1] > ltrend
    mr = st_returns[quality[quality].index].dropna()
    return mr

def generate_live_portfolio_classic(momentum_window, top_n, cap):
    """Existing Classic 90/10 live weights (no trigger)."""
    universe = get_nasdaq_100_plus_tickers()
    if not universe:
        return None, None

    start_date = (datetime.today() - relativedelta(months=momentum_window + 8)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    prices = fetch_market_data(universe, start_date, end_date)
    if prices.empty:
        return None, None

    # Momentum sleeve (90%)
    monthly_prices_mom = prices.resample('ME').last()
    momentum = monthly_prices_mom.pct_change(periods=momentum_window).iloc[-1].dropna()
    positive_mom = momentum[momentum > 0]
    top_performers = positive_mom.nlargest(top_n)
    if top_performers.empty:
        mom_w = pd.Series(dtype=float)
    else:
        raw = top_performers / top_performers.sum()
        mom_w = cap_weights(raw, cap=cap) * 0.90

    # Mean reversion (10%)
    short_term = prices.pct_change(21).iloc[-1]
    long_trend = prices.rolling(200).mean().iloc[-1]
    quality = prices.iloc[-1] > long_trend
    mr_candidates = short_term[quality[quality].index].dropna()
    dip_stocks = mr_candidates.nsmallest(5)
    if dip_stocks.empty:
        mr_w = pd.Series(dtype=float)
    else:
        mr_w = pd.Series(1/len(dip_stocks), index=dip_stocks.index) * 0.10

    final = mom_w.add(mr_w, fill_value=0.0)
    final = final / final.sum() if final.sum() > 0 else final
    display_df = pd.DataFrame({'Weight': final}).sort_values('Weight', ascending=False)
    display_df['Weight'] = display_df['Weight'].map('{:.2%}'.format)
    raw_df = pd.DataFrame({'Weight': final}).sort_values('Weight', ascending=False)
    return display_df, raw_df

def generate_live_portfolio_isa(preset: dict, prev_portfolio: pd.DataFrame | None):
    """
    ISA Dynamic live weights:
      - Momentum: top N by M-month momentum, capped
      - MR: worst short-term (lookback_days) among stocks in long-term uptrend
      - Combine 85/15
      - Trigger: if portfolio health >= trigger, keep previous weights (no change)
    """
    universe = get_nasdaq_100_plus_tickers()
    if not universe:
        return None, None, "No universe available."

    start_date = (datetime.today() - relativedelta(months=preset['mom_lb'] + 8)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    prices = fetch_market_data(universe, start_date, end_date)
    if prices.empty:
        return None, None, "No price data."

    monthly = prices.resample('ME').last()

    # Momentum selection
    mom_scores = _momentum_scores_monthly(monthly, preset['mom_lb'])
    mom_scores = mom_scores[mom_scores > 0]
    top_m = mom_scores.nlargest(preset['mom_topn'])
    mom_raw = (top_m / top_m.sum()) if top_m.sum() > 0 else pd.Series(dtype=float)
    mom_w = cap_weights(mom_raw, cap=preset['mom_cap']) * preset['mom_w'] if not mom_raw.empty else pd.Series(dtype=float)

    # MR selection
    mr_scores = _mr_scores_daily_to_monthly(prices, preset['mr_lb'], preset['mr_ma'])
    dips = mr_scores.nsmallest(preset['mr_topn'])
    mr_w = (pd.Series(1/len(dips), index=dips.index) * preset['mr_w']) if len(dips) > 0 else pd.Series(dtype=float)

    # Combine
    new_w = mom_w.add(mr_w, fill_value=0.0)
    new_w = new_w / new_w.sum() if new_w.sum() > 0 else new_w

    # Trigger decision vs previous portfolio
    decision = "No previous portfolio found — proposing a full rebalance."
    if prev_portfolio is not None and not prev_portfolio.empty and 'Weight' in prev_portfolio.columns:
        prev_w = prev_portfolio['Weight'].astype(float)
        if len(top_m) > 0:
            top_score = float(top_m.iloc[0])
            held_scores = mom_scores.reindex(prev_w.index).fillna(0.0)
            health = float((held_scores * prev_w).sum() / max(top_score, 1e-9))
            if health >= preset['trigger']:
                # keep previous weights
                new_w = prev_w.copy()
                decision = f"Health {health:.2f} ≥ trigger {preset['trigger']:.2f} — holding existing portfolio."
            else:
                decision = f"Health {health:.2f} < trigger {preset['trigger']:.2f} — rebalancing to new targets."
        else:
            decision = "Momentum scores empty — holding previous portfolio."

    # Display + raw
    display_df = pd.DataFrame({'Weight': new_w}).sort_values('Weight', ascending=False)
    display_df_fmt = display_df.copy()
    display_df_fmt['Weight'] = display_df_fmt['Weight'].map('{:.2%}'.format)
    raw_df = display_df.copy()
    return display_df_fmt, raw_df, decision

# =========================
# Quick backtests for app
# =========================
def run_backtest_for_app(momentum_window, top_n, cap):
    """Classic 90/10 hybrid vs QQQ (since 2018)."""
    start_date = '2018-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    universe = get_nasdaq_100_plus_tickers()
    tickers = universe + ['QQQ']
    prices = fetch_market_data(tickers, start_date, end_date)
    if prices.empty or 'QQQ' not in prices.columns:
        return None, None

    valid_universe = [t for t in universe if t in prices.columns]
    daily = prices[valid_universe]
    qqq = prices['QQQ']

    mom = run_backtest_gross(daily, momentum_window, top_n, cap)
    mr  = run_backtest_mean_reversion(daily, 21, 5, 200)
    hybrid = (mom * 0.90) + (mr * 0.10)

    strat_cum = (1 + hybrid.fillna(0)).cumprod()
    qqq_cum = (1 + qqq.resample('ME').last().pct_change()).cumprod()
    return strat_cum, qqq_cum.reindex(strat_cum.index, method='ffill')

def run_backtest_isa_dynamic():
    """ISA preset backtest (no live trigger), since 2018."""
    params = STRATEGY_PRESETS["ISA Dynamic (0.75)"]
    start_date = '2018-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    universe = get_nasdaq_100_plus_tickers()
    tickers = universe + ['QQQ']
    prices = fetch_market_data(tickers, start_date, end_date)
    if prices.empty or 'QQQ' not in prices.columns:
        return None, None

    valid_universe = [t for t in universe if t in prices.columns]
    daily = prices[valid_universe]
    qqq = prices['QQQ']

    monthly = daily.resample('ME').last()
    fwd = monthly.pct_change().shift(-1)

    mom_scores_all = monthly.pct_change(periods=params['mom_lb']).shift(1)

    portfolio_rets = pd.Series(index=monthly.index, dtype=float)
    for m in monthly.index:
        scores = mom_scores_all.loc[m].dropna()
        scores = scores[scores > 0]
        if scores.empty:
            portfolio_rets.loc[m] = 0.0
            continue
        top = scores.nlargest(params['mom_topn'])
        raw = top / top.sum()
        w_m = cap_weights(raw, cap=params['mom_cap'])

        # MR sleeve
        st_ret = daily.pct_change(params['mr_lb']).resample('ME').last()
        lt_ma  = daily.rolling(params['mr_ma']).mean().resample('ME').last()
        if m in st_ret.index and m in lt_ma.index:
            quality = monthly.loc[m] > lt_ma.loc[m]
            mr_pool = st_ret.loc[m, quality[quality].index].dropna()
            dips = mr_pool.nsmallest(params['mr_topn'])
            w_r = pd.Series(1/len(dips), index=dips.index) if len(dips) > 0 else pd.Series(dtype=float)
        else:
            w_r = pd.Series(dtype=float)

        w = (w_m * params['mom_w']).add(w_r * params['mr_w'], fill_value=0.0)
        w = w / w.sum() if w.sum() > 0 else w
        valid = w.index.intersection(fwd.columns)
        ret_m = (fwd.loc[m, valid] * w[valid]).sum() if m in fwd.index else 0.0
        portfolio_rets.loc[m] = float(ret_m)

    strat_cum = (1 + portfolio_rets.fillna(0.0)).cumprod()
    qqq_cum = (1 + qqq.resample('ME').last().pct_change()).cumprod()
    return strat_cum, qqq_cum.reindex(strat_cum.index, method='ffill')

# =========================
# Gist persistence
# =========================
def save_portfolio_to_gist(portfolio_df: pd.DataFrame):
    if not GIST_API_URL or not GITHUB_TOKEN:
        st.sidebar.warning("Gist credentials not configured.")
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
    if not GIST_API_URL or not GITHUB_TOKEN:
        return pd.DataFrame(columns=['Weight'])
    try:
        response = requests.get(GIST_API_URL, headers=HEADERS)
        response.raise_for_status()
        gist_content = response.json()['files'].get(FILENAME, {}).get('content', '')
        if not gist_content or gist_content == '{}':
            return pd.DataFrame(columns=['Weight'])
        return pd.read_json(gist_content, orient="index")
    except Exception:
        return pd.DataFrame(columns=['Weight'])

def diff_portfolios(prev_df: pd.DataFrame, curr_df: pd.DataFrame, tol: float) -> dict:
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
