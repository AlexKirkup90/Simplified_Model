# backend.py
import os
import io
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta

# =========================
# Gist API (optional)
# =========================
GIST_ID = st.secrets.get("GIST_ID")
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN")
GIST_API_URL = f"https://api.github.com/gists/{GIST_ID}" if GIST_ID else None
HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'} if GITHUB_TOKEN else {}
GIST_PORTF_FILE = 'portfolio.json'
LIVE_PERF_FILE = "live_perf.csv"

# Local persistence fallback
LOCAL_PORTF_FILE = "last_portfolio.csv"

# =========================
# ISA preset
# =========================
STRATEGY_PRESETS = {
    "ISA Dynamic (0.75)": {
        "mom_lb": 15, "mom_topn": 8, "mom_cap": 0.25,
        "mr_lb": 21,  "mr_topn": 3, "mr_ma": 200,
        "mom_w": 0.85, "mr_w": 0.15,
        "trigger": 0.75
    }
}

REGIME_MA = 200

# =========================
# Universe & Data
# =========================
@st.cache_data(ttl=86400)
def get_nasdaq_100_plus_tickers() -> List[str]:
    """Fetch Nasdaq-100 tickers and add a few common tech names."""
    try:
        payload = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        nasdaq_100 = payload[4]['Ticker'].tolist()
        extras = ['TSLA', 'SHOP', 'SNOW', 'PLTR', 'ETSY', 'RIVN', 'COIN']
        if 'SQ' in extras:
            extras.remove('SQ')  # acquired
        return sorted(list(set(nasdaq_100 + extras)))
    except Exception as e:
        st.error(f"Failed to fetch ticker list: {e}")
        return []

@st.cache_data(ttl=43200)
def fetch_market_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Daily Close (auto-adjusted)."""
    try:
        fetch_start = (pd.to_datetime(start_date) - pd.DateOffset(months=14)).strftime('%Y-%m-%d')
        data = yf.download(tickers, start=fetch_start, end=end_date, auto_adjust=True, progress=False)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data.dropna(axis=1, how='all')
    except Exception as e:
        st.error(f"Failed to download market data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=43200)
def fetch_price_volume(tickers: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Daily Close & Volume as aligned DataFrames."""
    try:
        fetch_start = (pd.to_datetime(start_date) - pd.DateOffset(months=14)).strftime('%Y-%m-%d')
        df = yf.download(
            tickers, start=fetch_start, end=end_date,
            auto_adjust=True, progress=False
        )[['Close','Volume']]
        if isinstance(df, pd.Series):
            df = df.to_frame()

        if isinstance(df.columns, pd.MultiIndex):
            close = df['Close']
            vol   = df['Volume']
        else:
            close = df[['Close']]
            vol   = df[['Volume']]
            if len(tickers) == 1:
                close.columns = [tickers[0]]
                vol.columns   = [tickers[0]]

        close = close.dropna(axis=1, how='all')
        vol   = vol.reindex_like(close).fillna(0)
        return close, vol
    except Exception as e:
        st.error(f"Failed to download price/volume: {e}")
        return pd.DataFrame(), pd.DataFrame()

# =========================
# Persistence (Gist + Local)
# =========================
def save_portfolio_to_gist(portfolio_df: pd.DataFrame):
    """Save portfolio weights to Gist (JSON)."""
    if not GIST_API_URL or not GITHUB_TOKEN:
        st.sidebar.warning("Gist secrets not configured; skipping Gist save.")
        return
    try:
        json_content = portfolio_df.to_json(orient="index")
        payload = {'files': {GIST_PORTF_FILE: {'content': json_content}}}
        resp = requests.patch(GIST_API_URL, headers=HEADERS, json=payload)
        resp.raise_for_status()
        st.sidebar.success("✅ Successfully saved portfolio to Gist.")
    except Exception as e:
        st.sidebar.error(f"Gist save failed: {e}")

def load_previous_portfolio() -> Optional[pd.DataFrame]:
    """Load last saved portfolio from Gist (if configured) else local CSV."""
    # Try Gist first
    if GIST_API_URL and GITHUB_TOKEN:
        try:
            resp = requests.get(GIST_API_URL, headers=HEADERS)
            resp.raise_for_status()
            files = resp.json().get('files', {})
            content = files.get(GIST_PORTF_FILE, {}).get('content', '')
            if content and content != '{}':
                return pd.read_json(io.StringIO(content), orient="index")
        except Exception:
            pass
    # Local fallback
    if os.path.exists(LOCAL_PORTF_FILE):
        try:
            df = pd.read_csv(LOCAL_PORTF_FILE)
            if 'Weight' in df.columns and 'Ticker' in df.columns:
                return df.set_index('Ticker')
        except Exception:
            return None
    return None

def save_current_portfolio(df: pd.DataFrame):
    """Save latest portfolio locally for diffs."""
    try:
        out = df.copy()
        if out.index.name is None:
            out.index.name = 'Ticker'
        out.reset_index().to_csv(LOCAL_PORTF_FILE, index=False)
    except Exception as e:
        st.sidebar.warning(f"Could not save local portfolio: {e}")

# =========================
# Math utils & KPIs
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
    # fallback: median spacing
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

def kpi_row(name: str, rets: pd.Series,
            trade_log: Optional[pd.DataFrame] = None,
            turnover_series: Optional[pd.Series] = None) -> List[str]:
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

    tpy = 0.0
    if trade_log is not None and len(trade_log) > 0 and 'date' in trade_log.columns:
        tl = trade_log.copy()
        tl['year'] = pd.to_datetime(tl['date']).dt.year
        grp = tl.groupby('year').size()
        if len(grp) > 0:
            tpy = float(grp.mean())

    topy = 0.0
    if turnover_series is not None and len(turnover_series) > 0:
        s = pd.Series(turnover_series)
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
# Liquidity
# =========================
def median_dollar_volume(close_df: pd.DataFrame, vol_df: pd.DataFrame, window: int = 60) -> pd.Series:
    aligned_vol = vol_df.reindex_like(close_df).fillna(0)
    dollar = close_df * aligned_vol
    med = dollar.rolling(window).median().iloc[-1]
    return med.dropna()

def filter_by_liquidity(close_df: pd.DataFrame, vol_df: pd.DataFrame, min_dollar: float) -> List[str]:
    if close_df.empty or vol_df.empty:
        return []
    med = median_dollar_volume(close_df, vol_df, window=60)
    return med[med >= min_dollar].index.tolist()

# =========================
# Signals: blended momentum
# =========================
def blended_momentum_scores(monthly_prices: pd.DataFrame) -> pd.Series:
    m = monthly_prices
    r3  = m.pct_change(3).iloc[-1]
    r6  = m.pct_change(6).iloc[-1]
    r12 = m.pct_change(12).iloc[-1]

    def z(s: pd.Series) -> pd.Series:
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty or s.std(ddof=0) == 0:
            return pd.Series(0.0, index=m.columns)
        zs = (s - s.mean()) / (s.std(ddof=0) + 1e-9)
        return zs.reindex(m.columns).fillna(0.0)

    z3, z6, z12 = z(r3), z(r6), z(r12)
    score = 0.2*z3 + 0.4*z6 + 0.4*z12
    return score.dropna()

# =========================
# Sleeves (with turnover)
# =========================
def _weights_to_turnover(prev_w: pd.Series, new_w: pd.Series) -> float:
    prev = prev_w.reindex(new_w.index, fill_value=0.0)
    return 0.5 * float((new_w - prev).abs().sum())

def run_backtest_gross(daily_prices: pd.DataFrame,
                       momentum_window: int = 6,
                       top_n: int = 15,
                       cap: float = 0.25) -> Tuple[pd.Series, pd.Series]:
    """Momentum sleeve (single lookback) → returns, turnover."""
    monthly = daily_prices.resample('ME').last()
    fwd = monthly.pct_change().shift(-1)
    mom = monthly.pct_change(momentum_window).shift(1)

    rets = pd.Series(index=mom.index, dtype=float)
    tno  = pd.Series(index=mom.index, dtype=float)
    prev_w = pd.Series(dtype=float)

    for m in mom.index:
        scores = mom.loc[m].dropna()
        scores = scores[scores > 0]
        if scores.empty:
            rets.loc[m] = 0.0; tno.loc[m] = 0.0; continue

        top = scores.nlargest(top_n)
        raw = top / top.sum()
        w = cap_weights(raw, cap=cap)
        w = w / w.sum() if w.sum() > 0 else w

        valid = w.index.intersection(fwd.columns)
        rets.loc[m] = float((fwd.loc[m, valid] * w[valid]).sum())

        tno.loc[m] = _weights_to_turnover(prev_w, w)
        prev_w = w

    return rets.fillna(0.0), tno.fillna(0.0)

def run_backtest_mean_reversion(daily_prices: pd.DataFrame,
                                lookback_period_mr: int = 21,
                                top_n_mr: int = 5,
                                long_ma_days: int = 200) -> Tuple[pd.Series, pd.Series]:
    monthly = daily_prices.resample('ME').last()
    fwd = monthly.pct_change().shift(-1)
    st = daily_prices.pct_change(lookback_period_mr).resample('ME').last()
    lt = daily_prices.rolling(long_ma_days).mean().resample('ME').last()

    rets = pd.Series(index=monthly.index, dtype=float)
    tno  = pd.Series(index=monthly.index, dtype=float)
    prev_w = pd.Series(dtype=float)

    for m in monthly.index:
        quality = monthly.loc[m] > lt.loc[m]
        pool = quality[quality].index
        if len(pool) == 0:
            rets.loc[m] = 0.0; tno.loc[m] = 0.0; continue

        cand = st.loc[m, pool].dropna()
        if cand.empty:
            rets.loc[m] = 0.0; tno.loc[m] = 0.0; continue

        dips = cand.nsmallest(top_n_mr)
        if dips.empty:
            rets.loc[m] = 0.0; tno.loc[m] = 0.0; continue

        w = pd.Series(1/len(dips), index=dips.index)

        valid = w.index.intersection(fwd.columns)
        rets.loc[m] = float((fwd.loc[m, valid] * w[valid]).sum())

        tno.loc[m] = _weights_to_turnover(prev_w, w)
        prev_w = w

    return rets.fillna(0.0), tno.fillna(0.0)

def combine_hybrid(mom_rets: pd.Series, mr_rets: pd.Series,
                   mom_tno: Optional[pd.Series] = None,
                   mr_tno: Optional[pd.Series] = None,
                   mom_w: float = 0.9, mr_w: float = 0.1) -> Tuple[pd.Series, Optional[pd.Series]]:
    mom = pd.Series(mom_rets).reindex(mr_rets.index, fill_value=0.0)
    mr  = pd.Series(mr_rets).reindex(mom.index,    fill_value=0.0)
    combo = (mom * mom_w) + (mr * mr_w)
    if mom_tno is None and mr_tno is None:
        tno = None
    else:
        mtn = pd.Series(mom_tno).reindex(combo.index, fill_value=0.0) if mom_tno is not None else pd.Series(0.0, index=combo.index)
        rtn = pd.Series(mr_tno).reindex(combo.index,  fill_value=0.0) if mr_tno  is not None else pd.Series(0.0, index=combo.index)
        tno = (mtn * mom_w) + (rtn * mr_w)
    return combo, tno

def apply_costs(gross_returns: pd.Series, turnover: pd.Series, roundtrip_bps: float) -> pd.Series:
    """Net return = gross - turnover * (bps/10000)."""
    cost_frac = (roundtrip_bps / 10000.0)
    net = pd.Series(gross_returns).reindex(turnover.index).fillna(0.0) - pd.Series(turnover).fillna(0.0) * cost_frac
    return net

# =========================
# Live portfolio builders
# =========================
def _mr_scores_daily_to_monthly(daily_prices: pd.DataFrame, lookback_days: int, long_ma_days: int) -> pd.Series:
    st_returns = daily_prices.pct_change(lookback_days).iloc[-1]
    long_trend = daily_prices.rolling(long_ma_days).mean().iloc[-1]
    quality = daily_prices.iloc[-1] > long_trend
    return st_returns[quality[quality].index].dropna()

def generate_live_portfolio_classic(momentum_window: int, top_n: int, cap: float,
                                    min_dollar_volume: float = 0.0) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Classic 90/10 live weights (blended momentum; optional liquidity filter)."""
    universe = get_nasdaq_100_plus_tickers()
    if not universe:
        return None, None

    start = (datetime.today() - relativedelta(months=max(momentum_window, 12) + 8)).strftime('%Y-%m-%d')
    end   = datetime.today().strftime('%Y-%m-%d')
    close, vol = fetch_price_volume(universe, start, end)
    if close.empty:
        return None, None

    if min_dollar_volume > 0:
        keep = filter_by_liquidity(close, vol, min_dollar_volume)
        if not keep:
            return None, None
        close = close[keep]

    monthly = close.resample('ME').last()

    # Momentum sleeve (90%) with blended scores
    mom_scores = blended_momentum_scores(monthly)
    mom_scores = mom_scores[mom_scores > 0]
    top_performers = mom_scores.nlargest(top_n)
    if top_performers.empty:
        mom_w = pd.Series(dtype=float)
    else:
        raw = top_performers / top_performers.sum()
        mom_w = cap_weights(raw, cap=cap) * 0.90

    # MR sleeve (10%)
    mr_scores = _mr_scores_daily_to_monthly(close, 21, 200)
    dips = mr_scores.nsmallest(5)
    if dips.empty:
        mr_w = pd.Series(dtype=float)
    else:
        mr_w = pd.Series(1/len(dips), index=dips.index) * 0.10

    final = mom_w.add(mr_w, fill_value=0.0)
    final = final / final.sum() if final.sum() > 0 else final

    display_df = pd.DataFrame({'Weight': final}).sort_values('Weight', ascending=False)
    display_fmt = display_df.copy()
    display_fmt['Weight'] = display_fmt['Weight'].map('{:.2%}'.format)
    return display_fmt, display_df

def generate_live_portfolio_isa(preset: Dict, prev_portfolio: Optional[pd.DataFrame],
                                min_dollar_volume: float = 0.0) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """
    ISA Dynamic live weights with:
      - Momentum: blended 3/6/12m score, capped, top N
      - MR: worst short-term among stocks above long MA
      - Liquidity filter via median $ volume
      - Trigger decision vs previous portfolio
    """
    universe = get_nasdaq_100_plus_tickers()
    if not universe:
        return None, None, "No universe available."

    start = (datetime.today() - relativedelta(months=max(preset['mom_lb'], 12) + 8)).strftime('%Y-%m-%d')
    end   = datetime.today().strftime('%Y-%m-%d')
    close, vol = fetch_price_volume(universe, start, end)
    if close.empty:
        return None, None, "No price data."

    if min_dollar_volume > 0:
        keep = filter_by_liquidity(close, vol, min_dollar_volume)
        if not keep:
            return None, None, "No tickers pass liquidity filter."
        close = close[keep]

    monthly = close.resample('ME').last()

    # Momentum selection
    mom_scores = blended_momentum_scores(monthly)
    mom_scores = mom_scores[mom_scores > 0]
    top_m = mom_scores.nlargest(preset['mom_topn'])
    mom_raw = (top_m / top_m.sum()) if top_m.sum() > 0 else pd.Series(dtype=float)
    mom_w = cap_weights(mom_raw, cap=preset['mom_cap']) * preset['mom_w'] if not mom_raw.empty else pd.Series(dtype=float)

    # MR selection
    mr_scores = _mr_scores_daily_to_monthly(close, preset['mr_lb'], preset['mr_ma'])
    dips = mr_scores.nsmallest(preset['mr_topn'])
    mr_w = (pd.Series(1/len(dips), index=dips.index) * preset['mr_w']) if len(dips) > 0 else pd.Series(dtype=float)

    new_w = mom_w.add(mr_w, fill_value=0.0)
    new_w = new_w / new_w.sum() if new_w.sum() > 0 else new_w

    decision = "No previous portfolio found — proposing a full rebalance."
    if prev_portfolio is not None and not prev_portfolio.empty and 'Weight' in prev_portfolio.columns:
        prev_w = prev_portfolio['Weight'].astype(float)
        if len(top_m) > 0:
            top_score = float(top_m.iloc[0])
            held_scores = mom_scores.reindex(prev_w.index).fillna(0.0)
            health = float((held_scores * prev_w).sum() / max(top_score, 1e-9))
            if health >= preset['trigger']:
                new_w = prev_w.copy()
                decision = f"Health {health:.2f} ≥ trigger {preset['trigger']:.2f} — holding existing portfolio."
            else:
                decision = f"Health {health:.2f} < trigger {preset['trigger']:.2f} — rebalancing to new targets."
        else:
            decision = "Momentum scores empty — holding previous portfolio."

    display_df = pd.DataFrame({'Weight': new_w}).sort_values('Weight', ascending=False)
    display_fmt = display_df.copy()
    display_fmt['Weight'] = display_fmt['Weight'].map('{:.2%}'.format)
    return display_fmt, display_df, decision

# =========================
# Backtests for app (with costs & liquidity)
# =========================
def run_backtest_for_app(momentum_window: int, top_n: int, cap: float,
                         roundtrip_bps: float = 0.0,
                         min_dollar_volume: float = 0.0,
                         show_net: bool = False):
    start_date = '2018-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    universe = get_nasdaq_100_plus_tickers()
    tickers = universe + ['QQQ']

    close, vol = fetch_price_volume(tickers, start_date, end_date)
    if close.empty or 'QQQ' not in close.columns:
        return None, None, None, None

    # Only keep tickers we actually have data for
    available = [t for t in universe if t in close.columns]

    # Liquidity filter (on available only)
    if min_dollar_volume > 0 and available:
        keep = filter_by_liquidity(close[available], vol[available], min_dollar_volume)
        valid_universe = keep
    else:
        valid_universe = available

    if not valid_universe:
        return None, None, None, None

    daily = close[valid_universe]
    qqq = close['QQQ']

    mom_rets, mom_tno = run_backtest_gross(daily, momentum_window, top_n, cap)
    mr_rets,  mr_tno  = run_backtest_mean_reversion(daily, 21, 5, 200)
    hybrid_gross, hybrid_tno = combine_hybrid(mom_rets, mr_rets, mom_tno, mr_tno, mom_w=0.90, mr_w=0.10)
    hybrid_net = apply_costs(hybrid_gross, hybrid_tno, roundtrip_bps) if show_net else hybrid_gross

    strat_cum_gross = (1 + hybrid_gross.fillna(0)).cumprod()
    strat_cum_net   = (1 + hybrid_net.fillna(0)).cumprod()
    qqq_cum = (1 + qqq.resample('ME').last().pct_change()).cumprod()

    return strat_cum_gross, strat_cum_net, qqq_cum.reindex(strat_cum_gross.index, method='ffill'), hybrid_tno

 def run_backtest_isa_dynamic(roundtrip_bps: float = 0.0,
                             min_dollar_volume: float = 0.0,
                             show_net: bool = False):
    params = STRATEGY_PRESETS["ISA Dynamic (0.75)"]
    start_date = '2018-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    universe = get_nasdaq_100_plus_tickers()
    tickers = universe + ['QQQ']

    close, vol = fetch_price_volume(tickers, start_date, end_date)
    if close.empty or 'QQQ' not in close.columns:
        return None, None, None, None

    # Only keep tickers we actually have data for
    available = [t for t in universe if t in close.columns]

    # Liquidity filter (on available only)
    if min_dollar_volume > 0 and available:
        keep = filter_by_liquidity(close[available], vol[available], min_dollar_volume)
        valid_universe = keep
    else:
        valid_universe = available

    if not valid_universe:
        return None, None, None, None

    daily = close[valid_universe]
    qqq = close['QQQ']

    monthly = daily.resample('ME').last()
    fwd = monthly.pct_change().shift(-1)

    st_ret = daily.pct_change(params['mr_lb']).resample('ME').last()
    lt_ma  = daily.rolling(params['mr_ma']).mean().resample('ME').last()

    portfolio_rets = pd.Series(index=monthly.index, dtype=float)
    turnover_series = pd.Series(index=monthly.index, dtype=float)
    prev_w = pd.Series(dtype=float)

    for m in monthly.index:
        # Momentum blended
        mom_scores = blended_momentum_scores(monthly.loc[:m].iloc[-13:])
        mom_scores = mom_scores[mom_scores > 0]
        if mom_scores.empty:
            portfolio_rets.loc[m] = 0.0; turnover_series.loc[m] = 0.0; continue

        top = mom_scores.nlargest(params['mom_topn'])
        raw = top / top.sum()
        w_m = cap_weights(raw, cap=params['mom_cap'])

        # MR sleeve
        if m in st_ret.index and m in lt_ma.index:
            quality = monthly.loc[m] > lt_ma.loc[m]
            pool = quality[quality].index
            mr_pool = st_ret.loc[m, pool.intersection(st_ret.columns)].dropna()
            dips = mr_pool.nsmallest(params['mr_topn'])
            w_r = pd.Series(1/len(dips), index=dips.index) if len(dips) > 0 else pd.Series(dtype=float)
        else:
            w_r = pd.Series(dtype=float)

        # Combine
        w = (w_m * params['mom_w']).add(w_r * params['mr_w'], fill_value=0.0)
        w = w / w.sum() if w.sum() > 0 else w

        valid = w.index.intersection(fwd.columns)
        ret_m = (fwd.loc[m, valid] * w[valid]).sum() if m in fwd.index else 0.0
        portfolio_rets.loc[m] = float(ret_m)

        turnover_series.loc[m] = _weights_to_turnover(prev_w, w)
        prev_w = w

    gross = portfolio_rets.fillna(0.0)
    net = apply_costs(gross, turnover_series, roundtrip_bps) if show_net else gross

    strat_cum_gross = (1 + gross).cumprod()
    strat_cum_net   = (1 + net).cumprod()
    qqq_cum = (1 + qqq.resample('ME').last().pct_change()).cumprod()
    return strat_cum_gross, strat_cum_net, qqq_cum.reindex(strat_cum_gross.index, method='ffill'), turnover_series

# =========================
# Diff engine (for Plan tab)
# =========================
def diff_portfolios(prev_df: Optional[pd.DataFrame],
                    curr_df: Optional[pd.DataFrame],
                    tol: float = 0.01) -> Dict[str, object]:
    """
    Compare two portfolios (index=tickers, column 'Weight') and return:
      - sells: tickers in prev but not in curr
      - buys: tickers in curr but not in prev
      - rebalances: list[(ticker, old_w, new_w)] where |Δ| >= tol
    """
    if prev_df is None or prev_df.empty:
        prev_df = pd.DataFrame(columns=['Weight'])
    if curr_df is None or curr_df.empty:
        curr_df = pd.DataFrame(columns=['Weight'])

    prev_w = prev_df['Weight'] if 'Weight' in prev_df.columns else pd.Series(dtype=float)
    curr_w = curr_df['Weight'] if 'Weight' in curr_df.columns else pd.Series(dtype=float)

    tickers_prev = set(prev_w.index)
    tickers_curr = set(curr_w.index)
    sells = sorted(list(tickers_prev - tickers_curr))
    buys  = sorted(list(tickers_curr - tickers_prev))

    overlap = tickers_prev & tickers_curr
    rebalances = []
    for t in overlap:
        w_old = float(prev_w.get(t, 0.0))
        w_new = float(curr_w.get(t, 0.0))
        if abs(w_new - w_old) >= tol:
            rebalances.append((t, w_old, w_new))
    rebalances.sort(key=lambda x: abs(x[2] - x[1]), reverse=True)
    return {'sell': sells, 'buy': buys, 'rebalance': rebalances}

# =========================
# Explainability (“what changed & why”)
# =========================
def _signal_snapshot_for_explain(daily_prices: pd.DataFrame, params: Dict) -> pd.DataFrame:
    if daily_prices.empty:
        return pd.DataFrame()

    monthly = daily_prices.resample('ME').last()
    r3  = monthly.pct_change(3).iloc[-1]
    r6  = monthly.pct_change(6).iloc[-1]
    r12 = monthly.pct_change(12).iloc[-1]

    def z(s):
        s = s.replace([np.inf,-np.inf], np.nan).dropna()
        if s.std(ddof=0) == 0 or s.empty:
            return pd.Series(0.0, index=monthly.columns)
        zs = (s - s.mean()) / (s.std(ddof=0) + 1e-9)
        return zs.reindex(monthly.columns).fillna(0.0)

    mom_score = 0.2*z(r3) + 0.4*z(r6) + 0.4*z(r12)
    mom_rank  = mom_score.rank(ascending=False, method='min')

    st_ret = daily_prices.pct_change(params['mr_lb']).iloc[-1]
    long_ma = daily_prices.rolling(params['mr_ma']).mean().iloc[-1]
    above_ma = (daily_prices.iloc[-1] > long_ma).astype(int)

    snap = pd.DataFrame({
        'mom_score': mom_score,
        'mom_rank': mom_rank,
        f'st_ret_{params["mr_lb"]}d': st_ret,
        f'above_{params["mr_ma"]}dma': above_ma
    })
    return snap.sort_index()

def explain_portfolio_changes(prev_df: Optional[pd.DataFrame],
                              curr_df: Optional[pd.DataFrame],
                              daily_prices: pd.DataFrame,
                              params: Dict) -> pd.DataFrame:
    prev_df = prev_df if prev_df is not None else pd.DataFrame(columns=['Weight'])
    curr_df = curr_df if curr_df is not None else pd.DataFrame(columns=['Weight'])
    prev_w = prev_df['Weight'].astype(float) if 'Weight' in prev_df.columns else pd.Series(dtype=float)
    curr_w = curr_df['Weight'].astype(float) if 'Weight' in curr_df.columns else pd.Series(dtype=float)

    all_tickers = sorted(set(prev_w.index) | set(curr_w.index))
    if not all_tickers:
        return pd.DataFrame(columns=[
            'Ticker','Action','Old Wt','New Wt','Δ Wt (bps)',
            'Mom Rank','Mom Score',f'ST Return ({params["mr_lb"]}d)',f'Above {params["mr_ma"]}DMA'
        ])

    snap = _signal_snapshot_for_explain(daily_prices[all_tickers].dropna(axis=1, how='all'), params)

    rows = []
    for t in all_tickers:
        old_w = float(prev_w.get(t, 0.0))
        new_w = float(curr_w.get(t, 0.0))
        if abs(new_w - old_w) < 1e-9:
            continue

        if old_w == 0 and new_w > 0:
            action = 'Buy'
        elif new_w == 0 and old_w > 0:
            action = 'Sell'
        else:
            action = 'Rebalance'

        mom_rank = snap.at[t, 'mom_rank'] if t in snap.index else np.nan
        mom_score = snap.at[t, 'mom_score'] if t in snap.index else np.nan
        st_key = f'st_ret_{params["mr_lb"]}d'
        stv = snap.at[t, st_key] if t in snap.index else np.nan
        above_key = f'above_{params["mr_ma"]}dma'
        ab = snap.at[t, above_key] if t in snap.index else np.nan

        rows.append({
            'Ticker': t,
            'Action': action,
            'Old Wt': old_w,
            'New Wt': new_w,
            'Δ Wt (bps)': int(round((new_w - old_w) * 10000)),
            'Mom Rank': int(mom_rank) if pd.notna(mom_rank) else np.nan,
            'Mom Score': mom_score,
            f'ST Return ({params["mr_lb"]}d)': stv,
            f'Above {params["mr_ma"]}DMA': bool(ab) if pd.notna(ab) else None
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    action_order = pd.Categorical(out['Action'], categories=['Buy','Rebalance','Sell'], ordered=True)
    out = out.assign(ActionOrder=action_order).sort_values(['ActionOrder','Δ Wt (bps)'], ascending=[True, False]).drop(columns=['ActionOrder'])
    out['Old Wt'] = out['Old Wt'].map(lambda x: f"{x:.2%}")
    out['New Wt'] = out['New Wt'].map(lambda x: f"{x:.2%}")
    return out.reset_index(drop=True)

# =========================
# Regime & Live paper tracking
# =========================
def _safe_series(obj):
    return obj.squeeze() if isinstance(obj, (pd.DataFrame,)) else obj

def get_benchmark_series(ticker: str, start: str, end: str) -> pd.Series:
    px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)['Close']
    px = _safe_series(px)
    return pd.Series(px).dropna()

def compute_regime_metrics(universe_prices_daily: pd.DataFrame) -> Dict[str, float]:
    if universe_prices_daily.empty:
        return {}
    start = (universe_prices_daily.index.min() - pd.DateOffset(days=5)).strftime('%Y-%m-%d')
    end   = (universe_prices_daily.index.max() + pd.DateOffset(days=5)).strftime('%Y-%m-%d')
    qqq = get_benchmark_series("QQQ", start, end).reindex(universe_prices_daily.index).ffill().dropna()

    pct_above_ma = (universe_prices_daily.iloc[-1] >
                    universe_prices_daily.rolling(REGIME_MA).mean().iloc[-1]).mean()

    qqq_ma = qqq.rolling(REGIME_MA).mean()
    qqq_above_ma = float(qqq.iloc[-1] > qqq_ma.iloc[-1]) if len(qqq_ma.dropna()) else np.nan

    qqq_vol_10d = qqq.pct_change().rolling(10).std().iloc[-1]
    qqq_slope_50 = (qqq.rolling(50).mean().iloc[-1] / qqq.rolling(50).mean().iloc[-10] - 1) if len(qqq) > 60 else np.nan

    monthly = universe_prices_daily.resample('ME').last()
    pos_6m = (monthly.pct_change(6).iloc[-1] > 0).mean()

    return {
        "universe_above_200dma": float(pct_above_ma),
        "qqq_above_200dma": float(qqq_above_ma),
        "qqq_vol_10d": float(qqq_vol_10d),
        "breadth_pos_6m": float(pos_6m),
        "qqq_50dma_slope_10d": float(qqq_slope_50) if pd.notna(qqq_slope_50) else np.nan
    }

def load_live_perf() -> pd.DataFrame:
    # Try Gist
    if GIST_API_URL and GITHUB_TOKEN:
        try:
            resp = requests.get(GIST_API_URL, headers=HEADERS); resp.raise_for_status()
            files = resp.json().get('files', {})
            content = files.get(LIVE_PERF_FILE, {}).get('content', '')
            if not content:
                return pd.DataFrame(columns=['date','strat_ret','qqq_ret','note'])
            df = pd.read_csv(io.StringIO(content))
            df['date'] = pd.to_datetime(df['date'])
            return df
        except Exception:
            pass
    # Local start (empty)
    return pd.DataFrame(columns=['date','strat_ret','qqq_ret','note'])

def save_live_perf(df: pd.DataFrame):
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
    if weights is None or weights.empty or daily_prices.empty: return 0.0
    aligned = daily_prices[weights.index].dropna().iloc[-2:]  # last 2 days
    if len(aligned) < 2: return 0.0
    rets = aligned.pct_change().iloc[-1]
    return float((rets * weights).sum())

def record_live_snapshot(weights_df: pd.DataFrame, note: str = "") -> Dict[str, object]:
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
    log = load_live_perf().sort_values('date')
    if log.empty:
        return pd.DataFrame(columns=['date','strat_eq','qqq_eq'])
    df = log.copy()
    df['strat_eq'] = (1 + df['strat_ret'].fillna(0)).cumprod()
    df['qqq_eq']   = (1 + df['qqq_ret'].fillna(0)).cumprod()
    return df[['date','strat_eq','qqq_eq']]
