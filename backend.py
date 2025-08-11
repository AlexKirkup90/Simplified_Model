# backend.py
import os
import io
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

# =========================
# Gist API (optional)
# =========================
GIST_ID = st.secrets.get("GIST_ID")
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN")
GIST_API_URL = f"https://api.github.com/gists/{GIST_ID}" if GIST_ID else None
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
GIST_PORTF_FILE = "portfolio.json"
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
        "trigger": 0.75,
        "stability_days": 5  # must persist in top cohort this many consecutive trading days
    }
}

REGIME_MA = 200

# =========================
# Helpers
# =========================
def _safe_series(obj):
    return obj.squeeze() if isinstance(obj, pd.DataFrame) else obj

def _today_str():
    return datetime.today().strftime("%Y-%m-%d")

# =========================
# Universe & Data
# =========================
@st.cache_data(ttl=86400)
def get_nasdaq_100_plus_tickers() -> List[str]:
    """Fetch Nasdaq-100 tickers and add a few common tech names."""
    try:
        payload = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        nasdaq_100 = payload[4]["Ticker"].tolist()
        extras = ["TSLA", "SHOP", "SNOW", "PLTR", "ETSY", "RIVN", "COIN"]
        if "SQ" in extras:
            extras.remove("SQ")  # acquired
        return sorted(list(set(nasdaq_100 + extras)))
    except Exception as e:
        st.error(f"Failed to fetch NASDAQ-100 list: {e}")
        return []

@st.cache_data(ttl=86400)
def get_sp500_tickers() -> List[str]:
    """Fetch S&P 500 tickers from Wikipedia (dot to dash for tickers like BRK.B -> BRK-B)."""
    try:
        t = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        sp = t[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        return sorted(sp)
    except Exception as e:
        st.error(f"Failed to fetch S&P 500 list: {e}")
        return []

@st.cache_data(ttl=86400)
def get_universe(universe_name: str = "NASDAQ100+", top_liq_n: int = 150) -> List[str]:
    """
    Returns a list of tickers for the chosen universe:
      - "NASDAQ100+" : curated tech-heavy list
      - "S&P500"     : all current S&P500 members
      - "Hybrid Top150": NASDAQ100+ union with top-liquidity S&P names (later narrowed after prices are fetched)
    NOTE: Hybrid Top150 final selection happens after prices are downloaded (needs volume).
    """
    if universe_name == "NASDAQ100+":
        return get_nasdaq_100_plus_tickers()
    elif universe_name == "S&P500":
        return get_sp500_tickers()
    elif universe_name == "Hybrid Top150":
        # Return union (we'll trim to top-liquidity later)
        base = sorted(list(set(get_nasdaq_100_plus_tickers() + get_sp500_tickers())))
        return base
    else:
        return get_nasdaq_100_plus_tickers()

@st.cache_data(ttl=43200)
def fetch_market_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Daily Close (auto-adjusted)."""
    try:
        fetch_start = (pd.to_datetime(start_date) - pd.DateOffset(months=14)).strftime("%Y-%m-%d")
        data = yf.download(tickers, start=fetch_start, end=end_date, auto_adjust=True, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data.dropna(axis=1, how="all")
    except Exception as e:
        st.error(f"Failed to download market data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=43200)
def fetch_price_volume(tickers: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Daily Close & Volume as aligned DataFrames."""
    try:
        fetch_start = (pd.to_datetime(start_date) - pd.DateOffset(months=14)).strftime("%Y-%m-%d")
        df = yf.download(
            tickers, start=fetch_start, end=end_date,
            auto_adjust=True, progress=False
        )[["Close", "Volume"]]

        # Normalize structures for 1 vs many tickers
        if isinstance(df, pd.Series):
            df = df.to_frame()

        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"]
            vol = df["Volume"]
        else:
            # Single ticker fallback
            if "Close" in df.columns and "Volume" in df.columns:
                close = df[["Close"]].copy()
                vol = df[["Volume"]].copy()
                if len(tickers) == 1:
                    close.columns = [tickers[0]]
                    vol.columns = [tickers[0]]
            else:
                return pd.DataFrame(), pd.DataFrame()

        close = close.dropna(axis=1, how="all")
        vol = vol.reindex_like(close).fillna(0)
        return close, vol
    except Exception as e:
        st.error(f"Failed to download price/volume: {e}")
        return pd.DataFrame(), pd.DataFrame()

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

def top_by_liquidity(close_df: pd.DataFrame, vol_df: pd.DataFrame, top_n: int = 150) -> List[str]:
    mdv = median_dollar_volume(close_df, vol_df, window=60)
    return mdv.sort_values(ascending=False).head(top_n).index.tolist()

# =========================
# Universe-specific price pulls
# =========================
def get_universe_prices(universe_name: str,
                        start_date: str,
                        end_date: str,
                        min_dollar_volume: float = 0.0,
                        hybrid_top_n: int = 150) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Returns (close, volume, final_tickers) for a chosen universe.
    - For Hybrid Top150: we build union (NDX+SPX), then keep top-N by median $ volume.
    - For others: we optionally apply min_dollar_volume threshold.
    """
    base_tickers = get_universe(universe_name, top_liq_n=hybrid_top_n)
    close, vol = fetch_price_volume(base_tickers, start_date, end_date)
    if close.empty:
        return close, vol, []

    if universe_name == "Hybrid Top150":
        keep = top_by_liquidity(close, vol, top_n=hybrid_top_n)
        close = close[keep]
        vol = vol[keep]
        return close, vol, keep

    # Otherwise, apply an optional raw liquidity floor if asked
    if min_dollar_volume > 0:
        keep = filter_by_liquidity(close, vol, min_dollar_volume)
        close = close[keep] if keep else close.iloc[:, :0]
        vol = vol.reindex_like(close)
        return close, vol, keep

    return close, vol, [c for c in close.columns]

# =========================
# Persistence (Gist + Local)
# =========================
def save_portfolio_to_gist(portfolio_df: pd.DataFrame) -> None:
    """Save portfolio weights to Gist (JSON)."""
    if not GIST_API_URL or not GITHUB_TOKEN:
        st.sidebar.warning("Gist secrets not configured; skipping Gist save.")
        return
    try:
        json_content = portfolio_df.to_json(orient="index")
        payload = {"files": {GIST_PORTF_FILE: {"content": json_content}}}
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
            files = resp.json().get("files", {})
            content = files.get(GIST_PORTF_FILE, {}).get("content", "")
            if content and content != "{}":
                return pd.read_json(io.StringIO(content), orient="index")
        except Exception:
            pass
    # Local fallback
    if os.path.exists(LOCAL_PORTF_FILE):
        try:
            df = pd.read_csv(LOCAL_PORTF_FILE)
            if "Weight" in df.columns and "Ticker" in df.columns:
                return df.set_index("Ticker")
        except Exception:
            return None
    return None

def save_current_portfolio(df: pd.DataFrame) -> None:
    """Save latest portfolio locally for diffs."""
    try:
        out = df.copy()
        if out.index.name is None:
            out.index.name = "Ticker"
        out.reset_index().to_csv(LOCAL_PORTF_FILE, index=False)
    except Exception as e:
        st.sidebar.warning(f"Could not save local portfolio: {e}")

# =========================
# Calendar helpers (Monthly Lock)
# =========================
def first_trading_day(dt: pd.Timestamp, ref_index: Optional[pd.DatetimeIndex] = None) -> pd.Timestamp:
    month_start = pd.Timestamp(year=dt.year, month=dt.month, day=1)
    if ref_index is not None and len(ref_index) > 0:
        month_dates = ref_index[(ref_index >= month_start) & (ref_index < month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1))]
        if len(month_dates) > 0:
            return pd.Timestamp(month_dates[0]).normalize()
    bdays = pd.bdate_range(month_start, month_start + pd.offsets.MonthEnd(1))
    return pd.Timestamp(bdays[0]).normalize()

def is_rebalance_today(today: date, price_index: Optional[pd.DatetimeIndex]) -> bool:
    ts = pd.Timestamp(today)
    ftd = first_trading_day(ts, price_index)
    return ts.normalize() == ftd

# =========================
# Math utils & KPIs
# =========================
def cap_weights(weights: pd.Series, cap: float = 0.25) -> pd.Series:
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
        if f.startswith(("B", "D")): return 252.0
        if f.startswith("W"):        return 52.0
        if f.startswith("M"):        return 12.0
        if f.startswith("Q"):        return 4.0
        if f.startswith(("A", "Y")): return 1.0
    deltas = np.diff(idx.view("i8")) / 1e9
    med_days = np.median(deltas) / 86400.0
    if med_days <= 2.5:  return 252.0
    if med_days <= 9:    return 52.0
    if med_days <= 45:   return 12.0
    if med_days <= 150:  return 4.0
    return 1.0

def _freq_label(py: float) -> str:
    if abs(py - 252) < 1: return "Daily (252py)"
    if abs(py - 52) < 1:  return "Weekly (52py)"
    if abs(py - 12) < .5: return "Monthly (12py)"
    if abs(py - 4) < .5:  return "Quarterly (4py)"
    if abs(py - 1) < .2:  return "Yearly (1py)"
    return f"{py:.1f}py"

def equity_curve(returns: pd.Series) -> pd.Series:
    r = pd.Series(returns).fillna(0.0)
    return (1 + r).cumprod()

def drawdown(curve: pd.Series) -> pd.Series:
    return curve / curve.cummax() - 1

def kpi_row(name: str,
            rets: pd.Series,
            trade_log: Optional[pd.DataFrame] = None,
            turnover_series: Optional[pd.Series] = None,
            avg_trade_size: float = 0.02) -> List[str]:
    """
    KPI row with robust turnover:
      - Turnover/yr = mean of calendar-year sums of monthly turnover
      - Trades/yr   ≈ Turnover/yr ÷ avg_trade_size (default 2% per single-leg trade)
    """
    r = pd.Series(rets).dropna().astype(float)
    if r.empty:
        return [name, "-", "-", "-", "-", "-", "-", "-", "-", "-"]

    # infer periodicity
    idx = pd.to_datetime(r.index)
    # very light inference: monthly-ish unless clearly daily/weekly
    def _infer_py(ix):
        if len(ix) < 3: return 12.0
        try: f = pd.infer_freq(ix)
        except Exception: f = None
        if f:
            F = f.upper()
            if F.startswith(("B","D")): return 252.0
            if F.startswith("W"):       return 52.0
            if F.startswith("M"):       return 12.0
            if F.startswith("Q"):       return 4.0
            if F.startswith(("A","Y")): return 1.0
        # fallback by median spacing
        d = np.median(np.diff(ix.view("i8")))/1e9/86400.0
        return 252.0 if d<=2.5 else 52.0 if d<=9 else 12.0 if d<=45 else 4.0 if d<=150 else 1.0

    py = _infer_py(idx)
    freq_label = (
        "Daily (252py)" if abs(py-252)<1 else
        "Weekly (52py)" if abs(py-52)<1 else
        "Monthly (12py)" if abs(py-12)<0.5 else
        "Quarterly (4py)" if abs(py-4)<0.5 else
        "Yearly (1py)" if abs(py-1)<0.2 else
        f"{py:.1f}py"
    )

    # core KPIs
    eq = (1 + r).cumprod()
    n  = max(len(r), 1)
    ann_return = eq.iloc[-1] ** (py / n) - 1
    mu, sd = r.mean(), r.std()
    sharpe  = (mu * py) / (sd * np.sqrt(py) + 1e-9)
    down_sd = r.clip(upper=0).std()
    sortino = (mu * py) / (down_sd * np.sqrt(py) + 1e-9) if down_sd > 0 else np.nan
    dd      = (eq/eq.cummax() - 1).min()
    calmar  = ann_return / abs(dd) if dd != 0 else np.nan
    eq_mult = float(eq.iloc[-1])

    # robust turnover calc (group-by-year mean)
    tpy = 0.0
    if turnover_series is not None and len(turnover_series) > 0:
        ts = pd.Series(turnover_series).copy()
        ts.index = pd.to_datetime(ts.index)
        yearly_sum = ts.groupby(ts.index.year).sum()
        if len(yearly_sum) > 0:
            tpy = float(yearly_sum.mean())

    # trades per year (single-leg) from average trade size assumption
    trades_per_year = (tpy / avg_trade_size) if tpy > 0 and avg_trade_size > 0 else 0.0

    return [
        name, freq_label,
        f"{ann_return*100:.2f}%",
        f"{sharpe:.2f}",
        f"{sortino:.2f}" if not np.isnan(sortino) else "N/A",
        f"{calmar:.2f}"  if not np.isnan(calmar)  else "N/A",
        f"{dd*100:.2f}%",
        f"{trades_per_year:.1f}",
        f"{tpy:.2f}",
        f"{eq_mult:.2f}x"
    ]

# =========================
# Momentum scoring (blended)
# =========================
def blended_momentum_scores(monthly_prices: pd.DataFrame) -> pd.Series:
    m = monthly_prices
    if m.shape[0] < 13:
        return pd.Series(dtype=float)
    r3  = m.pct_change(3).iloc[-1]
    r6  = m.pct_change(6).iloc[-1]
    r12 = m.pct_change(12).iloc[-1]

    def z(s: pd.Series, cols) -> pd.Series:
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty or s.std(ddof=0) == 0:
            return pd.Series(0.0, index=cols)
        zs = (s - s.mean()) / (s.std(ddof=0) + 1e-9)
        return zs.reindex(cols).fillna(0.0)

    cols = m.columns
    z3, z6, z12 = z(r3, cols), z(r6, cols), z(r12, cols)
    score = 0.2*z3 + 0.4*z6 + 0.4*z12
    return score.dropna()

def blended_momentum_scores_daily(daily_prices: pd.DataFrame) -> pd.DataFrame:
    px = daily_prices.copy()
    if px.shape[0] < 260:
        return pd.DataFrame(index=px.index, columns=px.columns)

    r63  = px.pct_change(63)
    r126 = px.pct_change(126)
    r252 = px.pct_change(252)

    def z_df(df: pd.DataFrame) -> pd.DataFrame:
        mu = df.mean(axis=1)
        sd = df.std(axis=1).replace(0, np.nan)
        z = (df.sub(mu, axis=0)).div(sd, axis=0)
        return z.fillna(0.0)

    z63, z126, z252 = z_df(r63), z_df(r126), z_df(r252)
    score = 0.2*z63 + 0.4*z126 + 0.4*z252
    return score

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
    monthly = daily_prices.resample("ME").last()
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
    monthly = daily_prices.resample("ME").last()
    fwd = monthly.pct_change().shift(-1)
    st = daily_prices.pct_change(lookback_period_mr).resample("ME").last()
    lt = daily_prices.rolling(long_ma_days).mean().resample("ME").last()

    rets = pd.Series(index=monthly.index, dtype=float)
    tno  = pd.Series(index=monthly.index, dtype=float)
    prev_w = pd.Series(dtype=float)

    for m in monthly.index:
        quality = monthly.loc[m] > lt.loc[m]
        pool = quality[quality].index
        if len(pool) == 0:
            rets.loc[m] = 0.0; tno.loc[m] = 0.0; continue

        cand = st.loc[m, [c for c in pool if c in st.columns]].dropna()
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
# Stability Filter (daily momentum persistence)
# =========================
def stability_passers(daily_prices: pd.DataFrame, top_n: int, stability_days: int) -> List[str]:
    scores = blended_momentum_scores_daily(daily_prices)
    if scores.empty:
        return []
    last_dates = scores.index[-stability_days:]
    tops = []
    for d in last_dates:
        s = scores.loc[d].dropna()
        tops.append(set(s.nlargest(top_n).index) if not s.empty else set())
    stable = set.intersection(*tops) if all(len(t) > 0 for t in tops) else set()
    return sorted(list(stable))

# =========================
# Portfolio builders (MONTHLY LOCK + stability)
# =========================
def _mr_scores_daily_to_monthly(daily_prices: pd.DataFrame, lookback_days: int, long_ma_days: int) -> pd.Series:
    st_returns = daily_prices.pct_change(lookback_days).iloc[-1]
    long_trend = daily_prices.rolling(long_ma_days).mean().iloc[-1]
    quality = daily_prices.iloc[-1] > long_trend
    return st_returns[quality[quality].index].dropna()

def _build_classic_weights(daily_close: pd.DataFrame,
                           momentum_window: int, top_n: int, cap: float) -> pd.Series:
    monthly = daily_close.resample("ME").last()
    mom_scores = blended_momentum_scores(monthly)
    mom_scores = mom_scores[mom_scores > 0]
    top_performers = mom_scores.nlargest(top_n)
    if top_performers.empty:
        mom_w = pd.Series(dtype=float)
    else:
        raw = top_performers / top_performers.sum()
        mom_w = cap_weights(raw, cap=cap) * 0.90

    mr_scores = _mr_scores_daily_to_monthly(daily_close, 21, 200)
    dips = mr_scores.nsmallest(5)
    if dips.empty:
        mr_w = pd.Series(dtype=float)
    else:
        mr_w = pd.Series(1/len(dips), index=dips.index) * 0.10

    final = mom_w.add(mr_w, fill_value=0.0)
    return final / final.sum() if final.sum() > 0 else final

def _build_isa_weights(daily_close: pd.DataFrame, preset: Dict) -> pd.Series:
    monthly = daily_close.resample("ME").last()

    # Momentum sleeve with stability gating on entries
    mom_scores = blended_momentum_scores(monthly)
    mom_scores = mom_scores[mom_scores > 0]
    top_m = mom_scores.nlargest(preset["mom_topn"])
    stable = stability_passers(daily_close, top_n=preset["mom_topn"], stability_days=preset["stability_days"])
    if len(stable) > 0:
        top_m = top_m.reindex(stable).dropna()
        if top_m.empty:
            top_m = mom_scores.nlargest(preset["mom_topn"])

    mom_raw = (top_m / top_m.sum()) if top_m.sum() > 0 else pd.Series(dtype=float)
    mom_w = cap_weights(mom_raw, cap=preset["mom_cap"]) * preset["mom_w"] if not mom_raw.empty else pd.Series(dtype=float)

    # MR sleeve
    mr_scores = _mr_scores_daily_to_monthly(daily_close, preset["mr_lb"], preset["mr_ma"])
    dips = mr_scores.nsmallest(preset["mr_topn"])
    mr_w = (pd.Series(1/len(dips), index=dips.index) * preset["mr_w"]) if len(dips) > 0 else pd.Series(dtype=float)

    new_w = mom_w.add(mr_w, fill_value=0.0)
    return new_w / new_w.sum() if new_w.sum() > 0 else new_w

def _format_display(weights: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    display_df = pd.DataFrame({"Weight": weights}).sort_values("Weight", ascending=False)
    display_fmt = display_df.copy()
    display_fmt["Weight"] = display_fmt["Weight"].map("{:.2%}".format)
    return display_fmt, display_df

# =========================
# Live builders (now with universe)
# =========================
def generate_live_portfolio_classic(momentum_window: int, top_n: int, cap: float,
                                    universe_name: str = "NASDAQ100+",
                                    min_dollar_volume: float = 0.0) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    start = (datetime.today() - relativedelta(months=max(momentum_window, 12) + 8)).strftime("%Y-%m-%d")
    end   = _today_str()
    close, vol, final_list = get_universe_prices(universe_name, start, end,
                                                 min_dollar_volume=min_dollar_volume)
    if close.empty:
        return None, None
    w = _build_classic_weights(close, momentum_window, top_n, cap)
    return _format_display(w)

def generate_live_portfolio_isa_monthly(preset: Dict,
                                        prev_portfolio: Optional[pd.DataFrame],
                                        universe_name: str = "NASDAQ100+",
                                        min_dollar_volume: float = 0.0,
                                        hybrid_top_n: int = 150) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """
    ISA Dynamic live weights with MONTHLY LOCK + stability filter + trigger.
    If today is NOT the first trading day of the month, we **hold** the previous portfolio.
    """
    start = (datetime.today() - relativedelta(months=max(preset["mom_lb"], 12) + 8)).strftime("%Y-%m-%d")
    end   = _today_str()
    close, vol, tickers = get_universe_prices(universe_name, start, end,
                                              min_dollar_volume=min_dollar_volume,
                                              hybrid_top_n=hybrid_top_n)
    if close.empty:
        return None, None, "No price data."

    today = datetime.today().date()
    is_monthly = is_rebalance_today(today, close.index)
    decision = "Not the monthly rebalance day — holding previous portfolio."
    if not is_monthly:
        if prev_portfolio is not None and not prev_portfolio.empty:
            disp, raw = _format_display(prev_portfolio["Weight"])
            return disp, raw, decision
        else:
            decision = "No saved portfolio; proposing initial allocation (monthly lock will apply from next month)."

    # Build candidate weights
    new_w = _build_isa_weights(close, preset)

    # Trigger vs previous portfolio (only if we have previous)
    if prev_portfolio is not None and not prev_portfolio.empty and "Weight" in prev_portfolio.columns:
        monthly = close.resample("ME").last()
        mom_scores = blended_momentum_scores(monthly)
        if not mom_scores.empty and len(new_w) > 0:
            top_m = mom_scores.nlargest(preset["mom_topn"])
            top_score = float(top_m.iloc[0]) if len(top_m) > 0 else 1e-9
            prev_w = prev_portfolio["Weight"].astype(float)
            held_scores = mom_scores.reindex(prev_w.index).fillna(0.0)
            health = float((held_scores * prev_w).sum() / max(top_score, 1e-9))
            if health >= preset["trigger"]:
                decision = f"Health {health:.2f} ≥ trigger {preset['trigger']:.2f} — holding existing portfolio."
                return _format_display(prev_w)
            else:
                decision = f"Health {health:.2f} < trigger {preset['trigger']:.2f} — rebalancing to new targets."

    disp, raw = _format_display(new_w)
    return disp, raw, decision

# =========================
# Backtests for app (universe + costs)
# =========================
def run_backtest_for_app(momentum_window: int, top_n: int, cap: float,
                         universe_name: str = "NASDAQ100+",
                         roundtrip_bps: float = 0.0,
                         min_dollar_volume: float = 0.0,
                         hybrid_top_n: int = 150,
                         show_net: bool = False) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """Classic 90/10 hybrid vs QQQ (since 2018), with universe + liquidity + optional costs."""
    start_date = "2018-01-01"
    end_date = _today_str()

    # Universe data
    close, vol, tickers = get_universe_prices(universe_name, start_date, end_date,
                                              min_dollar_volume=min_dollar_volume,
                                              hybrid_top_n=hybrid_top_n)
    # Benchmark
    qqq = fetch_market_data(["QQQ"], start_date, end_date)
    if close.empty or qqq.empty or "QQQ" not in qqq.columns:
        return None, None, None, None

    daily = close
    qqq_s = qqq["QQQ"]

    mom_rets, mom_tno = run_backtest_gross(daily, momentum_window, top_n, cap)
    mr_rets,  mr_tno  = run_backtest_mean_reversion(daily, 21, 5, 200)
    hybrid_gross, hybrid_tno = combine_hybrid(mom_rets, mr_rets, mom_tno, mr_tno, mom_w=0.90, mr_w=0.10)
    hybrid_net = apply_costs(hybrid_gross, hybrid_tno, roundtrip_bps) if show_net else hybrid_gross

    strat_cum_gross = (1 + pd.Series(hybrid_gross).fillna(0)).cumprod()
    strat_cum_net   = (1 + pd.Series(hybrid_net).fillna(0)).cumprod()
    qqq_cum = (1 + qqq_s.resample("ME").last().pct_change()).cumprod()
    return strat_cum_gross, strat_cum_net, qqq_cum.reindex(strat_cum_gross.index, method="ffill"), hybrid_tno

def run_backtest_isa_dynamic(universe_name: str = "NASDAQ100+",
                             roundtrip_bps: float = 0.0,
                             min_dollar_volume: float = 0.0,
                             hybrid_top_n: int = 150,
                             show_net: bool = False) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """ISA preset backtest (since 2018) with universe + liquidity + costs."""
    params = STRATEGY_PRESETS["ISA Dynamic (0.75)"]
    start_date = "2018-01-01"
    end_date = _today_str()

    close, vol, tickers = get_universe_prices(universe_name, start_date, end_date,
                                              min_dollar_volume=min_dollar_volume,
                                              hybrid_top_n=hybrid_top_n)
    qqq = fetch_market_data(["QQQ"], start_date, end_date)
    if close.empty or qqq.empty or "QQQ" not in qqq.columns:
        return None, None, None, None

    daily = close
    qqq_s = qqq["QQQ"]

    monthly = daily.resample("ME").last()
    fwd = monthly.pct_change().shift(-1)

    st_ret = daily.pct_change(params["mr_lb"]).resample("ME").last()
    lt_ma  = daily.rolling(params["mr_ma"]).mean().resample("ME").last()

    portfolio_rets = pd.Series(index=monthly.index, dtype=float)
    turnover_series = pd.Series(index=monthly.index, dtype=float)
    prev_w = pd.Series(dtype=float)

    for m in monthly.index:
        mom_scores = blended_momentum_scores(monthly.loc[:m].iloc[-13:])
        mom_scores = mom_scores[mom_scores > 0]
        if mom_scores.empty:
            portfolio_rets.loc[m] = 0.0; turnover_series.loc[m] = 0.0; continue

        top = mom_scores.nlargest(params["mom_topn"])
        raw = top / top.sum()
        w_m = cap_weights(raw, cap=params["mom_cap"])

        # MR sleeve
        if m in st_ret.index and m in lt_ma.index:
            quality = monthly.loc[m] > lt_ma.loc[m]
            pool = quality[quality].index
            mr_pool = st_ret.loc[m, [c for c in pool if c in st_ret.columns]].dropna()
            dips = mr_pool.nsmallest(params["mr_topn"])
            w_r = pd.Series(1/len(dips), index=dips.index) if len(dips) > 0 else pd.Series(dtype=float)
        else:
            w_r = pd.Series(dtype=float)

        # Combine
        w = (w_m * params["mom_w"]).add(w_r * params["mr_w"], fill_value=0.0)
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
    qqq_cum = (1 + qqq_s.resample("ME").last().pct_change()).cumprod()
    return strat_cum_gross, strat_cum_net, qqq_cum.reindex(strat_cum_gross.index, method="ffill"), turnover_series

# =========================
# Diff engine (for Plan tab)
# =========================
def diff_portfolios(prev_df: Optional[pd.DataFrame],
                    curr_df: Optional[pd.DataFrame],
                    tol: float = 0.01) -> Dict[str, object]:
    if prev_df is None or prev_df.empty:
        prev_df = pd.DataFrame(columns=["Weight"])
    if curr_df is None or curr_df.empty:
        curr_df = pd.DataFrame(columns=["Weight"])

    prev_w = prev_df["Weight"] if "Weight" in prev_df.columns else pd.Series(dtype=float)
    curr_w = curr_df["Weight"] if "Weight" in curr_df.columns else pd.Series(dtype=float)

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
    return {"sell": sells, "buy": buys, "rebalance": rebalances}

# =========================
# Explainability (“what changed & why”)
# =========================
def _signal_snapshot_for_explain(daily_prices: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Build a one-row snapshot of signals used in the explanation table.
    Assumes daily_prices already restricted to available tickers.
    """
    if daily_prices is None or daily_prices.empty:
        return pd.DataFrame()

    monthly = daily_prices.resample("ME").last()
    if monthly.shape[0] < 13:
        return pd.DataFrame()

    # Keep a stable list of columns we actually have data for
    cols = monthly.columns.tolist()

    r3  = monthly.pct_change(3).iloc[-1]
    r6  = monthly.pct_change(6).iloc[-1]
    r12 = monthly.pct_change(12).iloc[-1]

    def z(s: pd.Series, cols_) -> pd.Series:
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty or s.std(ddof=0) == 0:
            return pd.Series(0.0, index=cols_)
        zs = (s - s.mean()) / (s.std(ddof=0) + 1e-9)
        return zs.reindex(cols_).fillna(0.0)

    mom_score = 0.2 * z(r3, cols) + 0.4 * z(r6, cols) + 0.4 * z(r12, cols)
    mom_rank  = mom_score.rank(ascending=False, method="min")

    st_ret   = daily_prices.pct_change(params["mr_lb"]).iloc[-1].reindex(cols)
    long_ma  = daily_prices.rolling(params["mr_ma"]).mean().iloc[-1].reindex(cols)
    last_px  = daily_prices.iloc[-1].reindex(cols)
    above_ma = (last_px > long_ma).astype(float)

    snap = pd.DataFrame({
        "mom_score": mom_score.reindex(cols),
        "mom_rank":  mom_rank.reindex(cols),
        f"st_ret_{params['mr_lb']}d": st_ret,
        f"above_{params['mr_ma']}dma": above_ma
    })
    return snap.sort_index()


def explain_portfolio_changes(prev_df: Optional[pd.DataFrame],
                              curr_df: Optional[pd.DataFrame],
                              daily_prices: pd.DataFrame,
                              params: Dict) -> pd.DataFrame:
    """
    Return a table describing buys/sells/rebalances with momentum & MR context.
    Safely ignores tickers missing from daily_prices.
    """
    # Normalize inputs
    prev_df = prev_df if prev_df is not None else pd.DataFrame(columns=["Weight"])
    curr_df = curr_df if curr_df is not None else pd.DataFrame(columns=["Weight"])
    prev_w = prev_df["Weight"].astype(float) if "Weight" in prev_df.columns else pd.Series(dtype=float)
    curr_w = curr_df["Weight"].astype(float) if "Weight" in curr_df.columns else pd.Series(dtype=float)

    all_tickers = sorted(set(prev_w.index) | set(curr_w.index))
    if not all_tickers:
        return pd.DataFrame(columns=[
            "Ticker","Action","Old Wt","New Wt","Δ Wt (bps)",
            "Mom Rank","Mom Score",f"ST Return ({params['mr_lb']}d)",f"Above {params['mr_ma']}DMA"
        ])

    if daily_prices is None or daily_prices.empty:
        # No price context available; still return the structural changes
        rows = []
        for t in all_tickers:
            old_w = float(prev_w.get(t, 0.0))
            new_w = float(curr_w.get(t, 0.0))
            if abs(new_w - old_w) < 1e-9:
                continue
            action = "Buy" if old_w == 0 and new_w > 0 else ("Sell" if new_w == 0 and old_w > 0 else "Rebalance")
            rows.append({
                "Ticker": t, "Action": action,
                "Old Wt": f"{old_w:.2%}", "New Wt": f"{new_w:.2%}",
                "Δ Wt (bps)": int(round((new_w - old_w) * 10000)),
                "Mom Rank": np.nan, "Mom Score": np.nan,
                f"ST Return ({params['mr_lb']}d)": np.nan,
                f"Above {params['mr_ma']}DMA": None
            })
        return pd.DataFrame(rows).sort_values("Δ Wt (bps)", ascending=False).reset_index(drop=True)

    # Only use tickers that actually exist in the price panel
    available = [t for t in all_tickers if t in daily_prices.columns]
    if len(available) == 0:
        # Nothing available; same fallback as above
        return explain_portfolio_changes(prev_df, curr_df, pd.DataFrame(), params)

    prices = daily_prices[available].copy()
    snap = _signal_snapshot_for_explain(prices, params)

    rows = []
    for t in all_tickers:
        old_w = float(prev_w.get(t, 0.0))
        new_w = float(curr_w.get(t, 0.0))
        if abs(new_w - old_w) < 1e-9:
            continue

        if old_w == 0 and new_w > 0:
            action = "Buy"
        elif new_w == 0 and old_w > 0:
            action = "Sell"
        else:
            action = "Rebalance"

        # Fill signal columns only if ticker had price data
        if t in snap.index:
            mom_rank = snap.at[t, "mom_rank"]
            mom_score = snap.at[t, "mom_score"]
            st_key = f"st_ret_{params['mr_lb']}d"
            stv = snap.at[t, st_key]
            above_key = f"above_{params['mr_ma']}dma"
            ab = bool(snap.at[t, above_key]) if pd.notna(snap.at[t, above_key]) else None
        else:
            mom_rank = np.nan; mom_score = np.nan; stv = np.nan; ab = None

        rows.append({
            "Ticker": t,
            "Action": action,
            "Old Wt": f"{old_w:.2%}",
            "New Wt": f"{new_w:.2%}",
            "Δ Wt (bps)": int(round((new_w - old_w) * 10000)),
            "Mom Rank": int(mom_rank) if pd.notna(mom_rank) else np.nan,
            "Mom Score": mom_score,
            f"ST Return ({params['mr_lb']}d)": stv,
            f"Above {params['mr_ma']}DMA": ab
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    action_order = pd.Categorical(out["Action"], categories=["Buy","Rebalance","Sell"], ordered=True)
    out = out.assign(ActionOrder=action_order) \
             .sort_values(["ActionOrder","Δ Wt (bps)"], ascending=[True, False]) \
             .drop(columns=["ActionOrder"]) \
             .reset_index(drop=True)
    return out
# =========================
# Regime & Live paper tracking
# =========================
def get_benchmark_series(ticker: str, start: str, end: str) -> pd.Series:
    px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px = _safe_series(px)
    return pd.Series(px).dropna()

def compute_regime_metrics(universe_prices_daily: pd.DataFrame) -> Dict[str, float]:
    if universe_prices_daily.empty:
        return {}
    start = (universe_prices_daily.index.min() - pd.DateOffset(days=5)).strftime("%Y-%m-%d")
    end   = (universe_prices_daily.index.max() + pd.DateOffset(days=5)).strftime("%Y-%m-%d")
    qqq = get_benchmark_series("QQQ", start, end).reindex(universe_prices_daily.index).ffill().dropna()

    pct_above_ma = (universe_prices_daily.iloc[-1] >
                    universe_prices_daily.rolling(REGIME_MA).mean().iloc[-1]).mean()

    qqq_ma = qqq.rolling(REGIME_MA).mean()
    qqq_above_ma = float(qqq.iloc[-1] > qqq_ma.iloc[-1]) if len(qqq_ma.dropna()) else np.nan

    qqq_vol_10d = qqq.pct_change().rolling(10).std().iloc[-1]
    qqq_slope_50 = (qqq.rolling(50).mean().iloc[-1] / qqq.rolling(50).mean().iloc[-10] - 1) if len(qqq) > 60 else np.nan

    monthly = universe_prices_daily.resample("ME").last()
    pos_6m = (monthly.pct_change(6).iloc[-1] > 0).mean()

    return {
        "universe_above_200dma": float(pct_above_ma),
        "qqq_above_200dma": float(qqq_above_ma),
        "qqq_vol_10d": float(qqq_vol_10d),
        "breadth_pos_6m": float(pos_6m),
        "qqq_50dma_slope_10d": float(qqq_slope_50) if pd.notna(qqq_slope_50) else np.nan
    }

def get_market_regime() -> Tuple[str, Dict[str, float]]:
    """Returns (label, metrics) using compute_regime_metrics on NASDAQ100+ as base."""
    start = (datetime.today() - relativedelta(months=12)).strftime("%Y-%m-%d")
    end   = _today_str()
    univ = get_nasdaq_100_plus_tickers()
    close, _ = fetch_price_volume(univ, start, end)
    metrics = compute_regime_metrics(close)
    if not metrics:
        return "Unknown", {}
    breadth = metrics.get("breadth_pos_6m", np.nan)
    qqq_abv = metrics.get("qqq_above_200dma", 0.0) >= 1.0
    vol10   = metrics.get("qqq_vol_10d", np.nan)

    label = "Risk-On"
    if ((not qqq_abv and (breadth < 0.35)) or (vol10 > 0.035 and not qqq_abv)):
        label = "Extreme Risk-Off"
    elif (not qqq_abv) or (breadth < 0.45):
        label = "Risk-Off"
    return label, metrics

def load_live_perf() -> pd.DataFrame:
    if GIST_API_URL and GITHUB_TOKEN:
        try:
            resp = requests.get(GIST_API_URL, headers=HEADERS)
            resp.raise_for_status()
            files = resp.json().get("files", {})
            content = files.get(LIVE_PERF_FILE, {}).get("content", "")
            if not content:
                return pd.DataFrame(columns=["date","strat_ret","qqq_ret","note"])
            df = pd.read_csv(io.StringIO(content))
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["date","strat_ret","qqq_ret","note"])

def save_live_perf(df: pd.DataFrame) -> None:
    if not GIST_API_URL or not GITHUB_TOKEN:
        return
    try:
        csv_str = df.to_csv(index=False)
        payload = {"files": {LIVE_PERF_FILE: {"content": csv_str}}}
        resp = requests.patch(GIST_API_URL, headers=HEADERS, json=payload)
        resp.raise_for_status()
    except Exception as e:
        st.sidebar.warning(f"Could not save live perf: {e}")

def calc_one_day_live_return(weights: pd.Series, daily_prices: pd.DataFrame) -> float:
    if weights is None or weights.empty or daily_prices.empty:
        return 0.0
    aligned = daily_prices[weights.index.intersection(daily_prices.columns)].dropna().iloc[-2:]
    if len(aligned) < 2:
        return 0.0
    rets = aligned.pct_change().iloc[-1]
    return float((rets * weights.reindex(aligned.columns).fillna(0.0)).sum())

def record_live_snapshot(weights_df: pd.DataFrame, note: str = "") -> Dict[str, object]:
    try:
        universe = get_nasdaq_100_plus_tickers()
        end_date = _today_str()
        start_date = (datetime.today() - relativedelta(days=40)).strftime("%Y-%m-%d")
        px = fetch_market_data(universe + ["QQQ"], start_date, end_date)
        if px.empty or "QQQ" not in px.columns:
            return {"ok": False, "msg": "No price data for live snapshot."}

        weights = weights_df["Weight"].astype(float)
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        weights = weights.reindex(px.columns).dropna()
        strat_1d = calc_one_day_live_return(weights, px[weights.index])
        qqq_1d   = px["QQQ"].pct_change().iloc[-1]

        log = load_live_perf()
        new_row = pd.DataFrame([{
            "date": pd.to_datetime(px.index[-1]).normalize(),
            "strat_ret": strat_1d,
            "qqq_ret": float(qqq_1d),
            "note": note
        }])
        out = pd.concat([log, new_row], ignore_index=True).drop_duplicates(subset=["date"], keep="last")
        save_live_perf(out)
        return {"ok": True, "strat_ret": strat_1d, "qqq_ret": float(qqq_1d), "rows": len(out)}
    except Exception as e:
        return {"ok": False, "msg": str(e)}

def get_live_equity() -> pd.DataFrame:
    log = load_live_perf().sort_values("date")
    if log.empty:
        return pd.DataFrame(columns=["date","strat_eq","qqq_eq"])
    df = log.copy()
    df["strat_eq"] = (1 + df["strat_ret"].fillna(0)).cumprod()
    df["qqq_eq"]   = (1 + df["qqq_ret"].fillna(0)).cumprod()
    return df[["date","strat_eq","qqq_eq"]]
