# backend.py — Enhanced Hybrid Top150 / Composite Rank / Sector Caps / Stickiness / ISA lock
import os, io, warnings
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

# =========================
# Config & Secrets
# =========================
GIST_ID = st.secrets.get("GIST_ID")
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN")
GIST_API_URL = f"https://api.github.com/gists/{GIST_ID}" if GIST_ID else None
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

GIST_PORTF_FILE = "portfolio.json"
LIVE_PERF_FILE  = "live_perf.csv"
LOCAL_PORTF_FILE = "last_portfolio.csv"

ROUNDTRIP_BPS_DEFAULT = 20
REGIME_MA = 200

# ISA preset
STRATEGY_PRESETS = {
    "ISA Dynamic (0.75)": {
        "mom_lb": 15, "mom_topn": 8, "mom_cap": 0.25,
        "mr_lb": 21,  "mr_topn": 3, "mr_ma": 200,
        "mom_w": 0.85, "mr_w": 0.15,
        "trigger": 0.75,
        "stability_days": 7,   # stickiness default
        "sector_cap": 0.30     # sector cap default
    }
}

# =========================
# NEW: Enhanced Data Validation
# =========================
def validate_market_data(prices_df: pd.DataFrame) -> List[str]:
    """Comprehensive data quality checks"""
    alerts = []
    
    if prices_df.empty:
        return ["No market data available"]
    
    # Check for suspicious price moves (>20% daily moves)
    daily_moves = prices_df.pct_change().abs()
    extreme_moves = daily_moves > 0.20
    
    if extreme_moves.any().any():
        extreme_count = extreme_moves.sum().sum()
        alerts.append(f"Warning: {extreme_count} extreme price moves (>20%) detected")
    
    # Check for stale data
    latest_date = prices_df.index.max()
    days_stale = (pd.Timestamp.now() - latest_date).days
    if days_stale > 3:
        alerts.append(f"Warning: Price data is {days_stale} days old")
    
    # Check data completeness
    missing_pct = prices_df.isnull().sum().sum() / (len(prices_df) * len(prices_df.columns)) * 100
    if missing_pct > 5:
        alerts.append(f"Warning: {missing_pct:.1f}% missing data points")
    
    return alerts

# =========================
# NEW: Volatility-Adjusted Position Sizing
# =========================
def get_volatility_adjusted_caps(weights: pd.Series, daily_prices: pd.DataFrame, 
                                lookback: int = 63, base_cap: float = 0.25) -> Dict[str, float]:
    """Adjust position caps based on individual stock volatility"""
    if daily_prices.empty or len(daily_prices) < lookback:
        return {ticker: base_cap for ticker in weights.index}
    
    # Calculate trailing volatilities
    returns = daily_prices.pct_change().dropna()
    if len(returns) < lookback:
        return {ticker: base_cap for ticker in weights.index}
        
    vols = returns.rolling(lookback).std().iloc[-1]
    median_vol = vols.median()
    
    adj_caps = {}
    for ticker in weights.index:
        if ticker not in vols.index:
            adj_caps[ticker] = base_cap
            continue
            
        stock_vol = vols[ticker]
        if pd.isna(stock_vol) or stock_vol <= 0:
            adj_caps[ticker] = base_cap
            continue
        
        # Scale cap inversely to volatility (lower vol = higher cap allowed)
        vol_ratio = median_vol / stock_vol
        vol_adjustment = np.clip(vol_ratio, 0.7, 1.3)  # 30% adjustment max
        
        adjusted_cap = base_cap * vol_adjustment
        adj_caps[ticker] = min(adjusted_cap, 0.35)  # Never exceed 35%
    
    return adj_caps

# =========================
# NEW: Signal Decay Modeling
# =========================
def apply_signal_decay(momentum_scores: pd.Series, signal_age_days: int = 0, 
                      half_life: int = 45) -> pd.Series:
    """Apply exponential decay to momentum signals based on age"""
    if signal_age_days <= 0:
        return momentum_scores
    
    decay_factor = 0.5 ** (signal_age_days / half_life)
    return momentum_scores * decay_factor

# =========================
# NEW: Regime-Adjusted Position Sizing
# =========================
def get_regime_adjusted_exposure(regime_metrics: Dict[str, float]) -> float:
    """Scale overall portfolio exposure based on market regime"""
    breadth = regime_metrics.get('breadth_pos_6m', 0.5)
    vol_regime = regime_metrics.get('qqq_vol_10d', 0.02)
    qqq_above_ma = regime_metrics.get('qqq_above_200dma', 1.0)
    
    # Base exposure
    exposure = 1.0
    
    # Adjust for breadth
    if breadth < 0.35:  # Very poor breadth
        exposure *= 0.7
    elif breadth < 0.45:  # Poor breadth
        exposure *= 0.85
    elif breadth > 0.65:  # Good breadth
        exposure *= 1.0
    else:  # Mixed breadth
        exposure *= 0.95
    
    # Adjust for volatility
    if vol_regime > 0.035:  # High volatility
        exposure *= 0.8
    elif vol_regime < 0.015:  # Low volatility
        exposure *= 1.05
    
    # Adjust for trend
    if qqq_above_ma < 1.0:  # Below 200DMA
        exposure *= 0.9
    
    return np.clip(exposure, 0.3, 1.2)  # Keep between 30% and 120%

# =========================
# NEW: Enhanced Drawdown Controls
# =========================
def get_drawdown_adjusted_exposure(current_returns: pd.Series, max_dd_threshold: float = 0.15) -> float:
    """Reduce exposure during sustained drawdowns"""
    if current_returns.empty:
        return 1.0
    
    equity_curve = (1 + current_returns.fillna(0)).cumprod()
    current_dd = (equity_curve / equity_curve.cummax() - 1).iloc[-1]
    
    if current_dd < -max_dd_threshold:
        # Scale down exposure based on drawdown severity
        dd_severity = abs(current_dd)
        exposure_reduction = min(dd_severity * 0.5, 0.7)  # Max 70% reduction
        return max(1.0 - exposure_reduction, 0.3)  # Never below 30%
    
    return 1.0

# =========================
# NEW: Portfolio Correlation Monitoring
# =========================
def calculate_portfolio_correlation_to_market(portfolio_returns: pd.Series, 
                                            market_returns: pd.Series = None) -> float:
    """Calculate portfolio correlation to market benchmark"""
    if market_returns is None:
        # Fetch QQQ data for correlation
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - relativedelta(months=6)).strftime('%Y-%m-%d')
            qqq_data = yf.download('QQQ', start=start_date, end=end_date, auto_adjust=True, progress=False)['Close']
            market_returns = qqq_data.pct_change().dropna()
        except:
            return np.nan
    
    # Align periods
    common_dates = portfolio_returns.index.intersection(market_returns.index)
    if len(common_dates) < 20:  # Need minimum periods
        return np.nan
    
    port_aligned = portfolio_returns.reindex(common_dates)
    market_aligned = market_returns.reindex(common_dates)
    
    correlation = port_aligned.corr(market_aligned)
    return correlation if not pd.isna(correlation) else 0.0

# =========================
# Helpers (Enhanced)
# =========================
def _safe_series(obj):
    return obj.squeeze() if isinstance(obj, pd.DataFrame) else obj

def to_yahoo_symbol(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-")

# =========================
# Universe builders & sectors (Enhanced with validation)
# =========================
@st.cache_data(ttl=86400)
def fetch_sp500_constituents() -> List[str]:
    """Get current S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        return sorted(list(set(tickers)))
    except Exception as e:
        st.warning(f"Failed to fetch S&P 500 list: {e}")
        return []

@st.cache_data(ttl=86400)
def get_sector_map(tickers: List[str]) -> Dict[str, str]:
    """Best-effort sector map via yfinance. Returns 'Unknown' when missing."""
    out = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info or {}
            sector = info.get("sector") or "Unknown"
        except Exception:
            sector = "Unknown"
        out[t] = sector
    return out

def get_nasdaq_100_plus_tickers() -> List[str]:
    """Get NASDAQ 100+ tickers including extras"""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        nasdaq_100 = tables[4]['Ticker'].astype(str).str.upper().tolist()
        extras = ['TSLA', 'SHOP', 'SNOW', 'PLTR', 'ETSY', 'RIVN', 'COIN']
        return sorted(set(nasdaq_100 + extras))
    except Exception:
        return ['AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA']  # Fallback

def get_universe(choice: str) -> Tuple[List[str], Dict[str, str], str]:
    """
    Returns (tickers, sectors_map, label) for:
      - 'NASDAQ100+'
      - 'S&P500 (All)'
      - 'Hybrid Top150'
    """
    choice = (choice or "").strip()
    if choice.lower().startswith("nasdaq"):
        tickers = get_nasdaq_100_plus_tickers()
        sectors = get_sector_map(tickers)
        return tickers, sectors, "NASDAQ100+"

    if choice.lower().startswith("s&p500"):
        tickers = fetch_sp500_constituents()
        sectors = get_sector_map(tickers)
        return tickers, sectors, "S&P500 (All)"

    # Default to Hybrid Top150
    tickers = fetch_sp500_constituents()
    sectors = get_sector_map(tickers)
    return tickers, sectors, "Hybrid Top150"

# =========================
# Universe prep for backtests (Enhanced)
# =========================
def _prepare_universe_for_backtest(
    universe_choice: str,
    start_date: str,
    end_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str], str]:
    """Fetches prices/volumes, applies Hybrid150 filter if needed."""
    base_tickers, base_sectors, label = get_universe(universe_choice)
    if not base_tickers:
        return pd.DataFrame(), pd.DataFrame(), {}, label

    close, vol = fetch_price_volume(base_tickers + ["QQQ"], start_date, end_date)
    if close.empty or "QQQ" not in close.columns:
        return pd.DataFrame(), pd.DataFrame(), {}, label

    # Validate data quality
    alerts = validate_market_data(close)
    if alerts:
        for alert in alerts[:3]:  # Show max 3 alerts
            st.warning(alert)

    sectors_map = {t: base_sectors.get(t, "Unknown") for t in close.columns if t != "QQQ"}

    if label == "Hybrid Top150":
        med = median_dollar_volume(
            close.drop(columns=["QQQ"], errors="ignore"),
            vol.drop(columns=["QQQ"], errors="ignore"),
            window=60
        ).sort_values(ascending=False)
        top_list = med.head(150).index.tolist()
        keep_cols = [c for c in top_list if c in close.columns]
        if keep_cols:
            close = close[keep_cols + ["QQQ"]]
            vol   = vol[keep_cols + ["QQQ"]]
            sectors_map = {t: sectors_map.get(t, "Unknown") for t in keep_cols}

    for _df in (close, vol):
        idx = pd.to_datetime(_df.index)
        if getattr(idx, "tz", None) is not None:
            _df.index = idx.tz_localize(None)

    return close, vol, sectors_map, label

# =========================
# Data fetching (cache) - Enhanced with validation
# =========================
@st.cache_data(ttl=43200)
def fetch_market_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    try:
        fetch_start = (pd.to_datetime(start_date) - pd.DateOffset(months=14)).strftime("%Y-%m-%d")
        data = yf.download(tickers, start=fetch_start, end=end_date, auto_adjust=True, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        result = data.dropna(axis=1, how="all")
        
        # Validate the fetched data
        alerts = validate_market_data(result)
        if alerts:
            st.info(f"Data quality check: {alerts[0]}")  # Show first alert only
        
        return result
    except Exception as e:
        st.error(f"Failed to download market data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=43200)
def fetch_price_volume(tickers: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        fetch_start = (pd.to_datetime(start_date) - pd.DateOffset(months=14)).strftime("%Y-%m-%d")
        df = yf.download(tickers, start=fetch_start, end=end_date, auto_adjust=True, progress=False)[["Close","Volume"]]
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"]
            vol   = df["Volume"]
        else:
            if "Close" in df.columns and "Volume" in df.columns:
                close = df[["Close"]].copy()
                vol   = df[["Volume"]].copy()
                if len(tickers) == 1:
                    close.columns = [tickers[0]]
                    vol.columns   = [tickers[0]]
            else:
                return pd.DataFrame(), pd.DataFrame()
        close = close.dropna(axis=1, how="all")
        vol   = vol.reindex_like(close).fillna(0)
        
        # Validate data
        alerts = validate_market_data(close)
        if alerts and len(alerts) > 0:
            st.info(f"Price data validation: {alerts[0]}")
        
        return close, vol
    except Exception as e:
        st.error(f"Failed to download price/volume: {e}")
        return pd.DataFrame(), pd.DataFrame()
        
# =========================
# Persistence (Gist + Local) - Unchanged
# =========================
def save_portfolio_to_gist(portfolio_df: pd.DataFrame) -> None:
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
    # Gist first
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
    try:
        out = df.copy()
        if out.index.name is None:
            out.index.name = "Ticker"
        out.reset_index().to_csv(LOCAL_PORTF_FILE, index=False)
    except Exception as e:
        st.sidebar.warning(f"Could not save local portfolio: {e}")

# =========================
# Calendar helpers (Monthly Lock) - Unchanged
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
# Math utils & KPIs (Enhanced)
# =========================
def cap_weights(weights: pd.Series, cap: float = 0.25, 
                vol_adjusted_caps: Optional[Dict[str, float]] = None) -> pd.Series:
    """Enhanced cap_weights with optional volatility adjustments"""
    if weights.empty:
        return weights
    w = weights.copy().astype(float)
    
    # Use volatility-adjusted caps if provided
    caps_to_use = vol_adjusted_caps if vol_adjusted_caps is not None else {ticker: cap for ticker in w.index}
    
    for _ in range(100):
        over_cap = pd.Series(False, index=w.index)
        for ticker in w.index:
            ticker_cap = caps_to_use.get(ticker, cap)
            if w[ticker] > ticker_cap:
                over_cap[ticker] = True
        
        if not over_cap.any():
            break
            
        excess = 0.0
        for ticker in w.index:
            if over_cap[ticker]:
                ticker_cap = caps_to_use.get(ticker, cap)
                excess += w[ticker] - ticker_cap
                w[ticker] = ticker_cap
        
        under = ~over_cap
        if w[under].sum() > 0:
            w[under] += (w[under] / w[under].sum()) * excess
        else:
            w += excess / len(w)
    return w

def apply_sector_caps(w: pd.Series, sectors_map: Dict[str,str], cap: float = 0.30, max_iter: int = 50) -> pd.Series:
    if w.empty: return w
    w = w.copy().astype(float)
    sectors = pd.Series({t: sectors_map.get(t, "Unknown") for t in w.index})
    for _ in range(max_iter):
        sums = w.groupby(sectors).sum()
        over = sums[sums > cap]
        if over.empty: break
        total_excess = 0.0
        for sec, sec_sum in over.items():
            idx = w.index[sectors == sec]
            scale = cap / sec_sum if sec_sum > 0 else 1.0
            new_sec = w.loc[idx] * scale
            total_excess += (w.loc[idx].sum() - new_sec.sum())
            w.loc[idx] = new_sec
        under_idx = w.index[~sectors.isin(over.index)]
        if len(under_idx) == 0 or w.loc[under_idx].sum() <= 0:
            w += total_excess / len(w)
        else:
            w.loc[under_idx] += (w.loc[under_idx] / w.loc[under_idx].sum()) * total_excess
    return w / w.sum() if w.sum() > 0 else w

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

    idx = pd.to_datetime(r.index)
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

    tpy = 0.0
    if turnover_series is not None and len(turnover_series) > 0:
        ts = pd.Series(turnover_series).copy()
        ts.index = pd.to_datetime(ts.index)
        yearly_sum = ts.groupby(ts.index.year).sum()
        if len(yearly_sum) > 0:
            tpy = float(yearly_sum.mean())

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
# Composite Signals (mom + trend + lowvol) + Stickiness (Enhanced)
# =========================
def blended_momentum_z(monthly: pd.DataFrame) -> pd.Series:
    if monthly.shape[0] < 13: return pd.Series(dtype=float)
    r3, r6, r12 = monthly.pct_change(3).iloc[-1], monthly.pct_change(6).iloc[-1], monthly.pct_change(12).iloc[-1]
    def z(s, cols):
        s = s.replace([np.inf,-np.inf], np.nan).dropna()
        if s.empty or s.std(ddof=0)==0: return pd.Series(0.0, index=cols)
        return ((s - s.mean()) / (s.std(ddof=0) + 1e-9)).reindex(cols).fillna(0.0)
    cols = monthly.columns
    return 0.2*z(r3,cols) + 0.4*z(r6,cols) + 0.4*z(r12,cols)

def lowvol_z(daily: pd.DataFrame) -> pd.Series:
    if daily.shape[0] < 80: return pd.Series(0.0, index=daily.columns)
    vol = daily.pct_change().rolling(63).std().iloc[-1].replace([np.inf,-np.inf], np.nan).dropna()
    if vol.empty or vol.std(ddof=0)==0: return pd.Series(0.0, index=daily.columns)
    z = (vol - vol.mean()) / (vol.std(ddof=0) + 1e-9)
    return z.reindex(daily.columns).fillna(0.0)

def trend_z(daily: pd.DataFrame) -> pd.Series:
    if daily.shape[0] < 220: return pd.Series(0.0, index=daily.columns)
    ma200 = daily.rolling(200).mean().iloc[-1]; last = daily.iloc[-1]
    dist = (last/ma200 - 1).replace([np.inf,-np.inf], np.nan).dropna()
    if dist.empty or dist.std(ddof=0)==0: return pd.Series(0.0, index=daily.columns)
    z = (dist - dist.mean()) / (dist.std(ddof=0) + 1e-9)
    return z.reindex(daily.columns).fillna(0.0)

def composite_score(daily: pd.DataFrame) -> pd.Series:
    monthly = daily.resample("ME").last()
    momz = blended_momentum_z(monthly)
    lvz  = lowvol_z(daily)
    tz   = trend_z(daily)
    return (0.6*momz.add(0.0, fill_value=0.0) + 0.2*(-lvz) + 0.2*tz).dropna()

def momentum_stable_names(daily: pd.DataFrame, top_n: int, days: int) -> List[str]:
    if daily.shape[0] < (days + 260): return []
    r63, r126, r252 = daily.pct_change(63), daily.pct_change(126), daily.pct_change(252)
    def zrow(df):
        mu = df.mean(axis=1); sd = df.std(axis=1).replace(0,np.nan)
        return (df.sub(mu,axis=0)).div(sd,axis=0).fillna(0.0)
    mscore = 0.2*zrow(r63) + 0.4*zrow(r126) + 0.4*zrow(r252)
    if mscore.shape[0] < days: return []
    tops = []
    for d in mscore.index[-days:]:
        s = mscore.loc[d].dropna()
        tops.append(set(s.nlargest(top_n).index) if not s.empty else set())
    return sorted(list(set.intersection(*tops))) if all(len(t)>0 for t in tops) else []

# =========================
# Sleeves (composite momentum + MR) with turnover (Enhanced)
# =========================
def run_momentum_composite_param(daily: pd.DataFrame, sectors_map: dict,
                                 top_n=8, name_cap=0.25, sector_cap=0.30, stickiness_days=7,
                                 use_enhanced_features=True):
    """Enhanced momentum composite with volatility-adjusted caps and regime awareness"""
    monthly = daily.resample("ME").last()
    fwd = monthly.pct_change().shift(-1)
    rets = pd.Series(index=monthly.index, dtype=float)
    tno  = pd.Series(index=monthly.index, dtype=float)
    prev_w = pd.Series(dtype=float)
    
    for m in monthly.index:
        hist = daily.loc[:m]
        comp = composite_score(hist)
        if comp.empty: 
            rets.loc[m]=0.0; tno.loc[m]=0.0; continue
            
        momz = blended_momentum_z(hist.resample("ME").last())
        comp = comp[momz.index]
        sel  = comp[momz > 0].dropna()
        if sel.empty: 
            rets.loc[m]=0.0; tno.loc[m]=0.0; continue
            
        picks = sel.nlargest(top_n)
        
        # Apply signal decay if enabled
        if use_enhanced_features:
            days_since_signal = 0  # Could be enhanced to track actual signal age
            picks = apply_signal_decay(picks, days_since_signal)
        
        # Stickiness filter
        stable = momentum_stable_names(hist, top_n=top_n, days=stickiness_days)
        if len(stable)>0:
            picks = picks.reindex([t for t in picks.index if t in stable]).dropna()
            if picks.empty: 
                picks = sel.nlargest(top_n)
        
        raw = (picks / picks.sum()).astype(float)
        
        # Enhanced position sizing with volatility adjustment
        if use_enhanced_features:
            vol_caps = get_volatility_adjusted_caps(raw, hist, base_cap=name_cap)
            raw = cap_weights(raw, cap=name_cap, vol_adjusted_caps=vol_caps)
        else:
            raw = cap_weights(raw, cap=name_cap)
            
        w = apply_sector_caps(raw, sectors_map, cap=sector_cap)
        
        # Apply regime-based exposure adjustment
        if use_enhanced_features:
            try:
                regime_metrics = compute_regime_metrics(hist)
                regime_exposure = get_regime_adjusted_exposure(regime_metrics)
                w = w * regime_exposure
            except:
                pass  # Fall back to original weights if regime calc fails
        
        valid = [t for t in w.index if t in fwd.columns]
        rets.loc[m] = float((fwd.loc[m, valid] * w.reindex(valid)).sum())
        tno.loc[m]  = 0.5 * float((w.reindex(prev_w.index, fill_value=0.0) - prev_w.reindex(w.index, fill_value=0.0)).abs().sum())
        prev_w = w
        
    return rets.fillna(0.0), tno.fillna(0.0)

def run_backtest_mean_reversion(daily_prices: pd.DataFrame,
                                lookback_period_mr: int = 21,
                                top_n_mr: int = 3,
                                long_ma_days: int = 200) -> Tuple[pd.Series, pd.Series]:
    monthly = daily_prices.resample("ME").last()
    fwd = monthly.pct_change().shift(-1)
    st = daily_prices.pct_change(lookback_period_mr).resample("ME").last()
    lt = daily_prices.rolling(long_ma_days).mean().resample("ME").last()
    rets = pd.Series(index=monthly.index, dtype=float)
    tno  = pd.Series(index=monthly.index, dtype=float)
    prev_w = pd.Series(dtype=float)
    for m in monthly.index:
        if m not in st.index or m not in lt.index:
            rets.loc[m]=0.0; tno.loc[m]=0.0; continue
        quality = monthly.loc[m] > lt.loc[m]
        pool = quality[quality].index
        cand = st.loc[m, [t for t in pool if t in st.columns]].dropna()
        dips = cand.nsmallest(top_n_mr)
        if dips.empty: rets.loc[m]=0.0; tno.loc[m]=0.0; continue
        w = pd.Series(1/len(dips), index=dips.index)
        valid = [t for t in w.index if t in fwd.columns]
        rets.loc[m] = float((fwd.loc[m, valid] * w.reindex(valid)).sum())
        tno.loc[m]  = 0.5 * float((w.reindex(prev_w.index, fill_value=0.0) - prev_w.reindex(w.index, fill_value=0.0)).abs().sum())
        prev_w = w
    return rets.fillna(0.0), tno.fillna(0.0)

def combine_hybrid(mom_rets: pd.Series, mr_rets: pd.Series,
                   mom_tno: Optional[pd.Series] = None,
                   mr_tno: Optional[pd.Series] = None,
                   mw: float = 0.9, rw: float = 0.1,
                   mom_w: Optional[float] = None,
                   mr_w: Optional[float] = None) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Combine momentum & mean-reversion sleeves. Supports both (mw, rw) and (mom_w, mr_w).
    Returns (combined_returns, combined_turnover or None).
    """
    # Backward/forward compatibility for arg names
    if mom_w is not None:
        mw = mom_w
    if mr_w is not None:
        rw = mr_w

    mom = pd.Series(mom_rets).copy()
    mr  = pd.Series(mr_rets).reindex(mom.index, fill_value=0.0)
    mom = mom.reindex(mr.index, fill_value=0.0)

    combo = (mom * mw) + (mr * rw)

    if mom_tno is None and mr_tno is None:
        return combo, None

    mtn = pd.Series(mom_tno).reindex(combo.index, fill_value=0.0) if mom_tno is not None else pd.Series(0.0, index=combo.index)
    rtn = pd.Series(mr_tno ).reindex(combo.index, fill_value=0.0) if mr_tno  is not None else pd.Series(0.0, index=combo.index)
    tno = (mtn * mw) + (rtn * rw)
    return combo, tno

def apply_costs(gross, turnover, roundtrip_bps):
    return gross - turnover*(roundtrip_bps/10000.0)

# =========================
# Liquidity utils (Hybrid150) - Unchanged
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
# Live portfolio builders (ISA MONTHLY LOCK + stickiness + sector caps) - Enhanced
# =========================
def _build_isa_weights(daily_close: pd.DataFrame, preset: Dict, sectors_map: Dict[str, str]) -> pd.Series:
    monthly = daily_close.resample("ME").last()

    # --- Composite momentum (ticker vector at latest date) ---
    comp_all = composite_score(daily_close)
    if isinstance(comp_all, pd.DataFrame):
        comp_vec = comp_all.iloc[-1].dropna()
    else:
        comp_vec = pd.Series(comp_all).dropna()

    # --- Long-term blended momentum z-score (month-end) ---
    momz = blended_momentum_z(monthly)  # Series indexed by ticker
    pos_idx = momz[momz > 0].index

    # Filter composite by positive momentum universe
    comp_vec = comp_vec.reindex(pos_idx).dropna()
    top_m = comp_vec.nlargest(preset["mom_topn"]) if not comp_vec.empty else pd.Series(dtype=float)

    # --- Stability (stickiness) filter on entries ---
    stable_names = set(momentum_stable_names(
        daily_close, top_n=preset["mom_topn"], days=preset.get("stability_days", 7)
    ))
    if stable_names and not top_m.empty:
        filtered = top_m.reindex([t for t in top_m.index if t in stable_names]).dropna()
        if not filtered.empty:
            top_m = filtered  # only replace if something remains

    # Weights for momentum sleeve (with enhanced name cap + sector caps)
    if not top_m.empty and top_m.sum() > 0:
        mom_raw = top_m / top_m.sum()
        
        # Apply volatility-adjusted caps
        vol_caps = get_volatility_adjusted_caps(mom_raw, daily_close, base_cap=preset["mom_cap"])
        mom_w = cap_weights(mom_raw, cap=preset["mom_cap"], vol_adjusted_caps=vol_caps) * preset["mom_w"]
        
        if not mom_w.empty:
            mom_w = apply_sector_caps(mom_w, sectors_map, cap=preset.get("sector_cap", 0.30))
    else:
        mom_w = pd.Series(dtype=float)

    # --- Mean-Reversion sleeve (quality + worst ST) ---
    st_ret = daily_close.pct_change(preset["mr_lb"]).iloc[-1]
    long_ma = daily_close.rolling(preset["mr_ma"]).mean().iloc[-1]
    quality = (daily_close.iloc[-1] > long_ma)
    pool = [t for t, ok in quality.items() if ok]
    mr_scores = st_ret.reindex(pool).dropna()
    dips = mr_scores.nsmallest(preset["mr_topn"])
    mr_w = (pd.Series(1 / len(dips), index=dips.index) * preset["mr_w"]) if len(dips) > 0 else pd.Series(dtype=float)

    # --- Combine sleeves & normalize ---
    new_w = mom_w.add(mr_w, fill_value=0.0)
    
    # Apply regime-based exposure adjustment
    try:
        regime_metrics = compute_regime_metrics(daily_close)
        regime_exposure = get_regime_adjusted_exposure(regime_metrics)
        new_w = new_w * regime_exposure
    except:
        pass  # Fall back to original weights if regime calc fails
    
    return new_w / new_w.sum() if new_w.sum() > 0 else new_w


def _format_display(weights: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    display_df = pd.DataFrame({"Weight": weights}).sort_values("Weight", ascending=False)
    display_fmt = display_df.copy()
    display_fmt["Weight"] = display_fmt["Weight"].map("{:.2%}".format)
    return display_fmt, display_df

def generate_live_portfolio_isa_monthly(
    preset: Dict,
    prev_portfolio: Optional[pd.DataFrame],
    min_dollar_volume: float = 0.0
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """
    Enhanced ISA Dynamic live weights with MONTHLY LOCK + composite + stability + sector caps.
    Now includes regime awareness and volatility adjustments.
    """
    universe_choice = st.session_state.get("universe", "Hybrid Top150")
    stickiness_days = st.session_state.get("stickiness_days", preset.get("stability_days", 7))
    sector_cap      = st.session_state.get("sector_cap", preset.get("sector_cap", 0.30))

    # build params from preset, then override with UI/session values
    params = dict(STRATEGY_PRESETS["ISA Dynamic (0.75)"])
    params["stability_days"] = int(stickiness_days)
    params["sector_cap"]     = float(sector_cap)
    params["mom_cap"]        = float(st.session_state.get("name_cap", params.get("mom_cap", 0.25)))
    
    # Universe base tickers + sectors
    base_tickers, base_sectors, label = get_universe(universe_choice)
    if not base_tickers:
        return None, None, "No universe available."

    # Fetch prices
    start = (datetime.today() - relativedelta(months=max(preset["mom_lb"], 12) + 8)).strftime("%Y-%m-%d")
    end   = datetime.today().strftime("%Y-%m-%d")
    close, vol = fetch_price_volume(base_tickers, start, end)
    if close.empty:
        return None, None, "No price data."

    # Special: Hybrid Top150 → reduce by 60d median dollar volume
    sectors_map = base_sectors.copy()
    if label == "Hybrid Top150":
        med = median_dollar_volume(close, vol, window=60).sort_values(ascending=False)
        top_list = med.head(150).index.tolist()
        close = close[top_list]
        vol   = vol[top_list]
        sectors_map = {t: base_sectors.get(t, "Unknown") for t in close.columns}

    # Liquidity floor (optional)
    if min_dollar_volume > 0:
        keep = filter_by_liquidity(close, vol, min_dollar_volume)
        if not keep:
            return None, None, "No tickers pass liquidity filter."
        close = close[keep]
        sectors_map = {t: sectors_map.get(t, "Unknown") for t in close.columns}

    # Monthly lock check — hold previous if not first trading day
    today = datetime.today().date()
    is_monthly = is_rebalance_today(today, close.index)
    decision = "Not the monthly rebalance day — holding previous portfolio."
    if not is_monthly:
        if prev_portfolio is not None and not prev_portfolio.empty:
            disp, raw = _format_display(prev_portfolio["Weight"].astype(float))
            return disp, raw, decision
        else:
            decision = "No saved portfolio; proposing initial allocation (monthly lock applies from next month)."

    # Build candidate weights (enhanced)
    params = dict(STRATEGY_PRESETS["ISA Dynamic (0.75)"])
    params["stability_days"] = stickiness_days
    params["sector_cap"]     = sector_cap

    new_w = _build_isa_weights(close, params, sectors_map)

    # Trigger vs previous portfolio (health of current)
    if prev_portfolio is not None and not prev_portfolio.empty and "Weight" in prev_portfolio.columns:
        monthly = close.resample("ME").last()
        mom_scores = blended_momentum_z(monthly)
        if not mom_scores.empty and len(new_w) > 0:
            top_m = mom_scores.nlargest(params["mom_topn"])
            top_score = float(top_m.iloc[0]) if len(top_m) > 0 else 1e-9
            prev_w = prev_portfolio["Weight"].astype(float)
            held_scores = mom_scores.reindex(prev_w.index).fillna(0.0)
            health = float((held_scores * prev_w).sum() / max(top_score, 1e-9))
            if health >= params["trigger"]:
                decision = f"Health {health:.2f} ≥ trigger {params['trigger']:.2f} — holding existing portfolio."
                disp, raw = _format_display(prev_w)
                return disp, raw, decision
            else:
                decision = f"Health {health:.2f} < trigger {params['trigger']:.2f} — rebalancing to new targets."

    disp, raw = _format_display(new_w)
    return disp, raw, decision


def run_backtest_isa_dynamic(
    roundtrip_bps: float = 0.0,
    min_dollar_volume: float = 0.0,
    show_net: bool = True,
    start_date: str = "2017-07-01",
    end_date: Optional[str] = None,
    universe_choice: Optional[str] = "Hybrid Top150",
    top_n: int = 8,
    name_cap: float = 0.25,
    sector_cap: float = 0.30,
    stickiness_days: int = 7,
    mr_topn: int = 3,
    mom_weight: float = 0.85,
    mr_weight: float = 0.15,
    use_enhanced_features: bool = True,
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """
    Enhanced ISA-Dynamic hybrid backtest with new features
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    # Universe & data (helper)
    close, vol, sectors_map, label = _prepare_universe_for_backtest(universe_choice, start_date, end_date)
    if close.empty or "QQQ" not in close.columns:
        return None, None, None, None

    # Liquidity floor (optional)
    if min_dollar_volume > 0:
        keep = filter_by_liquidity(
            close.drop(columns=["QQQ"], errors="ignore"),
            vol.drop(columns=["QQQ"], errors="ignore"),
            min_dollar_volume
        )
        keep_cols = [c for c in keep if c in close.columns]
        if not keep_cols:
            return None, None, None, None
        close = close[keep_cols + ["QQQ"]]
        sectors_map = {t: sectors_map.get(t, "Unknown") for t in keep_cols}

    daily = close.drop(columns=["QQQ"])
    qqq  = close["QQQ"]

    # Sleeves (enhanced)
    mom_rets, mom_tno = run_momentum_composite_param(
        daily,
        sectors_map,
        top_n=top_n,
        name_cap=name_cap,
        sector_cap=sector_cap,
        stickiness_days=stickiness_days,
        use_enhanced_features=use_enhanced_features
    )
    mr_rets,  mr_tno  = run_backtest_mean_reversion(daily, lookbook_period_mr=21, top_n_mr=mr_topn, long_ma_days=200)

    # Combine + costs
    hybrid_gross, hybrid_tno = combine_hybrid(mom_rets, mr_rets, mom_tno, mr_tno, mom_w=mom_weight, mr_w=mr_weight)
    
    # Apply drawdown-based exposure adjustment
    if use_enhanced_features:
        dd_exposure = get_drawdown_adjusted_exposure(hybrid_gross)
        hybrid_gross = hybrid_gross * dd_exposure
    
    hybrid_net = apply_costs(hybrid_gross, hybrid_tno, roundtrip_bps) if show_net else hybrid_gross

    # Cum curves
    strat_cum_gross = (1 + hybrid_gross.fillna(0)).cumprod()
    strat_cum_net   = (1 + hybrid_net.fillna(0)).cumprod() if show_net else None
    qqq_cum = (1 + qqq.resample("ME").last().pct_change()).cumprod().reindex(strat_cum_gross.index, method="ffill")

    return strat_cum_gross, strat_cum_net, qqq_cum, hybrid_tno
    
# =========================
# Diff engine (for Plan tab) - Unchanged
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
# Explainability (what changed & why) - Unchanged
# =========================
def _signal_snapshot_for_explain(daily_prices: pd.DataFrame, params: Dict) -> pd.DataFrame:
    if daily_prices.empty:
        return pd.DataFrame()
    monthly = daily_prices.resample("ME").last()
    if monthly.shape[0] < 13:
        return pd.DataFrame()

    r3  = monthly.pct_change(3).iloc[-1]
    r6  = monthly.pct_change(6).iloc[-1]
    r12 = monthly.pct_change(12).iloc[-1]
    def z(s: pd.Series, cols) -> pd.Series:
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.std(ddof=0) == 0 or s.empty:
            return pd.Series(0.0, index=cols)
        zs = (s - s.mean()) / (s.std(ddof=0) + 1e-9)
        return zs.reindex(cols).fillna(0.0)

    cols = monthly.columns
    mom_score = 0.2 * z(r3, cols) + 0.4 * z(r6, cols) + 0.4 * z(r12, cols)
    mom_rank  = mom_score.rank(ascending=False, method="min")

    st_ret = daily_prices.pct_change(params["mr_lb"]).iloc[-1]
    long_ma = daily_prices.rolling(params["mr_ma"]).mean().iloc[-1]
    above_ma = (daily_prices.iloc[-1] > long_ma).astype(int)

    snap = pd.DataFrame({
        "mom_score": mom_score,
        "mom_rank": mom_rank,
        f"st_ret_{params['mr_lb']}d": st_ret,
        f"above_{params['mr_ma']}dma": above_ma
    })
    return snap.sort_index()

def explain_portfolio_changes(prev_df: Optional[pd.DataFrame],
                              curr_df: Optional[pd.DataFrame],
                              daily_prices: pd.DataFrame,
                              params: Dict) -> pd.DataFrame:
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

    prices = daily_prices[all_tickers].dropna(axis=1, how="all")
    if prices.empty:
        return pd.DataFrame()

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

        mom_rank = snap.at[t, "mom_rank"] if t in snap.index else np.nan
        mom_score = snap.at[t, "mom_score"] if t in snap.index else np.nan
        st_key = f"st_ret_{params['mr_lb']}d"
        stv = snap.at[t, st_key] if t in snap.index else np.nan
        above_key = f"above_{params['mr_ma']}dma"
        ab = snap.at[t, above_key] if t in snap.index else np.nan

        rows.append({
            "Ticker": t,
            "Action": action,
            "Old Wt": old_w,
            "New Wt": new_w,
            "Δ Wt (bps)": int(round((new_w - old_w) * 10000)),
            "Mom Rank": int(mom_rank) if pd.notna(mom_rank) else np.nan,
            "Mom Score": mom_score,
            f"ST Return ({params['mr_lb']}d)": stv,
            f"Above {params['mr_ma']}DMA": bool(ab) if pd.notna(ab) else None
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    action_order = pd.Categorical(out["Action"], categories=["Buy","Rebalance","Sell"], ordered=True)
    out = out.assign(ActionOrder=action_order).sort_values(["ActionOrder","Δ Wt (bps)"], ascending=[True, False]).drop(columns=["ActionOrder"])
    out["Old Wt"] = out["Old Wt"].map(lambda x: f"{x:.2%}")
    out["New Wt"] = out["New Wt"].map(lambda x: f"{x:.2%}")
    return out.reset_index(drop=True)

# =========================
# Regime & Live paper tracking (Enhanced)
# =========================
def get_benchmark_series(ticker: str, start: str, end: str) -> pd.Series:
    px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px = _safe_series(px)
    return pd.Series(px).dropna()

def compute_regime_metrics(universe_prices_daily: pd.DataFrame) -> Dict[str, float]:
    """Enhanced regime metrics calculation"""
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
    """
    Enhanced market regime detection with additional context
    """
    try:
        univ = st.session_state.get("universe", "Hybrid Top150")
        base_tickers, _, _ = get_universe(univ)
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - relativedelta(months=12)).strftime("%Y-%m-%d")
        px = fetch_market_data(base_tickers, start, end)
        metrics = compute_regime_metrics(px)
        
        # Enhanced labeling with more nuanced categories
        breadth = metrics.get("breadth_pos_6m", np.nan)
        qqq_abv = metrics.get("qqq_above_200dma", np.nan)
        vol10   = metrics.get("qqq_vol_10d", np.nan)
        
        # More sophisticated regime classification
        if qqq_abv >= 1.0 and breadth > 0.65 and vol10 < 0.025:
            label = "Strong Risk-On"
        elif qqq_abv >= 1.0 and breadth > 0.50:
            label = "Risk-On"
        elif qqq_abv >= 1.0 and breadth > 0.35:
            label = "Cautious Risk-On"
        elif qqq_abv < 1.0 and breadth > 0.45:
            label = "Mixed"
        elif qqq_abv < 1.0 and breadth > 0.35:
            label = "Risk-Off"
        elif vol10 > 0.045:
            label = "High Volatility Risk-Off"
        else:
            label = "Extreme Risk-Off"
            
        return label, metrics
    except Exception:
        return "Neutral", {}

# =========================
# NEW: Strategy Health Monitoring
# =========================
def get_strategy_health_metrics(current_returns: pd.Series, 
                               benchmark_returns: pd.Series = None) -> Dict[str, float]:
    """Calculate comprehensive strategy health metrics"""
    if current_returns.empty:
        return {}
    
    health_metrics = {}
    
    # Rolling performance metrics
    if len(current_returns) >= 3:
        recent_3m = current_returns.iloc[-3:] if len(current_returns) >= 3 else current_returns
        health_metrics['recent_3m_return'] = recent_3m.mean()
        health_metrics['recent_3m_sharpe'] = (recent_3m.mean() * 12) / (recent_3m.std() * np.sqrt(12) + 1e-9)
    
    if len(current_returns) >= 6:
        recent_6m = current_returns.iloc[-6:]
        health_metrics['recent_6m_return'] = recent_6m.mean()
        health_metrics['hit_rate_6m'] = (recent_6m > 0).mean()
    
    # Drawdown analysis
    equity_curve = (1 + current_returns.fillna(0)).cumprod()
    current_dd = (equity_curve / equity_curve.cummax() - 1).iloc[-1]
    health_metrics['current_drawdown'] = current_dd
    
    # Volatility regime
    if len(current_returns) >= 12:
        recent_vol = current_returns.iloc[-12:].std() * np.sqrt(12)
        long_vol = current_returns.std() * np.sqrt(12)
        health_metrics['vol_regime_ratio'] = recent_vol / (long_vol + 1e-9)
    
    # Correlation to benchmark
    if benchmark_returns is not None:
        correlation = calculate_portfolio_correlation_to_market(current_returns, benchmark_returns)
        health_metrics['benchmark_correlation'] = correlation
    
    return health_metrics

def diagnose_strategy_issues(current_returns: pd.Series, 
                           turnover_series: pd.Series = None) -> List[str]:
    """Diagnose potential strategy issues"""
    issues = []
    
    if current_returns.empty:
        return ["No performance data available"]
    
    # Check recent performance
    if len(current_returns) >= 6:
        recent_6m = current_returns.iloc[-6:]
        if recent_6m.mean() < -0.02:  # Less than -2% monthly average
            issues.append("Poor recent performance (6M average < -2%)")
        
        hit_rate = (recent_6m > 0).mean()
        if hit_rate < 0.35:
            issues.append(f"Low hit rate ({hit_rate:.1%} positive months)")
    
    # Check drawdown
    equity_curve = (1 + current_returns.fillna(0)).cumprod()
    current_dd = (equity_curve / equity_curve.cummax() - 1).iloc[-1]
    if current_dd < -0.20:
        issues.append(f"Large drawdown ({current_dd:.1%})")
    
    # Check turnover efficiency
    if turnover_series is not None and len(turnover_series) > 0:
        avg_turnover = turnover_series.mean()
        if avg_turnover > 1.0:  # More than 100% monthly turnover
            issues.append("Excessive turnover (>100% monthly)")
    
    # Check volatility
    if len(current_returns) >= 12:
        recent_vol = current_returns.iloc[-12:].std() * np.sqrt(12)
        if recent_vol > 0.40:  # More than 40% annual volatility
            issues.append(f"High volatility ({recent_vol:.1%} annual)")
    
    if not issues:
        issues.append("No significant issues detected")
    
    return issues

# =========================
# Live performance tracking (Enhanced)
# =========================
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
        universe_choice = st.session_state.get("universe", "Hybrid Top150")
        base_tickers, _, _ = get_universe(universe_choice)
        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - relativedelta(days=40)).strftime("%Y-%m-%d")
        px = fetch_market_data(base_tickers + ["QQQ"], start_date, end_date)
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

# =========================
# NEW: Monte Carlo Forward Projections
# =========================
def run_monte_carlo_projections(historical_returns: pd.Series, 
                               n_scenarios: int = 5000,
                               horizon_months: int = 12,
                               confidence_levels: List[int] = [10, 50, 90],
                               block_size: int = 3) -> Dict:
    """Enhanced Monte Carlo projections with regime awareness"""
    
    if len(historical_returns) < 12:
        return {"error": "Insufficient historical data for Monte Carlo"}
    
    # Clean returns
    returns_clean = historical_returns.dropna()
    if len(returns_clean) < 6:
        return {"error": "Insufficient clean returns data"}
    
    # Block bootstrap to preserve short-term correlation
    rng = np.random.default_rng(42)
    scenarios = []
    
    for _ in range(n_scenarios):
        scenario_path = []
        months_needed = horizon_months
        
        while months_needed > 0:
            # Random starting point for block
            if block_size >= len(returns_clean):
                block_start = 0
                block = returns_clean.values
            else:
                block_start = rng.integers(0, len(returns_clean) - block_size + 1)
                block = returns_clean.iloc[block_start:block_start + block_size].values
            
            scenario_path.extend(block[:months_needed])
            months_needed -= len(block)
        
        # Calculate scenario outcome
        scenario_returns = np.array(scenario_path[:horizon_months])
        scenario_total = (1 + scenario_returns).prod() - 1
        scenarios.append(scenario_total)
    
    scenarios = np.array(scenarios)
    
    # Calculate percentiles
    percentiles = {}
    for conf in confidence_levels:
        percentiles[f'p{conf}'] = np.percentile(scenarios, conf)
    
    # Additional statistics
    results = {
        'scenarios': scenarios,
        'percentiles': percentiles,
        'mean_return': scenarios.mean(),
        'std_return': scenarios.std(),
        'prob_positive': (scenarios > 0).mean(),
        'prob_beat_10pct': (scenarios > 0.10).mean(),
        'downside_risk': scenarios[scenarios < 0].mean() if (scenarios < 0).any() else 0,
        'horizon_months': horizon_months
    }
    
    return results
