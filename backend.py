# backend.py — Hybrid Top150 / Composite Rank / Sector Caps / Stickiness / ISA lock
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
# Helpers
# =========================
def _safe_series(obj):
    return obj.squeeze() if isinstance(obj, pd.DataFrame) else obj

def to_yahoo_symbol(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-")

# =========================
# Universe builders & sectors
# =========================
@st.cache_data(ttl=86400)
def fetch_sp500_constituents() -> List[str]:
    """Get current S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        # Remove weird cases (e.g., BRK.B => BRK-B already handled via replace)
        return sorted(list(set(tickers)))
    except Exception as e:
        st.warning(f"Failed to fetch S&P 500 list: {e}")
        return []

@st.cache_data(ttl=86400)
def get_sector_map(tickers: List[str]) -> Dict[str, str]:
    """
    Best-effort sector map via yfinance. Returns 'Unknown' when missing.
    Cached to avoid repeated API calls.
    """
    out = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info or {}
            sector = info.get("sector") or "Unknown"
        except Exception:
            sector = "Unknown"
        out[t] = sector
    return out

def get_universe(choice: str) -> Tuple[List[str], Dict[str, str], str]:
    """
    Returns (tickers, sectors_map, label) for:
      - 'NASDAQ100+'
      - 'S&P500 (All)'
      - 'Hybrid Top150'  (start from S&P500; later reduced to top 150 by liquidity)
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
# Data fetching (cache)
# =========================
@st.cache_data(ttl=43200)
def fetch_market_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
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
    try:
        fetch_start = (pd.to_datetime(start_date) - pd.DateOffset(months=14)).strftime("%Y-%m-%d")
        df = yf.download(tickers, start=fetch_start, end=end_date, auto_adjust=True, progress=False)[["Close","Volume"]]
        if isinstance(df, pd.Series): df = df.to_frame()
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
        return close, vol
    except Exception as e:
        st.error(f"Failed to download price/volume: {e}")
        return pd.DataFrame(), pd.DataFrame()

# =========================
# Persistence (Gist + Local)
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
# Composite Signals (mom + trend + lowvol) + Stickiness
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
# Sleeves (composite momentum + MR) with turnover
# =========================
def run_momentum_composite_param(daily: pd.DataFrame, sectors_map: dict,
                                 top_n=8, name_cap=0.25, sector_cap=0.30, stickiness_days=7):
    monthly = daily.resample("ME").last()
    fwd = monthly.pct_change().shift(-1)
    rets = pd.Series(index=monthly.index, dtype=float)
    tno  = pd.Series(index=monthly.index, dtype=float)
    prev_w = pd.Series(dtype=float)
    for m in monthly.index:
        hist = daily.loc[:m]
        comp = composite_score(hist)
        if comp.empty: rets.loc[m]=0.0; tno.loc[m]=0.0; continue
        momz = blended_momentum_z(hist.resample("ME").last())
        comp = comp[momz.index]
        sel  = comp[momz > 0].dropna()
        if sel.empty: rets.loc[m]=0.0; tno.loc[m]=0.0; continue
        picks = sel.nlargest(top_n)
        stable = momentum_stable_names(hist, top_n=top_n, days=stickiness_days)
        if len(stable)>0:
            picks = picks.reindex([t for t in picks.index if t in stable]).dropna()
            if picks.empty: picks = sel.nlargest(top_n)
        raw = (picks / picks.sum()).astype(float)
        raw = cap_weights(raw, cap=name_cap)
        w   = apply_sector_caps(raw, sectors_map, cap=sector_cap)
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

def combine_hybrid(mom_rets, mr_rets, mom_tno, mr_tno, mw=0.85, rw=0.15):
    idx = mom_rets.index.union(mr_rets.index)
    mom_r = mom_rets.reindex(idx, fill_value=0.0)
    mr_r  = mr_rets.reindex(idx,  fill_value=0.0)
    mom_n = mom_tno.reindex(idx, fill_value=0.0)
    mr_n  = mr_tno.reindex(idx,  fill_value=0.0)
    gross = mom_r*mw + mr_r*rw
    tno   = mom_n*mw + mr_n*rw
    return gross, tno

def apply_costs(gross, turnover, roundtrip_bps):
    return gross - turnover*(roundtrip_bps/10000.0)

# =========================
# Liquidity utils (Hybrid150)
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
# Live portfolio builders (ISA MONTHLY LOCK + stickiness + sector caps)
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

    # Weights for momentum sleeve (with name cap + sector caps)
    if not top_m.empty and top_m.sum() > 0:
        mom_raw = top_m / top_m.sum()
        mom_w = cap_weights(mom_raw, cap=preset["mom_cap"]) * preset["mom_w"]
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
    ISA Dynamic live weights with MONTHLY LOCK + composite + stability + sector caps.
    Universe is taken from st.session_state.universe if present (default Hybrid Top150).
    """
    universe_choice = st.session_state.get("universe", "Hybrid Top150")
    stickiness_days = st.session_state.get("stickiness_days", preset.get("stability_days", 7))
    sector_cap      = st.session_state.get("sector_cap", preset.get("sector_cap", 0.30))

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

    # Build candidate weights
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
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """
    ISA-Dynamic hybrid backtest (composite momentum + mean reversion) with
    costs, liquidity, stickiness, sector caps, and QQQ benchmark.
    Returns: (strategy_cum_gross, strategy_cum_net_or_None, qqq_cum, turnover_series)
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

    # Sleeves
    mom_rets, mom_tno = run_momentum_composite_param(
        daily,
        sectors_map,
        top_n=top_n,
        name_cap=name_cap,
        sector_cap=sector_cap,
        stickiness_days=stickiness_days
    )
    mr_rets,  mr_tno  = run_backtest_mean_reversion(daily, lookback_period_mr=21, top_n_mr=mr_topn, long_ma_days=200)

    # Combine + costs
    hybrid_gross, hybrid_tno = combine_hybrid(mom_rets, mr_rets, mom_tno, mr_tno, mom_w=mom_weight, mr_w=mr_weight)
    hybrid_net = apply_costs(hybrid_gross, hybrid_tno, roundtrip_bps) if show_net else hybrid_gross

    # Cum curves
    strat_cum_gross = (1 + hybrid_gross.fillna(0)).cumprod()
    strat_cum_net   = (1 + hybrid_net.fillna(0)).cumprod() if show_net else None
    qqq_cum = (1 + qqq.resample("ME").last().pct_change()).cumprod().reindex(strat_cum_gross.index, method="ffill")

    return strat_cum_gross, strat_cum_net, qqq_cum, hybrid_tno
    
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
# Explainability (what changed & why)
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
    """
    Convenience wrapper used by the app's Regime tab.
    Uses NASDAQ100+ prices as a proxy universe if available; otherwise returns neutral.
    """
    try:
        univ = st.session_state.get("universe", "Hybrid Top150")
        base_tickers, _, _ = get_universe(univ)
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - relativedelta(months=12)).strftime("%Y-%m-%d")
        px = fetch_market_data(base_tickers, start, end)
        metrics = compute_regime_metrics(px)
        # Simple labeling
        breadth = metrics.get("breadth_pos_6m", np.nan)
        qqq_abv = metrics.get("qqq_above_200dma", np.nan)
        vol10   = metrics.get("qqq_vol_10d", np.nan)
        label = "Risk-On"
        if (qqq_abv < 1.0 and breadth < 0.45) or (vol10 > 0.035 and qqq_abv < 1.0):
            label = "Risk-Off"
        if (qqq_abv < 1.0 and breadth < 0.35) or (vol10 > 0.045 and qqq_abv < 1.0):
            label = "Extreme Risk-Off"
        return label, metrics
    except Exception:
        return "Neutral", {}

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
