# backend.py — Enhanced Hybrid Top150 / Composite Rank / Sector Caps / Stickiness / ISA lock
import os, io, warnings
from typing import Optional, Tuple, Dict, List, Any
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
AVG_TRADE_SIZE_DEFAULT = 0.02  # 2% avg single-leg trade size

# Defaults for regime-based exposure adjustments
VIX_TS_THRESHOLD_DEFAULT = 1.0   # VIX3M / VIX ratio; <1 implies stress
HY_OAS_THRESHOLD_DEFAULT = 6.0   # High-yield OAS level (%) signalling stress

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
# NEW: Enhanced Data Validation & Cleaning
# =========================
def clean_extreme_moves(prices_df: pd.DataFrame, max_daily_move: float = 0.30, 
                       min_price: float = 1.0) -> pd.DataFrame:
    """Clean extreme price moves that are likely data errors"""
    if prices_df.empty:
        return prices_df
    
    cleaned_df = prices_df.copy()
    total_corrections = 0
    
    for column in cleaned_df.columns:
        series = cleaned_df[column].copy()
        
        # Remove prices below minimum (likely stock splits not handled)
        low_price_mask = series < min_price
        if low_price_mask.any():
            # Forward fill from last good price
            series = series.where(~low_price_mask).ffill()
        
        # Calculate daily returns
        daily_returns = series.pct_change().abs()
        
        # Find extreme moves
        extreme_moves = daily_returns > max_daily_move
        extreme_dates = daily_returns[extreme_moves].index
        
        if len(extreme_dates) > 0:
            total_corrections += len(extreme_dates)
            
            for date in extreme_dates:
                # Replace extreme move with interpolated value
                date_idx = series.index.get_loc(date)
                
                if date_idx > 0 and date_idx < len(series) - 1:
                    # Interpolate between previous and next valid price
                    prev_price = series.iloc[date_idx - 1]
                    next_price = series.iloc[date_idx + 1]
                    
                    if pd.notna(prev_price) and pd.notna(next_price):
                        series.iloc[date_idx] = (prev_price + next_price) / 2
                    elif pd.notna(prev_price):
                        series.iloc[date_idx] = prev_price
                    elif pd.notna(next_price):
                        series.iloc[date_idx] = next_price
                elif date_idx > 0:
                    # Use previous price
                    series.iloc[date_idx] = series.iloc[date_idx - 1]
        
        cleaned_df[column] = series
    
    if total_corrections > 0:
        st.info(f"🧹 Data cleaning: Fixed {total_corrections} extreme price moves across all stocks")
    
    return cleaned_df

def fill_missing_data(prices_df: pd.DataFrame, max_gap_days: int = 5) -> pd.DataFrame:
    """Fill missing data gaps with intelligent interpolation"""
    if prices_df.empty:
        return prices_df
    
    filled_df = prices_df.copy()
    total_filled = 0
    
    for column in filled_df.columns:
        series = filled_df[column].copy()
        missing_mask = series.isna()
        
        if missing_mask.any():
            # Count consecutive missing values
            missing_groups = (missing_mask != missing_mask.shift()).cumsum()
            missing_counts = missing_mask.groupby(missing_groups).sum()
            
            # Only fill gaps smaller than max_gap_days
            small_gaps = missing_counts[missing_counts <= max_gap_days]
            
            for group_id in small_gaps.index:
                gap_mask = missing_groups == group_id
                gap_indices = series.index[gap_mask]
                
                if len(gap_indices) > 0:
                    # Forward fill first, then backward fill if needed
                    series = series.ffill()
                    series = series.bfill()
                    total_filled += len(gap_indices)
        
        filled_df[column] = series
    
    if total_filled > 0:
        st.info(f"🔧 Data filling: Filled {total_filled} missing data points with interpolation")
    
    return filled_df

def validate_and_clean_market_data(prices_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Comprehensive data validation and cleaning pipeline"""
    if prices_df.empty:
        return prices_df, ["No data to clean"]
    
    original_shape = prices_df.shape
    alerts = []
    
    # Step 1: Clean extreme moves
    cleaned_df = clean_extreme_moves(prices_df, max_daily_move=0.25, min_price=0.50)
    
    # Step 2: Fill missing data gaps
    filled_df = fill_missing_data(cleaned_df, max_gap_days=3)
    
    # Step 3: Remove columns with too much missing data
    missing_pct = filled_df.isnull().sum() / len(filled_df)
    good_columns = missing_pct[missing_pct < 0.20].index  # Keep columns with <20% missing
    
    if len(good_columns) < len(filled_df.columns):
        removed = len(filled_df.columns) - len(good_columns)
        alerts.append(f"Removed {removed} stocks with >20% missing data")
        filled_df = filled_df[good_columns]
    
    # Step 4: Final validation
    remaining_missing = filled_df.isnull().sum().sum()
    total_points = filled_df.shape[0] * filled_df.shape[1]
    final_missing_pct = (remaining_missing / total_points) * 100 if total_points > 0 else 0
    
    if final_missing_pct > 0:
        alerts.append(f"Final missing data: {final_missing_pct:.1f}%")
    
    # Step 5: Check for remaining extreme moves
    for col in filled_df.columns:
        daily_rets = filled_df[col].pct_change().abs()
        extreme_count = (daily_rets > 0.25).sum()
        if extreme_count > 0:
            alerts.append(f"{col}: {extreme_count} remaining extreme moves")
    
    final_shape = filled_df.shape
    alerts.append(f"Data shape: {original_shape} → {final_shape}")
    
    return filled_df, alerts

# =========================
# NEW: Enhanced Position Sizing (Fixes the 28.99% bug)
# =========================

def generate_td1_targets_asof(
    daily_close: pd.DataFrame,
    sectors_map: dict,
    preset: dict,
    asof: Optional[pd.Timestamp] = None,
    use_enhanced_features: bool = True,
    enable_hedge: bool = False,
    hedge_size: float = 0.0,
) -> pd.Series:
    """
    Build targets exactly as on Trading Day 1:
    - Freeze signals at the most recent month-end prior to `asof`
    - Run the same screens, ranking, caps, and (if used) exposure scaling
    """
    asof = pd.Timestamp(asof or pd.Timestamp.today()).normalize()
    last_eom = (asof - pd.offsets.MonthBegin(1)) + pd.offsets.MonthEnd(0)

    # Use *only* data available at last_eom
    hist = daily_close.loc[:last_eom].copy()
    if hist.empty:
        return pd.Series(dtype=float)

    # Your existing (fixed) builder runs full selection + caps
    weights = _build_isa_weights_fixed(hist, preset, sectors_map)  # returns weights summing to equity exposure

    regime_metrics: Dict[str, float] = {}
    # If TD1 applies regime exposure (not the drawdown scaler on *returns*, but the weight scaler), apply it here too
    if use_enhanced_features and len(hist) > 0 and len(weights) > 0:
        try:
            regime_metrics = compute_regime_metrics(hist)
            regime_exposure = get_regime_adjusted_exposure(regime_metrics)
            weights = weights * float(regime_exposure)   # leaves implied cash = 1 - sum(weights)
        except Exception:
            regime_metrics = {}

    # Optional QQQ hedge
    if enable_hedge and hedge_size > 0 and len(hist) > 0 and len(weights) > 0:
        try:
            returns = hist.pct_change().dropna()
            port_rets = (returns[weights.index] * weights).sum(axis=1)
            hedge_w = build_hedge_weight(port_rets, regime_metrics, hedge_size)
            if hedge_w > 0:
                weights.loc["QQQ"] = -hedge_w
        except Exception:
            pass

    return weights.fillna(0.0)

# ---- Software split caps (Option 2) ----
ENABLE_SOFTWARE_SUBCAPS: bool = True   # turn on/off the feature
PARENT_SOFTWARE_CAP: float = 0.30      # keep total Software <= 30%
SOFTWARE_SUBCAP: float = 0.18          # each Software sub-bucket <= 18%

def enforce_caps_iteratively(
    weights: pd.Series,
    sector_labels: dict[str, str],
    name_cap: float = 0.25,
    sector_cap: float = 0.30,
    group_caps: dict[str, float] | None = None,
    max_iter: int = 200,
    debug: bool = False
) -> pd.Series:
    """
    Enforce name caps, per-sector caps, and optional per-group (hierarchical) caps.
    - sector_labels: mapping ticker -> label (can be "Software:Security", etc.)
    - sector_cap: default cap for any sector not explicitly listed in group_caps
    - group_caps: optional dict like {"Software": 0.30, "Software:Security": 0.18, ...}
                  Parent groups are labels with no ":" (e.g., "Software").
    We *do not* force re-distribution; trimmed weight becomes cash (i.e., sum <= 1),
    which matches your existing preview behavior.
    """

    if weights.empty:
        return weights

    w = weights.astype(float).copy()
    w[w < 0] = 0.0

    # Normalize only if clearly >1 because some callers already pre-normalize
    if w.sum() > 1.0 + 1e-9:
        w = w / w.sum()

    # Vectorized helpers
    ser_sector = pd.Series({k: sector_labels.get(k, "Unknown") for k in w.index})
    ser_top = ser_sector.map(lambda s: s.split(":")[0])

    def _apply_name_caps(w: pd.Series) -> tuple[pd.Series, bool]:
        over = w[w > name_cap]
        if over.empty:
            return w, False
        w.loc[over.index] = name_cap
        return w, True

    def _apply_sector_caps(w: pd.Series) -> tuple[pd.Series, bool]:
        changed = False
        sums = w.groupby(ser_sector).sum()
        for sec, s in sums.items():
            cap = (group_caps.get(sec) if group_caps and sec in group_caps else sector_cap)
            if s > cap + 1e-12:
                f = cap / s
                idx = ser_sector[ser_sector == sec].index
                w.loc[idx] = w.loc[idx] * f
                changed = True
        return w, changed

    def _apply_parent_caps(w: pd.Series) -> tuple[pd.Series, bool]:
        if not group_caps:
            return w, False
        changed = False
        parent_labels = [k for k in group_caps.keys() if ":" not in k]
        if not parent_labels:
            return w, False
        sums = w.groupby(ser_top).sum()
        for parent in parent_labels:
            if parent in sums.index and sums[parent] > group_caps[parent] + 1e-12:
                f = group_caps[parent] / sums[parent]
                idx = ser_top[ser_top == parent].index
                w.loc[idx] = w.loc[idx] * f
                changed = True
        return w, changed

    for _ in range(max_iter):
        changed_any = False

        w, c1 = _apply_name_caps(w)
        w, c2 = _apply_sector_caps(w)
        w, c3 = _apply_parent_caps(w)

        changed_any = c1 or c2 or c3
        if not changed_any:
            break

    # Safety: clip tiny negatives from numerical noise
    w[w < 0] = 0.0
    return w

# --- Enhanced sector bucketing -----------------------------------------------
from typing import Dict, List, Optional

def get_enhanced_sector_map(tickers: list[str], base_map: dict[str, str] | None = None) -> dict[str, str]:
    """
    Return an 'enhanced' sector map. If a ticker is 'Software' in the base map,
    split it into sub-buckets to allow finer caps: Security, Data/AI, AdTech,
    Collaboration, Commerce, Other. Everything else passes through.

    Priority: if base_map is provided, use it. Otherwise fall back to 'Unknown'.
    """

    base_map = base_map or {}

    # --- software sub-buckets (ticker heuristics) ---
    sec_security = {
        "CRWD","ZS","FTNT","PANW","OKTA","S","TENB","NET"
    }
    sec_data_ai = {
        "PLTR","SNOW","MDB","DDOG","NRTX","AI"  # keep PLTR here
    }
    sec_adtech = {"APP","TTD"}
    sec_collab = {"ZM","TEAM"}
    sec_commerce = {"SHOP","MELI","ETSY","SQ","ADYEY","AFRM"}

    # Convenience helper
    def _software_bucket(t: str) -> str:
        u = t.upper()
        if u in sec_security:  return "Software:Security"
        if u in sec_data_ai:   return "Software:Data/AI"
        if u in sec_adtech:    return "Software:AdTech"
        if u in sec_collab:    return "Software:Collab"
        if u in sec_commerce:  return "Software:Commerce"
        return "Software:Other"

    out: dict[str, str] = {}
    for t in tickers:
        base = (base_map.get(t)
                or base_map.get(t.upper())
                or "Unknown")

        # keep a few custom buckets you already use if present in base
        if base in {"AI Hardware","Crypto/Fintech","Mega Tech","Semiconductors"}:
            out[t] = base
        elif base == "Software":
            out[t] = _software_bucket(t)
        else:
            out[t] = base
    return out

def build_group_caps(enhanced_map: dict[str, str]) -> dict[str, float]:
    """
    Build a dictionary of caps to apply in enforce_caps_iteratively.
    We keep the global Software cap AND add mini-caps to any present sub-buckets.
    """
    if not ENABLE_SOFTWARE_SUBCAPS:
        return {}

    caps: dict[str, float] = {}
    # Parent cap
    caps["Software"] = PARENT_SOFTWARE_CAP

    # Sub-buckets that actually appear
    present = {v for v in enhanced_map.values() if v.startswith("Software:")}
    for sb in present:
        caps[sb] = min(SOFTWARE_SUBCAP, PARENT_SOFTWARE_CAP)
    return caps

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
# NEW: Risk Parity Weighting
# =========================
def risk_parity_weights(prices: pd.DataFrame, tickers: List[str], lookback: int = 63) -> pd.Series:
    """Compute inverse-volatility risk parity weights.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily price history for the universe of tickers.
    tickers : List[str]
        List of tickers to weight.
    lookback : int, optional
        Number of trading days for volatility calculation, by default 63.

    Returns
    -------
    pd.Series
        Risk-parity weights summing to 1.0 for the provided tickers.
    """
    if not tickers:
        return pd.Series(dtype=float)

    # Ensure we have the necessary price data
    sub_prices = prices.reindex(columns=tickers)
    returns = sub_prices.pct_change().dropna()
    if returns.empty or len(returns) < lookback:
        # Fall back to equal weighting if not enough data
        return pd.Series(1 / len(tickers), index=tickers)

    vols = returns.rolling(lookback).std().iloc[-1]
    vols = vols.replace(0, np.nan)

    inv_vol = 1 / vols
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()
    if inv_vol.empty:
        return pd.Series(1 / len(tickers), index=tickers)

    weights = inv_vol / inv_vol.sum()
    return weights.reindex(tickers).fillna(0.0)

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
    vix_ts = regime_metrics.get('vix_term_structure', 1.0)
    hy_oas = regime_metrics.get('hy_oas', 4.0)
    
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

    # VIX term structure (3M/1M); backwardation (< threshold) is risk-off
    vix_thresh = st.session_state.get("vix_ts_threshold", VIX_TS_THRESHOLD_DEFAULT)
    if vix_ts < vix_thresh:
        exposure *= 0.8

    # High-yield credit spreads
    oas_thresh = st.session_state.get("hy_oas_threshold", HY_OAS_THRESHOLD_DEFAULT)
    if hy_oas > oas_thresh:
        exposure *= 0.8
    elif hy_oas < max(0.0, oas_thresh - 2):
        exposure *= 1.05
    
    # Adjust for trend
    if qqq_above_ma < 1.0:  # Below 200DMA
        exposure *= 0.9

    return np.clip(exposure, 0.3, 1.2)  # Keep between 30% and 120%

# =========================
# NEW: Regime-Based Parameter Mapping
# =========================
def map_metrics_to_settings(metrics: Dict[str, float]) -> Dict[str, float]:
    """Map regime metrics to strategy parameter settings.

    Parameters
    ----------
    metrics : dict
        Dictionary containing regime metrics such as ``qqq_vol_10d``.

    Returns
    -------
    dict
        Dictionary with keys ``top_n``, ``name_cap``, and ``sector_cap``.
    """

    vol = metrics.get("qqq_vol_10d", np.nan)

    # Baseline settings
    top_n = 8
    name_cap = 0.25
    sector_cap = 0.30

    if pd.notna(vol):
        if vol > 0.04:  # High volatility -> be more defensive
            top_n = 5
            name_cap = 0.20
            sector_cap = 0.25
        elif vol < 0.02:  # Low volatility -> allow broader exposure
            top_n = 10
            name_cap = 0.30
            sector_cap = 0.35

    return {"top_n": top_n, "name_cap": name_cap, "sector_cap": sector_cap}

# =========================
# NEW: Enhanced Drawdown Controls
# =========================
def get_drawdown_adjusted_exposure(current_returns: pd.Series,
                                   qqq_returns: pd.Series,
                                   threshold_fraction: float = 0.8) -> float:
    """Reduce exposure during sustained drawdowns

    Parameters
    ----------
    current_returns : pd.Series
        Strategy returns leading up to the evaluation point.
    qqq_returns : pd.Series
        Corresponding QQQ benchmark returns used to derive a dynamic
        drawdown threshold.
    threshold_fraction : float, optional
        Fraction of QQQ's drawdown used as the maximum tolerable drawdown
        before scaling, by default 0.8.
    """
    if current_returns.empty or qqq_returns.empty:
        return 1.0

    equity_curve = (1 + current_returns.fillna(0)).cumprod()
    strat_dd = (equity_curve / equity_curve.cummax() - 1).iloc[-1]

    qqq_curve = (1 + qqq_returns.fillna(0)).cumprod()
    qqq_dd = (qqq_curve / qqq_curve.cummax() - 1).iloc[-1]

    max_dd_threshold = threshold_fraction * abs(qqq_dd)

    if strat_dd < -max_dd_threshold:
        dd_severity = abs(strat_dd)
        exposure_reduction = min(dd_severity * 0.5, 0.7)  # Max 70% reduction
        return max(1.0 - exposure_reduction, 0.3)  # Never below 30%

    return 1.0

def apply_dynamic_drawdown_scaling(monthly_returns: pd.Series,
                                   qqq_monthly_returns: pd.Series,
                                   threshold_fraction: float = 0.8) -> pd.Series:
    """
    Walk-forward scaling: for each month, compute drawdowns of strategy history
    and the QQQ benchmark up to the previous month. The strategy's drawdown is
    compared against a threshold defined as ``threshold_fraction`` times the
    QQQ drawdown to determine the exposure for the following month.
    """
    r = pd.Series(monthly_returns).copy().fillna(0.0)
    qqq = pd.Series(qqq_monthly_returns).reindex(r.index).fillna(0.0)
    out = []
    hist = pd.Series(dtype=float)
    qqq_hist = pd.Series(dtype=float)
    for dt, val in r.items():
        q_val = qqq.loc[dt]
        exp = (get_drawdown_adjusted_exposure(hist, qqq_hist, threshold_fraction)
               if len(hist) and len(qqq_hist) else 1.0)
        out.append((dt, val * exp))
        hist = pd.concat([hist, pd.Series([val], index=[dt])])
        qqq_hist = pd.concat([qqq_hist, pd.Series([q_val], index=[dt])])
    return pd.Series(dict(out)).reindex(r.index)

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
# NEW: QQQ Hedge Builder
# =========================
def build_hedge_weight(portfolio_returns: pd.Series,
                       regime_metrics: Dict[str, float],
                       hedge_size: float,
                       corr_threshold: float = 0.8) -> float:
    """Determine QQQ hedge weight based on correlation and regime.

    The hedge activates only when the portfolio exhibits high correlation
    to QQQ *and* regime metrics flag bearish conditions. Returned weight is
    the fraction of capital to short in QQQ (0 <= w <= hedge_size).
    """
    if hedge_size <= 0 or portfolio_returns is None or portfolio_returns.empty:
        return 0.0

    corr = calculate_portfolio_correlation_to_market(portfolio_returns)
    if pd.isna(corr) or corr < 0:
        return 0.0

    bearish = (
        regime_metrics.get("qqq_above_200dma", 1.0) < 1.0 or
        regime_metrics.get("breadth_pos_6m", 1.0) < 0.40 or
        regime_metrics.get("qqq_50dma_slope_10d", 0.0) < 0.0
    )

    if not bearish or corr < corr_threshold:
        return 0.0

    scale = min(1.0, (corr - corr_threshold) / max(1e-6, 1 - corr_threshold))
    return float(np.clip(scale * hedge_size, 0.0, hedge_size))

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
    """Get current S&P 500 tickers with fallback to static list."""
    try:
        # Try with headers to avoid 403 blocking
        import requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url, attrs={'class': 'wikitable sortable'})
        df = tables[0]
        tickers = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        return sorted(list(set(tickers)))
    except Exception as e:
        st.warning(f"Failed to fetch S&P 500 list: {e}. Using fallback list.")
        # Static fallback list of major S&P 500 stocks (updated Aug 2025)
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "BRK-B", "LLY",
            "AVGO", "WMT", "JPM", "XOM", "UNH", "ORCL", "MA", "HD", "PG", "COST",
            "JNJ", "ABBV", "BAC", "CRM", "CVX", "KO", "AMD", "PEP", "TMO", "WFC",
            "CSCO", "ACN", "MRK", "DIS", "LIN", "ABT", "DHR", "NFLX", "VZ", "TXN",
            "QCOM", "CMCSA", "PM", "COP", "SPGI", "RTX", "UNP", "NKE", "T", "LOW",
            "INTU", "IBM", "GS", "HON", "CAT", "AXP", "UPS", "MS", "NEE", "MDT",
            "PFE", "INTC", "BLK", "GE", "DE", "TJX", "AMGN", "SYK", "ADP", "BKNG",
            "VRTX", "ADI", "C", "MU", "LRCX", "TMUS", "GILD", "SCHW", "AMAT", "EOG",
            "MMC", "NOW", "SHW", "ZTS", "PYPL", "CMG", "PANW", "FI", "ICE", "DUK",
            "PGR", "AON", "SO", "TGT", "ITW", "BSX", "WM", "CL", "MCD", "FCX",
            # Add more major names to get closer to 500
            "EMR", "APD", "KLAC", "SNPS", "CDNS", "ORLY", "MAR", "APH", "MSI", "SLB",
            "HUM", "ADSK", "ECL", "FDX", "NXPI", "ROP", "CME", "ROST", "AJG", "TFC",
            "PCAR", "KMB", "MNST", "FAST", "PAYX", "CTAS", "AMP", "OTIS", "DXCM", "EA",
            "VRSK", "BDX", "EXC", "KR", "GWW", "MLM", "VMC", "CTSH", "CARR", "URI",
            "WBD", "IDXX", "ANSS", "A", "SPG", "HES", "EW", "XEL", "PSA", "YUM",
            "CMI", "WELL", "CHTR", "ALL", "GD", "F", "GM", "STZ", "HCA", "AIG"
        ]

_SECTOR_CACHE_TTL = 86400
_SECTOR_CACHE: Dict[str, str] = {}

@st.cache_data(ttl=_SECTOR_CACHE_TTL)
def get_sector_map(tickers: List[str]) -> Dict[str, str]:
    """Return a mapping from ticker to sector name.

    Each ticker is fetched from :mod:`yfinance` at most once. Results are
    cached per ticker in the module-level ``_SECTOR_CACHE`` and the full mapping
    is cached by Streamlit's ``st.cache_data`` decorator for 24 hours
    (``_SECTOR_CACHE_TTL``). This minimises repeated network requests across
    executions. Tickers with missing sector information are stored as
    ``"Unknown"``.
    """
    new_tickers = [t for t in tickers if t not in _SECTOR_CACHE]
    for t in new_tickers:
        try:
            info = yf.Ticker(t).info or {}
            sector = info.get("sector") or "Unknown"
        except Exception:
            sector = "Unknown"
        _SECTOR_CACHE[t] = sector
    return {t: _SECTOR_CACHE.get(t, "Unknown") for t in tickers}

def get_nasdaq_100_plus_tickers() -> List[str]:
    """Get NASDAQ 100+ tickers with fallback list"""
    try:
        # Try with headers to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100', attrs={'class': 'wikitable'})
        nasdaq_100 = tables[4]['Ticker'].astype(str).str.upper().tolist()
        extras = ['TSLA', 'SHOP', 'SNOW', 'PLTR', 'ETSY', 'RIVN', 'COIN']
        return sorted(set(nasdaq_100 + extras))
    except Exception:
        # Static fallback for NASDAQ 100+ (updated Aug 2025)
        return [
            'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'COST',
            'NFLX', 'AMD', 'PEP', 'QCOM', 'CSCO', 'INTU', 'TXN', 'CMCSA', 'HON', 'AMGN',
            'BKNG', 'VRTX', 'ADI', 'GILD', 'MELI', 'LRCX', 'ADP', 'SBUX', 'PYPL', 'REGN',
            'AMAT', 'KLAC', 'MDLZ', 'SNPS', 'CRWD', 'CDNS', 'MAR', 'MRVL', 'ORLY', 'CSX',
            'DASH', 'ASML', 'ADSK', 'PCAR', 'ROP', 'NXPI', 'ABNB', 'FTNT', 'CHTR', 'AEP',
            'FAST', 'MNST', 'ODFL', 'ROST', 'BKR', 'EA', 'VRSK', 'EXC', 'XEL', 'TEAM',
            'CSGP', 'DDOG', 'GEHC', 'KDP', 'CTSH', 'FANG', 'ZS', 'ANSS', 'DXCM', 'BIIB',
            'WBD', 'MRNA', 'KHC', 'IDXX', 'CCEP', 'ON', 'MDB', 'ILMN', 'GFS', 'WBA',
            'SIRI', 'ARM', 'SMCI', 'TTD', 'CDW', 'ZM', 'GEN', 'PDD', 'ALGN', 'WDAY',
            # Your extras
            'SHOP', 'SNOW', 'PLTR', 'ETSY', 'RIVN', 'COIN'
        ]

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
    """Fetches prices/volumes, applies Hybrid150 filter if needed, with enhanced data cleaning."""
    base_tickers, base_sectors, label = get_universe(universe_choice)
    if not base_tickers:
        return pd.DataFrame(), pd.DataFrame(), {}, label

    close, vol = fetch_price_volume(base_tickers + ["QQQ"], start_date, end_date)
    if close.empty or "QQQ" not in close.columns:
        return pd.DataFrame(), pd.DataFrame(), {}, label

    # Enhanced data validation is now built into fetch_price_volume
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

    # Ensure timezone-naive datetimes
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
        
        # Enhanced data cleaning pipeline
        if not result.empty:
            cleaned_result, cleaning_alerts = validate_and_clean_market_data(result)
            
            # Show cleaning summary
            if cleaning_alerts:
                for alert in cleaning_alerts[:2]:  # Show top 2 cleaning actions
                    st.info(f"🧹 Data cleaning: {alert}")
            
            return cleaned_result
        
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
        
        # Enhanced data cleaning for prices
        if not close.empty:
            cleaned_close, close_alerts = validate_and_clean_market_data(close)
            
            # Clean volume data (less aggressive)
            vol_aligned = vol.reindex_like(cleaned_close).fillna(0)
            # Replace any negative or extreme volumes with more robust method
            for col in vol_aligned.columns:
                col_data = vol_aligned[col]
                if len(col_data.dropna()) > 0:
                    # Clip extreme volumes per column
                    q99 = col_data.quantile(0.99)
                    vol_aligned[col] = col_data.clip(lower=0, upper=q99)
            
            # Show cleaning summary
            if close_alerts:
                for alert in close_alerts[:1]:  # Show top cleaning action
                    st.info(f"🧹 Price/Volume cleaning: {alert}")
            
            return cleaned_close, vol_aligned
        
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
            avg_trade_size: float = AVG_TRADE_SIZE_DEFAULT) -> List[str]:
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
def run_momentum_composite_param(
    daily: pd.DataFrame,
    sectors_map: dict,
    top_n: int = 8,
    name_cap: float = 0.25,
    sector_cap: float = 0.30,
    stickiness_days: int = 7,
    use_enhanced_features: bool = True,
):
    """Enhanced momentum sleeve with stickiness, volatility-aware name caps, and
    simultaneous sector+name (and group) enforcement using the enhanced taxonomy."""
    debug_caps = bool(st.session_state.get("debug_caps", False))

    monthly = daily.resample("ME").last()
    fwd = monthly.pct_change().shift(-1)  # next-month returns
    rets = pd.Series(index=monthly.index, dtype=float)
    tno  = pd.Series(index=monthly.index, dtype=float)
    prev_w = pd.Series(dtype=float)

    for m in monthly.index:
        hist = daily.loc[:m]

        # Composite momentum score -> 1-D vector for the current month
        comp_all = composite_score(hist)
        if isinstance(comp_all, pd.DataFrame):
            comp = comp_all.iloc[-1].dropna()
        else:
            comp = pd.Series(comp_all).dropna()

        if comp.empty:
            rets.loc[m] = 0.0
            tno.loc[m]  = 0.0
            prev_w = pd.Series(dtype=float)
            continue

        # Restrict to positive blended momentum universe
        momz = blended_momentum_z(hist.resample("ME").last())
        comp = comp.reindex(momz.index).dropna()
        sel  = comp[momz > 0].dropna()
        if sel.empty:
            rets.loc[m] = 0.0
            tno.loc[m]  = 0.0
            prev_w = pd.Series(dtype=float)
            continue

        # Pick top-N by score
        picks = sel.nlargest(top_n)

        # Optional: signal decay shaping
        if use_enhanced_features:
            days_since_signal = 0  # TODO: wire up true signal age if you track it
            picks = apply_signal_decay(picks, days_since_signal)

        # Stickiness filter (prefer persistent names)
        stable = set(momentum_stable_names(hist, top_n=top_n, days=stickiness_days))
        if stable:
            filtered = picks.reindex([t for t in picks.index if t in stable]).dropna()
            if not filtered.empty:
                picks = filtered
            else:
                picks = sel.nlargest(top_n)  # fallback if stickiness empties set

        # Raw weights ~ proportional to scores
        if picks.empty or np.isclose(picks.sum(), 0.0):
            rets.loc[m] = 0.0
            tno.loc[m]  = 0.0
            prev_w = pd.Series(dtype=float)
            continue

        raw = (picks / picks.sum()).astype(float)

        # Soft cap by volatility-aware name caps (optional)
        if use_enhanced_features:
            vol_caps = get_volatility_adjusted_caps(raw, hist, base_cap=name_cap)
            raw = cap_weights(raw, cap=name_cap, vol_adjusted_caps=vol_caps)
        else:
            raw = cap_weights(raw, cap=name_cap)

        # Enhanced taxonomy + hierarchical group caps (e.g., Software parent + sub-buckets)
        enhanced_sectors = get_enhanced_sector_map(list(raw.index), base_map=sectors_map)
        # Ensure any missing tickers fall back to provided base map
        for t in raw.index:
            if t not in enhanced_sectors:
                enhanced_sectors[t] = sectors_map.get(t, "Other")
        group_caps = build_group_caps(enhanced_sectors)

        # Hard enforcement of name + sector (and group) caps
        w = enforce_caps_iteratively(
            raw.astype(float),
            enhanced_sectors,
            name_cap=name_cap,
            sector_cap=sector_cap,
            group_caps=group_caps,
            debug=debug_caps,
        )

        # Regime-based exposure scaling (keeps exposure < 1 when risk-off)
        if use_enhanced_features and len(hist) > 0 and len(w) > 0:
            try:
                regime_metrics  = compute_regime_metrics(hist)
                regime_exposure = get_regime_adjusted_exposure(regime_metrics)
                w = w * regime_exposure
            except Exception:
                pass

        if w is None or len(w) == 0 or np.isclose(w.sum(), 0.0) or m not in fwd.index:
            rets.loc[m] = 0.0
            tno.loc[m]  = 0.0
            prev_w = pd.Series(dtype=float)
            continue

        # Next-month return with these weights
        valid = [t for t in w.index if t in fwd.columns]
        rets.loc[m] = float((fwd.loc[m, valid] * w.reindex(valid).fillna(0.0)).sum())

        # Turnover (0.5 * L1 distance)
        tno.loc[m] = 0.5 * float(
            (w.reindex(prev_w.index, fill_value=0.0) - prev_w.reindex(w.index, fill_value=0.0)).abs().sum()
        )
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
# Screening utils (liquidity + fundamentals)
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

def fetch_fundamental_metrics(tickers: List[str]) -> pd.DataFrame:
    """Fetch basic fundamental metrics for the given tickers."""
    rows = []
    for t in tickers:
        profitability = np.nan
        leverage = np.nan
        try:
            info = yf.Ticker(t).info
            profitability = info.get("returnOnAssets")
            if profitability is None:
                profitability = info.get("profitMargins")
            leverage = info.get("debtToEquity")
            if leverage is not None and leverage > 10:
                # many providers return percentage values
                leverage = leverage / 100.0
        except Exception:
            pass
        rows.append({"Ticker": t, "profitability": profitability, "leverage": leverage})
    return pd.DataFrame(rows).set_index("Ticker")

def fundamental_quality_filter(
    fund_df: pd.DataFrame,
    min_profitability: float = 0.0,
    max_leverage: float = 2.0,
    profitability_col: str = "profitability",
    leverage_col: str = "leverage",
) -> List[str]:
    """Return tickers passing simple fundamental quality rules."""
    if fund_df.empty:
        return []
    if profitability_col not in fund_df.columns or leverage_col not in fund_df.columns:
        return fund_df.index.tolist()
    df = fund_df.copy()
    mask = df[profitability_col].fillna(-np.inf) >= min_profitability
    mask &= df[leverage_col].fillna(np.inf) <= max_leverage
    return df.index[mask].tolist()

# =========================
# Live portfolio builders (ISA MONTHLY LOCK + stickiness + sector caps) - MODIFIED
# =========================
def _build_isa_weights_fixed(
    daily_close: pd.DataFrame,
    preset: Dict,
    sectors_map: Dict[str, str]
) -> pd.Series:
    """Apply position sizing + hierarchical caps (name/sector + Software sub-caps) to the final combined portfolio."""
    debug_caps = bool(st.session_state.get("debug_caps", False))

    if debug_caps:
        st.info(f"🔧 _build_isa_weights_fixed(): frame has {len(daily_close.columns)} stocks")

    monthly = daily_close.resample("ME").last()

    # --- Momentum Component (NO CAPS YET) ---
    comp_all = composite_score(daily_close)
    comp_vec = comp_all.iloc[-1].dropna() if isinstance(comp_all, pd.DataFrame) else pd.Series(comp_all).dropna()

    momz = blended_momentum_z(monthly)
    pos_idx = momz[momz > 0].index
    comp_vec = comp_vec.reindex(pos_idx).dropna()
    top_m = comp_vec.nlargest(preset["mom_topn"]) if not comp_vec.empty else pd.Series(dtype=float)

    # Stickiness filter (keep names that have persisted in the top set)
    stable_names = set(
        momentum_stable_names(
            daily_close,
            top_n=preset["mom_topn"],
            days=preset.get("stickiness_days", 7)  # <- align key name with the rest of the app
        )
    )
    if stable_names and not top_m.empty:
        filtered = top_m.reindex([t for t in top_m.index if t in stable_names]).dropna()
        if not filtered.empty:
            top_m = filtered

    # Raw momentum weights (scaled by sleeve weight)
    mom_raw = (top_m / top_m.sum()) * preset["mom_w"] if not top_m.empty and top_m.sum() > 0 else pd.Series(dtype=float)

    # --- Mean Reversion Component (NO CAPS YET) ---
    st_ret  = daily_close.pct_change(preset["mr_lb"]).iloc[-1]
    long_ma = daily_close.rolling(preset["mr_ma"]).mean().iloc[-1]
    quality = (daily_close.iloc[-1] > long_ma)
    pool    = [t for t, ok in quality.items() if ok]
    dips    = st_ret.reindex(pool).dropna().nsmallest(preset["mr_topn"])
    mr_raw  = (pd.Series(1 / len(dips), index=dips.index) * preset["mr_w"]) if len(dips) > 0 else pd.Series(dtype=float)

    # --- Combine Components BEFORE Applying Caps ---
    combined_raw = mom_raw.add(mr_raw, fill_value=0.0)
    if combined_raw.empty or combined_raw.sum() <= 0:
        if debug_caps:
            st.warning("⚠️ combined_raw is empty; returning early.")
        return combined_raw

    if debug_caps:
        st.info(f"🔧 Pre-caps portfolio: {len(combined_raw)} positions totaling {combined_raw.sum():.1%}")

    # Apply risk parity weights before enforcing caps
    rp_weights = risk_parity_weights(daily_close, combined_raw.index.tolist())
    combined_raw = combined_raw.mul(rp_weights, fill_value=0.0)
    combined_raw = combined_raw / combined_raw.sum() if combined_raw.sum() > 0 else combined_raw

    # Enhanced sector map (uses your base sectors_map) + hierarchical caps for Software sub-buckets
    enhanced_sectors = get_enhanced_sector_map(list(combined_raw.index), base_map=sectors_map)
    group_caps = build_group_caps(enhanced_sectors)  # <- adds Software parent (30%) + sub-caps (e.g., 18%)

    if debug_caps:
        sector_breakdown: Dict[str, float] = {}
        for ticker, weight in combined_raw.items():
            sec = enhanced_sectors.get(ticker, "Other")
            sector_breakdown[sec] = sector_breakdown.get(sec, 0.0) + float(weight)
        st.info("📊 Pre-caps sectors: " + str(dict(sorted(sector_breakdown.items(), key=lambda x: x[1], reverse=True))))

    # --- Enforce caps on the COMPLETE portfolio ---
    final_weights = enforce_caps_iteratively(
        combined_raw.astype(float),
        enhanced_sectors,
        name_cap=preset["mom_cap"],
        sector_cap=preset.get("sector_cap", 0.30),
        group_caps=group_caps,             # <- IMPORTANT: turns on the sub-caps
        debug=debug_caps
    )

    if final_weights.empty or final_weights.sum() <= 0:
        if debug_caps:
            st.warning("⚠️ enforce_caps_iteratively returned empty weights; returning empty.")
        return final_weights

    # Optional: sanity check (logs only)
    if debug_caps:
        violations = check_constraint_violations(
            final_weights, enhanced_sectors, preset["mom_cap"], preset.get("sector_cap", 0.30)
        )
        if violations:
            st.warning("⚠️ Post-enforcement violations (unexpected): " + "; ".join(violations))
        else:
            st.success("✅ Post-enforcement: no constraint violations detected.")

    # Keep weights summing to 1 across equities (cash is whatever is left at the portfolio level)
    return final_weights / final_weights.sum() if final_weights.sum() > 0 else final_weights

def check_constraint_violations(weights: pd.Series, sectors_map: Dict[str, str], 
                              name_cap: float, sector_cap: float) -> List[str]:
    """
    Check for constraint violations in final portfolio
    Returns list of violation descriptions
    """
    violations = []
    
    # Check name caps
    for ticker, weight in weights.items():
        if weight > name_cap + 0.01:  # 1% tolerance
            violations.append(f"{ticker}: {weight:.1%} > {name_cap:.1%}")
    
    # Check sector caps
    sectors = pd.Series({t: sectors_map.get(t, "Unknown") for t in weights.index})
    sector_sums = weights.groupby(sectors).sum()
    
    for sector, total_weight in sector_sums.items():
        if total_weight > sector_cap + 0.01:  # 1% tolerance
            violations.append(f"{sector}: {total_weight:.1%} > {sector_cap:.1%}")
    
    return violations

def _format_display(weights: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    display_df = pd.DataFrame({"Weight": weights}).sort_values("Weight", ascending=False)
    display_fmt = display_df.copy()
    display_fmt["Weight"] = display_fmt["Weight"].map("{:.2%}".format)
    return display_fmt, display_df

def generate_live_portfolio_isa_monthly(
    preset: Dict,
    prev_portfolio: Optional[pd.DataFrame],
    min_dollar_volume: float = 0.0,
    as_of: date | None = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """
    Enhanced ISA Dynamic live weights with MONTHLY LOCK + composite + stability + sector caps.
    Now includes regime awareness, volatility adjustments, and an optional `as_of` date.
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

    today = as_of or date.today()
    # Fetch prices
    start = (today - relativedelta(months=max(preset["mom_lb"], 12) + 8)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")
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

    # Fundamental quality filter
    min_prof = st.session_state.get("min_profitability", 0.0)
    max_lev = st.session_state.get("max_leverage", 2.0)
    fundamentals = fetch_fundamental_metrics(close.columns.tolist())
    keep = fundamental_quality_filter(fundamentals, min_profitability=min_prof, max_leverage=max_lev)
    if not keep:
        return None, None, "No tickers pass quality filter."
    close = close[keep]
    sectors_map = {t: sectors_map.get(t, "Unknown") for t in close.columns}

    # Monthly lock check — hold previous if not first trading day
    is_monthly = is_rebalance_today(today, close.index)
    decision = "Not the monthly rebalance day — holding previous portfolio."
    if not is_monthly:
        if prev_portfolio is not None and not prev_portfolio.empty:
            disp, raw = _format_display(prev_portfolio["Weight"].astype(float))
            return disp, raw, decision
        else:
            decision = "No saved portfolio; proposing initial allocation (monthly lock applies from next month)."

    # Build candidate weights (enhanced) – keep UI overrides!
    params = dict(STRATEGY_PRESETS["ISA Dynamic (0.75)"])
    params["stability_days"] = int(stickiness_days)
    params["sector_cap"]     = float(sector_cap)
    params["mom_cap"]        = float(st.session_state.get("name_cap", params.get("mom_cap", 0.25)))

    new_w = _build_isa_weights_fixed(close, params, sectors_map)

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
    enable_hedge: bool = False,
    hedge_size: float = 0.0,
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:
    """
    Enhanced ISA-Dynamic hybrid backtest with new features and optional QQQ hedge.
    """
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

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

    # Fundamental quality filter
    min_prof = st.session_state.get("min_profitability", 0.0)
    max_lev = st.session_state.get("max_leverage", 2.0)
    fundamentals = fetch_fundamental_metrics(daily.columns.tolist())
    keep = fundamental_quality_filter(fundamentals, min_profitability=min_prof, max_leverage=max_lev)
    if not keep:
        return None, None, None, None
    daily = daily[keep]
    sectors_map = {t: sectors_map.get(t, "Unknown") for t in keep}

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
    mr_rets, mr_tno = run_backtest_mean_reversion(
        daily, lookback_period_mr=21, top_n_mr=mr_topn, long_ma_days=200
    )

    # Combine + costs
    hybrid_gross, hybrid_tno = combine_hybrid(
        mom_rets, mr_rets, mom_tno, mr_tno, mom_w=mom_weight, mr_w=mr_weight
    )

    # Apply drawdown-based exposure adjustment (walk-forward)
    if use_enhanced_features:
        qqq_monthly = qqq.resample('ME').last().pct_change()
        hybrid_gross = apply_dynamic_drawdown_scaling(
            hybrid_gross, qqq_monthly, threshold_fraction=0.8
        )

    # Optional QQQ hedge taken from cash
    if enable_hedge and hedge_size > 0:
        qqq_monthly = qqq.resample("ME").last().pct_change().reindex(hybrid_gross.index)
        hedged = hybrid_gross.copy()
        for dt in hedged.index:
            # Use history up to previous month for correlation
            past_rets = hedged.loc[:dt].iloc[:-1]
            hist_daily = daily.loc[:dt]
            regime_metrics = compute_regime_metrics(hist_daily)
            hedge_w = build_hedge_weight(past_rets, regime_metrics, hedge_size)
            if hedge_w > 0 and pd.notna(qqq_monthly.loc[dt]):
                hedged.loc[dt] = hedged.loc[dt] - qqq_monthly.loc[dt] * hedge_w
        hybrid_gross = hedged

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
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    try:
        px = data["Close"]
    except Exception:
        # Fallback: take first column if "Close" not present (e.g., FRED series)
        px = data.iloc[:, 0] if hasattr(data, "iloc") else data
    px = _safe_series(px)
    return pd.Series(px).dropna()

def compute_regime_metrics(universe_prices_daily: pd.DataFrame) -> Dict[str, float]:
    """Enhanced regime metrics calculation"""
    if universe_prices_daily.empty:
        return {}
    start = (universe_prices_daily.index.min() - pd.DateOffset(days=5)).strftime("%Y-%m-%d")
    end   = (universe_prices_daily.index.max() + pd.DateOffset(days=5)).strftime("%Y-%m-%d")
    qqq = get_benchmark_series("QQQ", start, end).reindex(universe_prices_daily.index).ffill().dropna()

    # Additional benchmarks for regime metrics
    try:
        vix = get_benchmark_series("^VIX", start, end).reindex(universe_prices_daily.index).ffill()
        vix3m = get_benchmark_series("^VIX3M", start, end).reindex(universe_prices_daily.index).ffill()
        vix_ts = float(vix3m.iloc[-1] / vix.iloc[-1]) if len(vix) and len(vix3m) else np.nan
    except Exception:
        vix_ts = np.nan
    try:
        hy_oas = get_benchmark_series("BAMLH0A0HYM2", start, end).reindex(universe_prices_daily.index).ffill()
        hy_oas_last = float(hy_oas.iloc[-1]) if len(hy_oas) else np.nan
    except Exception:
        hy_oas_last = np.nan

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
        "qqq_50dma_slope_10d": float(qqq_slope_50) if pd.notna(qqq_slope_50) else np.nan,
        "vix_term_structure": float(vix_ts) if pd.notna(vix_ts) else np.nan,
        "hy_oas": float(hy_oas_last) if pd.notna(hy_oas_last) else np.nan,
    }

def get_market_regime() -> Tuple[str, Dict[str, float]]:
    """
    Enhanced market regime detection with additional context
    """
    try:
        univ = st.session_state.get("universe", "Hybrid Top150")
        base_tickers, _, _ = get_universe(univ)
        end = date.today().strftime("%Y-%m-%d")
        start = (date.today() - relativedelta(months=12)).strftime("%Y-%m-%d")
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


def assess_market_conditions(as_of: date | None = None) -> Dict[str, Any]:
    """Assess market conditions and derive configuration settings.

    Parameters
    ----------
    as_of : date or None
        Evaluation date.  Defaults to today when ``None``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with two keys:
        ``metrics`` – market and macro metrics derived from
        :func:`compute_regime_metrics` and ``get_market_regime``.
        ``settings`` – recommended caps and other knobs for the
        strategy based on those metrics.
    """

    asof_ts = pd.Timestamp(as_of or date.today()).normalize()

    metrics: Dict[str, Any] = {}
    try:
        # Build universe and price history for the window preceding ``as_of``
        univ = st.session_state.get("universe", "Hybrid Top150")
        base_tickers, _, _ = get_universe(univ)
        end = asof_ts.strftime("%Y-%m-%d")
        start = (asof_ts - relativedelta(months=12)).strftime("%Y-%m-%d")
        hist = fetch_market_data(base_tickers, start, end)
        metrics.update(compute_regime_metrics(hist))
    except Exception:
        metrics = {}

    # Add regime label (uses existing helper which internally recomputes metrics)
    regime_label, _ = get_market_regime()
    metrics["regime"] = regime_label

    # Derive settings based on key thresholds
    breadth = metrics.get("breadth_pos_6m", 0.5)
    vol10 = metrics.get("qqq_vol_10d", 0.02)
    vix_ts = metrics.get("vix_term_structure", 1.0)
    hy_oas = metrics.get("hy_oas", 4.0)

    vix_thresh = st.session_state.get("vix_ts_threshold", VIX_TS_THRESHOLD_DEFAULT)
    oas_thresh = st.session_state.get("hy_oas_threshold", HY_OAS_THRESHOLD_DEFAULT)

    sector_cap = 0.30
    name_cap = 0.25
    stickiness_days = 7

    risk_off = (
        breadth < 0.35
        or vol10 > 0.045
        or vix_ts < vix_thresh
        or hy_oas > oas_thresh
    )

    risk_on = (
        breadth > 0.65
        and vol10 < 0.025
        and vix_ts >= vix_thresh
        and hy_oas < max(0.0, oas_thresh - 1)
    )

    if risk_off:
        sector_cap = 0.20
        name_cap = 0.20
        stickiness_days = 14
    elif risk_on:
        sector_cap = 0.35
        name_cap = 0.30
        stickiness_days = 5

    settings = {
        "sector_cap": float(sector_cap),
        "name_cap": float(name_cap),
        "stickiness_days": int(stickiness_days),
    }

    return {"metrics": metrics, "settings": settings}

# =========================
# NEW: Market Condition Assessment Helpers
# =========================
def map_metrics_to_settings(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Map regime metrics to strategy settings.

    Parameters
    ----------
    metrics : Dict[str, float]
        Output of :func:`compute_regime_metrics` or any dictionary providing
        the same keys.  The function is agnostic to missing values – absent
        metrics simply leave the corresponding setting unchanged.

    Returns
    -------
    Dict[str, Any]
        Dictionary with recommended strategy adjustments:

        ``target_exposure``
            Base equity allocation (starts at ``1.0``).  It is scaled down
            when risk signals arise.
        ``name_cap`` and ``sector_cap``
            Maximum weights for single names and sectors.  Stress metrics
            tighten these caps.

    Notes
    -----
    The following heuristic rules are applied:

    - **VIX term structure** (`vix_term_structure`): a ratio below the
      configured threshold (contango → backwardation) indicates volatility
      stress.  Exposure is cut by 20 % and both caps are tightened by 10 %.
    - **High‑yield OAS** (`hy_oas`): spreads above the configured threshold
      signal credit stress.  Exposure is reduced by an additional 20 % and
      caps are tightened by 20 %.
    - **Short‑term volatility** (`qqq_vol_10d`): high realised volatility
      (>3.5 %) trims exposure by 10 %; very low vol (<1.5 %) allows a small
      5 % increase.
    - **Market breadth** (`breadth_pos_6m`): poor breadth (<45 %) cuts
      exposure by 10 %; strong breadth (>65 %) adds 5 %.
    - **Trend** (`qqq_above_200dma`): if the benchmark is below its
      long‑term average the exposure is reduced by 10 %.

    These numbers are intentionally simple – they merely provide a
    deterministic, easily testable mapping rather than an optimised model.
    """

    if not metrics:
        # Return defaults when no information is supplied
        base_name_cap = float(st.session_state.get("name_cap", 0.25))
        base_sector_cap = float(st.session_state.get("sector_cap", 0.30))
        return {
            "target_exposure": 1.0,
            "name_cap": base_name_cap,
            "sector_cap": base_sector_cap,
        }

    # Start from user configured caps so the mapping is relative to the
    # current strategy settings.
    exposure = get_regime_adjusted_exposure(metrics)
    name_cap = float(st.session_state.get("name_cap", 0.25))
    sector_cap = float(st.session_state.get("sector_cap", 0.30))

    vix_thresh = st.session_state.get("vix_ts_threshold", VIX_TS_THRESHOLD_DEFAULT)
    oas_thresh = st.session_state.get("hy_oas_threshold", HY_OAS_THRESHOLD_DEFAULT)

    vix_ts = metrics.get("vix_term_structure", np.nan)
    if pd.notna(vix_ts) and vix_ts < vix_thresh:
        exposure *= 0.8
        name_cap *= 0.9
        sector_cap *= 0.9

    hy_oas = metrics.get("hy_oas", np.nan)
    if pd.notna(hy_oas) and hy_oas > oas_thresh:
        exposure *= 0.8
        name_cap *= 0.8
        sector_cap *= 0.8
    elif pd.notna(hy_oas) and hy_oas < max(0.0, oas_thresh - 2):
        exposure *= 1.05

    vol10 = metrics.get("qqq_vol_10d", np.nan)
    if pd.notna(vol10) and vol10 > 0.035:
        exposure *= 0.9
    elif pd.notna(vol10) and vol10 < 0.015:
        exposure *= 1.05

    breadth = metrics.get("breadth_pos_6m", np.nan)
    if pd.notna(breadth) and breadth < 0.45:
        exposure *= 0.9
    elif pd.notna(breadth) and breadth > 0.65:
        exposure *= 1.05

    qqq_above = metrics.get("qqq_above_200dma", np.nan)
    if pd.notna(qqq_above) and qqq_above < 1.0:
        exposure *= 0.9

    return {
        "target_exposure": float(np.clip(exposure, 0.3, 1.2)),
        "name_cap": float(max(0.05, name_cap)),
        "sector_cap": float(max(0.10, sector_cap)),
    }


def assess_market_conditions(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Convert raw regime metrics into a qualitative assessment.

    Parameters
    ----------
    metrics : Dict[str, float]
        Regime metrics as produced by :func:`compute_regime_metrics`.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:

        ``label``
            Human readable regime description based on the recommended
            exposure level.
        ``settings``
            Output of :func:`map_metrics_to_settings`.
        ``metrics``
            Echo of the input metrics for downstream use.
    """

    settings = map_metrics_to_settings(metrics)
    exposure = settings.get("target_exposure", 1.0)

    # Label regimes using simple exposure bands so that unit tests can make
    # deterministic assertions.
    if exposure >= 1.05:
        label = "Strong Risk-On"
    elif exposure >= 0.95:
        label = "Risk-On"
    elif exposure >= 0.75:
        label = "Cautious Risk-On"
    elif exposure >= 0.5:
        label = "Risk-Off"
    else:
        label = "Extreme Risk-Off"

    return {"label": label, "settings": settings, "metrics": metrics}

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
        end_date = date.today().strftime("%Y-%m-%d")
        start_date = (date.today() - relativedelta(days=40)).strftime("%Y-%m-%d")
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
