# backend.py — Enhanced Hybrid Top150 / Composite Rank / Sector Caps / Stickiness / ISA lock
from __future__ import annotations

import logging
import os, io, warnings, json, hashlib
from typing import Optional, Tuple, Dict, List, Any, Callable, Iterable
from dataclasses import replace, asdict

import numpy as np
import pandas as pd

# Optional Polars support (convert to pandas if provided)
try:  # don't hard-require polars
    import polars as pl  # type: ignore
    _HAS_POLARS = True
except Exception:
    pl = None  # type: ignore
    _HAS_POLARS = False
import yfinance as yf
import requests
from io import StringIO
import logging
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover - optional dependency fallback
    Parallel = None  # type: ignore[assignment]
    delayed = None  # type: ignore[assignment]
try:
    from numba import njit  # type: ignore[import]
    NUMBA_OK = True
except Exception:  # pragma: no cover - optional dependency fallback
    NUMBA_OK = False

    def njit(*args, **kwargs):  # type: ignore[override]
        if args and callable(args[0]):
            return args[0]

        def _decorator(func):
            return func

        return _decorator
import optimizer
import strategy_core
from strategy_core import HybridConfig
from portfolio_utils import cap_weights, l1_turnover

# Bridge to strategy_core sanitizer to keep a single source of truth
try:
    from strategy_core import _sanitize_tickers as _sc_sanitize_tickers  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback if module layout changes
    def _sc_sanitize_tickers(tickers: Iterable[str]) -> list[str]:
        return sorted({(str(x).strip().upper()) for x in (tickers or []) if str(x).strip()})

warnings.filterwarnings("ignore")

# =========================
# Backtest cache helpers
# =========================

def _build_hybrid_cache_key(
    tickers: Iterable[str],
    cfg: HybridConfig,
    start: pd.Timestamp,
    apply_vol_target: bool,
) -> str:
    payload = {
        "tickers": sorted(str(t) for t in tickers),
        "cfg": asdict(cfg),
        "start": pd.to_datetime(start).strftime("%Y-%m-%d"),
        "apply_vol_target": bool(apply_vol_target),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _run_hybrid_backtest_with_cache(
    daily_prices: pd.DataFrame,
    cfg: HybridConfig,
    cache_key: str,
    apply_vol_target: bool,
    use_incremental: bool = True,
    get_constituents: Optional[Callable[[pd.Timestamp], Iterable[str]]] = None,
) -> Dict[str, pd.Series]:
    def _run_full() -> Dict[str, pd.Series]:
        kwargs: Dict[str, Any] = {"apply_vol_target": apply_vol_target}
        if get_constituents is not None:
            kwargs["get_constituents"] = get_constituents
        return strategy_core.run_hybrid_backtest(daily_prices, cfg, **kwargs)

    if not use_incremental or not cache_key:
        return _run_full()

    cached = strategy_core.load_backtest_cache(cache_key)
    if cached is not None:
        cached_series = cached.get("hybrid_rets_net", pd.Series(dtype=float))
        if not isinstance(cached_series, pd.Series):
            cached_series = pd.Series(dtype=float)
        cached_idx = cached_series.index
        monthly_idx = daily_prices.resample("M").last().index
        extends = False
        if len(cached_idx) > 0 and len(monthly_idx) > 0:
            extends = monthly_idx.min() >= cached_idx.min() and monthly_idx.max() > cached_idx.max()
        if extends:
            kwargs: Dict[str, Any] = {"apply_vol_target": apply_vol_target}
            if get_constituents is not None:
                kwargs["get_constituents"] = get_constituents
            return strategy_core.run_hybrid_backtest_incremental(
                daily_prices,
                cfg,
                cache_key=cache_key,
                **kwargs,
            )

    result = _run_full()
    if use_incremental and cache_key:
        strategy_core.save_backtest_cache(cache_key, result)
    return result

# =========================
# Optional numba helpers
# =========================
_Z_EPS = 1e-9

@njit(cache=True)
def _mad_scale(values: np.ndarray) -> tuple[float, float]:  # pragma: no cover - exercised via wrapper
    count = 0
    for v in values:
        if np.isfinite(v):
            count += 1

    if count == 0:
        return 0.0, 0.0

    clean = np.empty(count, dtype=np.float64)
    idx = 0
    for v in values:
        if np.isfinite(v):
            clean[idx] = v
            idx += 1

    clean.sort()
    mid = count // 2
    if count % 2 == 0:
        median = 0.5 * (clean[mid - 1] + clean[mid])
    else:
        median = clean[mid]

    deviations = np.empty(count, dtype=np.float64)
    for i in range(count):
        deviations[i] = abs(clean[i] - median)

    deviations.sort()
    if count % 2 == 0:
        mad = 0.5 * (deviations[mid - 1] + deviations[mid])
    else:
        mad = deviations[mid]

    if mad > 0.0:
        return median, 1.4826 * mad

    mean_val = 0.0
    for v in clean:
        mean_val += v
    mean_val /= count

    var = 0.0
    for v in clean:
        diff = v - mean_val
        var += diff * diff
    if count > 0:
        var /= count

    std = np.sqrt(var) if var > 0.0 else 0.0
    return median, std

@njit(cache=True)
def _nanmean_std(values: np.ndarray) -> tuple[float, float, int]:  # pragma: no cover - exercised via wrapper
    count = 0
    mean_val = 0.0
    m2 = 0.0
    for v in values:
        if not np.isfinite(v):
            continue
        count += 1
        delta = v - mean_val
        mean_val += delta / count
        delta2 = v - mean_val
        m2 += delta * delta2

    if count == 0:
        return np.nan, np.nan, 0

    variance = m2 / count if count > 0 else np.nan
    if variance < 0.0:
        variance = 0.0

    return mean_val, np.sqrt(variance), count

@njit(cache=True)
def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:  # pragma: no cover - exercised via wrapper
    n = values.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan

    if window <= 0:
        return out

    for end in range(window - 1, n):
        count = 0
        mean_val = 0.0
        m2 = 0.0
        start = end - window + 1
        for idx in range(start, end + 1):
            v = values[idx]
            if not np.isfinite(v):
                continue
            count += 1
            delta = v - mean_val
            mean_val += delta / count
            delta2 = v - mean_val
            m2 += delta * delta2

        if count == window:
            if count > 1:
                variance = m2 / (count - 1)
                if variance < 0.0:
                    variance = 0.0
                out[end] = np.sqrt(variance)
            else:
                out[end] = np.nan
        else:
            out[end] = np.nan

    return out

@njit(cache=True)
def _last_rolling_mean(values: np.ndarray, window: int) -> float:  # pragma: no cover - exercised via wrapper
    n = values.shape[0]
    if window <= 0 or n < window:
        return np.nan

    start = n - window
    total = 0.0
    for idx in range(start, n):
        v = values[idx]
        if not np.isfinite(v):
            return np.nan
        total += v

    return total / window

def _zscore_series(series: pd.Series, cols: pd.Index) -> pd.Series:
    cleaned = series.replace([np.inf, -np.inf], np.nan)

    if not NUMBA_OK:
        dropped = cleaned.dropna()
        if dropped.empty:
            return pd.Series(0.0, index=cols)
        std = float(dropped.std(ddof=0))
        if std == 0.0 or not np.isfinite(std):
            return pd.Series(0.0, index=cols)
        z_vals = (dropped - dropped.mean()) / (std + _Z_EPS)
        return z_vals.reindex(cols).fillna(0.0)

    values = cleaned.to_numpy(dtype=np.float64, copy=True)
    mask = np.isfinite(values)
    valid = int(mask.sum())
    if valid == 0:
        return pd.Series(0.0, index=cols)

    finite_values = values[mask]
    mean_val, std_val, count = _nanmean_std(finite_values)
    if count == 0 or not np.isfinite(std_val) or std_val == 0.0:
        return pd.Series(0.0, index=cols)

    normalized = (finite_values - mean_val) / (std_val + _Z_EPS)
    result = np.zeros(len(values), dtype=np.float64)
    idx = 0
    for i in range(len(values)):
        if mask[i]:
            result[i] = normalized[idx]
            idx += 1

    series_result = pd.Series(result, index=series.index)
    return series_result.reindex(cols).fillna(0.0)

# =========================
# Config & Secrets
# =========================

# Streamlit might not be available (e.g., non-UI contexts)
try:
    import streamlit as st  # type: ignore
    _HAS_ST = True
except Exception:
    st = None  # type: ignore
    _HAS_ST = False

def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Try Streamlit secrets first; fall back to environment variables."""
    if _HAS_ST:
        try:
            # st.secrets behaves like a dict
            val = st.secrets.get(key)  # type: ignore[attr-defined]
            if val not in (None, ""):
                return str(val)
        except Exception:
            pass
    return os.getenv(key, default)

GIST_ID = _get_secret("GIST_ID")
GITHUB_TOKEN = _get_secret("GITHUB_TOKEN")
GIST_API_URL = f"https://api.github.com/gists/{GIST_ID}" if GIST_ID else None
HEADERS = (
    {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "isa-dynamic/1.0",
    }
    if GITHUB_TOKEN
    else {}
)

GIST_PORTF_FILE   = "portfolio.json"
LIVE_PERF_FILE    = "live_perf.csv"
LOCAL_PORTF_FILE  = "last_portfolio.csv"
ASSESS_LOG_FILE   = "assess_log.csv"

ROUNDTRIP_BPS_DEFAULT   = 20
REGIME_MA               = 200
AVG_TRADE_SIZE_DEFAULT  = 0.02  # 2% avg single-leg trade size
HEDGE_MAX_DEFAULT       = 0.20
HEDGE_TICKER_LABEL      = "QQQ (Hedge)"
HEDGE_TICKER_ALIASES    = {HEDGE_TICKER_LABEL, "QQQ_HEDGE", "QQQ-HEDGE"}

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

# Default mapping from regime metrics to strategy parameters.
PARAM_MAP_DEFAULTS = {
    "low_vol": 0.02,
    "high_vol": 0.04,
    "top_n_low": 10,
    "top_n_mid": 8,
    "top_n_high": 5,
    "name_cap_low": 0.30,
    "name_cap_mid": 0.25,
    "name_cap_high": 0.20,
    "sector_cap_low": 0.35,
    "sector_cap_mid": 0.30,
    "sector_cap_high": 0.25,
}

_YF_BATCH_SIZE = 200

PERF: Dict[str, Any] = {
    "fast_io": True,
    # Enable the optional polars-backed compute path for faster signal generation
    # by setting this flag to True (requires the polars package).
    "use_polars": False,
    "parallel_grid": False,
    "n_jobs": 4,
    "yf_batch": _YF_BATCH_SIZE,
    "cache_days": 365,
}

def _env_flag(name: str, default: str = "0") -> bool:
    """Interpret environment variable ``name`` as a boolean flag."""
    val = os.getenv(name, default)
    if val is None:
        return False
    return str(val).strip().lower() in {"1", "true", "yes", "on"}

PERF.update({
    "fast_io": _env_flag("FAST_IO", "0"),
})

def _emit_info(msg: str, info: Optional[Callable[[str], None]] = None) -> None:
    """Prefer provided info callback, then Streamlit (if available), else logging."""
    # 1) Caller-provided callback
    if callable(info):
        try:
            info(msg)
            return
        except Exception:
            pass

    # 2) Streamlit info (if we're running under Streamlit)
    if _HAS_ST:
        try:
            st.info(msg)
            return
        except Exception:
            pass

    # 3) Fallback to logging
    logging.info(msg)

def _record_hedge_state(scope: str,
                        weight: float,
                        correlation: float | None,
                        regime_metrics: Dict[str, float]) -> None:
    """Persist the latest hedge details for UI/reporting purposes."""
    if not _HAS_ST:
        return

    summary = {
        "weight": float(weight or 0.0),
        "correlation": (None if correlation is None or pd.isna(correlation)
                         else float(correlation)),
        "qqq_above_200dma": regime_metrics.get("qqq_above_200dma"),
        "breadth_pos_6m": regime_metrics.get("breadth_pos_6m"),
        "timestamp": datetime.utcnow().isoformat(),
    }
    st.session_state[f"latest_{scope}_hedge"] = summary

# =========================
# NEW: Enhanced Data Validation & Cleaning
# =========================
import numpy as np
import pandas as pd

def linear_interpolate_short_gaps(
    prices: pd.DataFrame,
    max_gap: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Linearly interpolate gaps up to `max_gap` consecutive business days per column.

    Parameters
    ----------
    prices : DataFrame
        Price DataFrame indexed by date, columns are tickers.
    max_gap : int
        Maximum consecutive NaNs to fill per gap (gaps longer than this are left as NaN).

    Returns
    -------
    filled : DataFrame
        Copy of `prices` with short gaps linearly imputed.
    imputed_mask : DataFrame[bool]
        True where values were imputed by this routine.
    """
    if prices is None or prices.empty:
        return prices, pd.DataFrame(index=getattr(prices, "index", []),
                                    columns=getattr(prices, "columns", []))

    filled = prices.copy()
    imputed_mask = pd.DataFrame(False, index=filled.index, columns=filled.columns)

    for column in filled.columns:
        series = filled[column]
        na_flags = series.isna()
        if not na_flags.any():
            continue

        # Find runs of consecutive NaNs
        gaps: list[tuple[int, int]] = []
        start = None
        for i, is_na in enumerate(na_flags.values):
            if is_na and start is None:
                start = i
            elif (not is_na) and start is not None:
                gaps.append((start, i - 1))
                start = None
        if start is not None:
            gaps.append((start, len(series) - 1))

        # Fill short gaps using linear interpolation between surrounding points
        for s, e in gaps:
            gap_len = e - s + 1
            if gap_len > max_gap:
                continue

            left = s - 1
            right = e + 1
            # Need valid endpoints on both sides to interpolate
            left_val = series.iat[left] if left >= 0 else np.nan
            right_val = series.iat[right] if right < len(series) else np.nan

            if left < 0 or pd.isna(left_val):
                if pd.isna(right_val):
                    continue
                idx_slice = series.index[s : e + 1]
                filled.loc[idx_slice, column] = right_val
                imputed_mask.loc[idx_slice, column] = True
                continue

            if right >= len(series) or pd.isna(right_val):
                idx_slice = series.index[s : e + 1]
                filled.loc[idx_slice, column] = left_val
                imputed_mask.loc[idx_slice, column] = True
                continue

            idx_slice = series.index[s : e + 1]
            # Linear ramp from left_val to right_val (exclude endpoints)
            steps = np.arange(1, gap_len + 1, dtype=float) / (gap_len + 1)
            vals = float(left_val) + steps * (float(right_val) - float(left_val))

            filled.loc[idx_slice, column] = vals
            imputed_mask.loc[idx_slice, column] = True

    return filled, imputed_mask

def _use_polars_engine() -> bool:
    """Return True when the optimized polars path should be used."""
    return bool(PERF.get("use_polars") and _HAS_POLARS and pl is not None)

def _prepare_polars_daily_frame(daily: pd.DataFrame) -> tuple["pl.DataFrame", list[str]]:
    """Convert a pandas daily price frame into a polars DataFrame and column list."""
    if not _use_polars_engine():  # pragma: no cover - guarded by caller
        raise RuntimeError("polars is not available")

    if isinstance(daily.index, pd.DatetimeIndex):
        daily_sorted = daily.sort_index()
        if daily_sorted.index.tz is not None:
            daily_sorted = daily_sorted.copy()
            daily_sorted.index = daily_sorted.index.tz_localize(None)
    else:
        daily_sorted = daily.sort_index()

    idx_name = daily_sorted.index.name or "date"
    daily_reset = daily_sorted.reset_index().rename(columns={idx_name: "date"})
    pl_df = pl.from_pandas(daily_reset)
    if "date" in pl_df.columns and pl_df["date"].dtype != pl.Datetime:
        pl_df = pl_df.with_columns(pl.col("date").cast(pl.Datetime))

    value_cols = [col for col in pl_df.columns if col != "date"]
    return pl_df.sort("date"), value_cols

def _polars_to_pandas_indexed(df: "pl.DataFrame") -> pd.DataFrame:
    """Convert a polars frame with a date column back to a pandas DataFrame index."""
    if df.height == 0:
        return pd.DataFrame()
    pdf = df.to_pandas()
    if "date" in pdf.columns:
        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf.set_index("date")
    return pdf.sort_index()

# ---------- Factor z-score helpers (Pandas-first, optional Polars input) ----------

def _to_pandas(obj):
    """Convert Polars DataFrame/Series to pandas if needed; otherwise return as-is."""
    if obj is None:
        return None
    try:
        if '_HAS_POLARS' in globals() and _HAS_POLARS and 'pl' in globals():
            if isinstance(obj, pl.DataFrame) or isinstance(obj, pl.Series):
                return obj.to_pandas()
    except Exception:
        # Fall through to return original object
        pass
    return obj


def blended_momentum_z(monthly: pd.DataFrame | "pl.DataFrame") -> pd.Series:
    """
    Blended momentum z-score using 3/6/12M horizons with weights 0.2/0.4/0.4.
    Accepts pandas or polars DataFrame (prices at month end).
    """
    if monthly is None:
        return pd.Series(dtype=float)

    monthly = _to_pandas(monthly)
    if not isinstance(monthly, pd.DataFrame) or monthly.shape[0] < 13:
        return pd.Series(dtype=float)

    # Ensure chronological order
    try:
        monthly = monthly.sort_index()
    except Exception:
        pass

    cols = monthly.columns
    r3 = monthly.pct_change(3).iloc[-1]
    r6 = monthly.pct_change(6).iloc[-1]
    r12 = monthly.pct_change(12).iloc[-1]

    z3 = _zscore_series(r3, cols)
    z6 = _zscore_series(r6, cols)
    z12 = _zscore_series(r12, cols)

    return 0.2 * z3 + 0.4 * z6 + 0.4 * z12


def lowvol_z(
    daily: pd.DataFrame | "pl.DataFrame",
    vol_series: pd.Series | "pl.Series" | None = None,
) -> pd.Series:
    """
    Low-volatility factor as a z-score of 63-day rolling std of daily returns.
    If 'vol_series' is given (precomputed vol), z-score that instead.
    Accepts pandas or polars inputs.
    """
    if daily is None:
        return pd.Series(dtype=float)

    daily = _to_pandas(daily)
    vol_series = _to_pandas(vol_series)

    if not isinstance(daily, pd.DataFrame):
        return pd.Series(dtype=float)
    cols = daily.columns

    # Use precomputed vol if provided
    if vol_series is not None:
        try:
            return _zscore_series(pd.Series(vol_series), cols)
        except Exception:
            # fall through to compute from daily
            pass

    if daily.shape[0] < 80:  # need enough data for a stable 63d window
        return pd.Series(0.0, index=cols)

    vol = (
        daily.pct_change()
        .rolling(63)
        .std()
        .iloc[-1]
        .replace([np.inf, -np.inf], np.nan)
    )
    return _zscore_series(vol, cols)


def trend_z(
    daily: pd.DataFrame | "pl.DataFrame",
    dist_series: pd.Series | "pl.Series" | None = None,
) -> pd.Series:
    """
    Trend factor as z-score of distance from 200DMA: (last / MA200 - 1).
    If 'dist_series' is given, z-score that instead.
    Accepts pandas or polars inputs.
    """
    if daily is None:
        return pd.Series(dtype=float)

    daily = _to_pandas(daily)
    dist_series = _to_pandas(dist_series)

    if not isinstance(daily, pd.DataFrame):
        return pd.Series(dtype=float)
    cols = daily.columns

    # Use precomputed distance if provided
    if dist_series is not None:
        try:
            return _zscore_series(pd.Series(dist_series), cols)
        except Exception:
            # fall through to compute from daily
            pass

    if daily.shape[0] < 220:  # ensure we have >=200d + buffer
        return pd.Series(0.0, index=cols)

    ma200 = daily.rolling(200).mean().iloc[-1]
    last = daily.iloc[-1]
    dist = (last / ma200 - 1.0).replace([np.inf, -np.inf], np.nan)
    return _zscore_series(dist, cols)
  
def blended_momentum_z(monthly: pd.DataFrame | "pl.DataFrame") -> pd.Series:
    """
    Blended momentum z-score using 3/6/12M horizons with weights 0.2/0.4/0.4.
    Accepts pandas or polars DataFrame (prices at month end).
    """
    if monthly is None:
        return pd.Series(dtype=float)

    monthly = _to_pandas(monthly)
    if not isinstance(monthly, pd.DataFrame) or monthly.shape[0] < 13:
        return pd.Series(dtype=float)

    # Ensure chronological order
    try:
        monthly = monthly.sort_index()
    except Exception:
        pass

    cols = monthly.columns
    r3 = monthly.pct_change(3).iloc[-1]
    r6 = monthly.pct_change(6).iloc[-1]
    r12 = monthly.pct_change(12).iloc[-1]

    z3 = _zscore_series(r3, cols)
    z6 = _zscore_series(r6, cols)
    z12 = _zscore_series(r12, cols)

    return 0.2 * z3 + 0.4 * z6 + 0.4 * z12

def lowvol_z(
    daily: pd.DataFrame | "pl.DataFrame",
    vol_series: pd.Series | "pl.Series" | None = None,
) -> pd.Series:
    """
    Low-volatility factor as a z-score of 63-day rolling std of daily returns.
    If 'vol_series' is given (precomputed vol), z-score that instead.
    Accepts pandas or polars inputs.
    """
    if daily is None:
        return pd.Series(dtype=float)

    daily = _to_pandas(daily)
    vol_series = _to_pandas(vol_series)

    if not isinstance(daily, pd.DataFrame):
        return pd.Series(dtype=float)
    cols = daily.columns
    
    # Use precomputed vol if provided
    if vol_series is not None:
        try:
            return _zscore_series(pd.Series(vol_series), cols)
        except Exception:
            # fall through to compute from daily
            pass

    if daily.shape[0] < 80:  # need enough data for a stable 63d window
        return pd.Series(0.0, index=cols)

    vol = (
        daily.pct_change()
        .rolling(63)
        .std()
        .iloc[-1]
        .replace([np.inf, -np.inf], np.nan)
    )
    return _zscore_series(vol, cols)

def trend_z(
    daily: pd.DataFrame | "pl.DataFrame",
    dist_series: pd.Series | "pl.Series" | None = None,
) -> pd.Series:
    """
    Trend factor as z-score of distance from 200DMA: (last / MA200 - 1).
    If 'dist_series' is given, z-score that instead.
    Accepts pandas or polars inputs.
    """
    if daily is None:
        return pd.Series(dtype=float)
    daily = _to_pandas(daily)
    dist_series = _to_pandas(dist_series)

    if not isinstance(daily, pd.DataFrame):
        return pd.Series(dtype=float)
    cols = daily.columns

    # Use precomputed distance if provided
    if dist_series is not None:
        try:
            return _zscore_series(pd.Series(dist_series), cols)
        except Exception:
            # fall through to compute from daily
            pass

    if daily.shape[0] < 220:  # ensure we have >=200d + buffer
        return pd.Series(0.0, index=cols)

    ma200 = daily.rolling(200).mean().iloc[-1]
    last = daily.iloc[-1]
    dist = (last / ma200 - 1.0).replace([np.inf, -np.inf], np.nan)
    return _zscore_series(dist, cols)

@st.cache_data(ttl=43200)
def compute_signal_panels(daily: pd.DataFrame) -> Dict[str, pd.DataFrame | pd.Series]:
    """Cache common signal inputs derived from daily close data."""
    if daily is None or daily.empty:
        empty_df = pd.DataFrame()
        return {
            "monthly": empty_df,
            "r3": empty_df,
            "r6": empty_df,
            "r12": empty_df,
            "vol63": empty_df,
            "ma200": empty_df,
            "dist200": empty_df,
        }

    daily = daily.dropna(how="all", axis=1)
    if daily.shape[1] == 0:
        empty_df = pd.DataFrame()
        return {
            "monthly": empty_df,
            "r3": empty_df,
            "r6": empty_df,
            "r12": empty_df,
            "vol63": empty_df,
            "ma200": empty_df,
            "dist200": empty_df,
        }

    # ---- Normalize index (unique, sorted, datetime) ----
    daily = _ensure_unique_sorted_index(daily)

    # ---- Prefer Polars path if engine switch exists and returns True ----
    use_polars = False
    try:
        if "_use_polars_engine" in globals() and callable(globals()["_use_polars_engine"]):
            use_polars = bool(_use_polars_engine())
    except Exception:
        use_polars = False

    if use_polars:
        try:
            return _compute_signal_panels_polars(daily)
        except Exception:
            logging.exception("Falling back to pandas signal panel computation")

    # ---- pandas fallback ----
    daily_sorted = daily.sort_index()
    monthly = daily_sorted.resample("M").last()
    r3 = daily_sorted.pct_change(63)
    r6 = daily_sorted.pct_change(126)
    r12 = daily_sorted.pct_change(252)
    daily_returns = daily_sorted.pct_change()
    vol63 = daily_returns.rolling(63).std()
    ma200 = daily_sorted.rolling(200).mean()
    dist200 = (daily_sorted / ma200 - 1).replace([np.inf, -np.inf], np.nan)

    return {
        "monthly": monthly,
        "r3": r3,
        "r6": r6,
        "r12": r12,
        "vol63": vol63,
        "ma200": ma200,
        "dist200": dist200,
    }
  
    if _use_polars_engine():
        try:
            return _compute_signal_panels_polars(daily)
        except Exception:
            logging.exception("Falling back to pandas signal panel computation")

    daily_sorted = daily
    monthly = daily_sorted.resample("M").last()
    r3 = daily_sorted.pct_change(63)
    r6 = daily_sorted.pct_change(126)
    r12 = daily_sorted.pct_change(252)
    daily_returns = daily_sorted.pct_change()
    vol63 = daily_returns.rolling(63).std()
    ma200 = daily_sorted.rolling(200).mean()
    dist200 = (daily_sorted / ma200 - 1).replace([np.inf, -np.inf], np.nan)

    return {
        "monthly": monthly,
        "r3": r3,
        "r6": r6,
        "r12": r12,
        "vol63": vol63,
        "ma200": ma200,
        "dist200": dist200,
    }

def _robust_return_stats(returns: pd.Series) -> tuple[float, float]:
    """Robust location/scale (median & MAD*1.4826) with sane fallbacks."""
    if NUMBA_OK:
        values = returns.to_numpy(dtype=np.float64, copy=False)
        median, scale = _mad_scale(values)
        return float(median), float(scale)

    r = returns.dropna()
    if r.empty:
        return 0.0, 0.0
    med = float(r.median())
    mad = float((r - med).abs().median())
    if mad and mad > 0:
        scale = 1.4826 * mad
    else:
        std = float(r.std(ddof=0))
        scale = std if std and std > 0 else 0.0
    return med, scale

def cap_abnormal_returns(
    prices: pd.DataFrame,
    *,
    max_daily_move: float | None = None,
    zscore_threshold: float | None = 4.0,
    min_price: float | None = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cap abnormal daily returns (robust z-scores and/or hard cap). Returns (cleaned_prices, mask)."""
    if prices is None or prices.empty:
        empty_mask = pd.DataFrame(False, index=getattr(prices, "index", []), columns=getattr(prices, "columns", []))
        return prices.copy() if hasattr(prices, "copy") else pd.DataFrame(), empty_mask

    cleaned = prices.copy()
    mask = pd.DataFrame(False, index=cleaned.index, columns=cleaned.columns)

    for col in cleaned.columns:
        series = cleaned[col].astype(float)
        if series.dropna().shape[0] <= 1:
            continue

        returns = series.pct_change(fill_method=None)
        loc, scale = _robust_return_stats(returns)

        if (zscore_threshold is None or scale <= 0) and max_daily_move is None:
            # nothing to do for this column
            continue

        if scale > 0 and zscore_threshold is not None:
            z = (returns - loc) / scale
            outliers = z.abs() > float(zscore_threshold)
        else:
            outliers = pd.Series(False, index=returns.index)

        if max_daily_move is not None:
            outliers |= returns.abs() > float(max_daily_move)

        if not outliers.any():
            continue

        col_mask = mask[col]

        for idx in outliers.index[outliers]:
            # guard on index and NA
            if idx not in returns.index or pd.isna(returns.loc[idx]):
                continue

            # position of the outlier point in the level series
            try:
                pos = series.index.get_loc(idx)
            except KeyError:
                continue
            if pos <= 0:
                continue  # need a previous valid price to reconstruct

            # previous valid price
            prev_values = series.iloc[:pos].dropna()
            if prev_values.empty:
                continue
            prev_price = float(prev_values.iloc[-1])
            if not np.isfinite(prev_price) or prev_price == 0.0:
                continue

            # clamp the return value by both z-threshold and hard cap (if provided)
            ret_val = float(returns.loc[idx])
            if max_daily_move is not None:
                ret_val = float(np.clip(ret_val, -max_daily_move, max_daily_move))
            if zscore_threshold is not None and scale > 0:
                hi = loc + zscore_threshold * scale
                lo = loc - zscore_threshold * scale
                ret_val = float(np.clip(ret_val, lo, hi))

            # reconstruct a new price
            original_price = float(series.iloc[pos])
            new_price = prev_price * (1.0 + ret_val)
            if min_price is not None and np.isfinite(min_price):
                new_price = max(new_price, float(min_price))

            if not np.isfinite(new_price):
                continue
            if np.isclose(new_price, original_price, rtol=1e-9, atol=1e-12):
                continue

            # apply the change and update masks / neighboring returns
            series.iloc[pos] = new_price
            col_mask.loc[idx] = True

            returns.loc[idx] = ret_val
            if pos + 1 < len(series):
                nxt_idx = series.index[pos + 1]
                nxt_val = series.iloc[pos + 1]
                if nxt_idx in returns.index and pd.notna(nxt_val) and new_price != 0:
                    returns.loc[nxt_idx] = (nxt_val / new_price) - 1.0

        cleaned[col] = series
        mask[col] = col_mask

    return cleaned, mask

def clean_extreme_moves(
    prices: pd.DataFrame,
    *,
    max_daily_move: float = 0.30,
    min_price: float = 0.5,
    zscore_threshold: float | None = 4.0,
    info: Optional[Callable[[str], None]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clamp implausible price swings (robust) and return (cleaned, mask)."""
    if prices is None or prices.empty:
        empty_mask = pd.DataFrame(False, index=getattr(prices, "index", []), columns=getattr(prices, "columns", []))
        return prices.copy() if hasattr(prices, "copy") else pd.DataFrame(), empty_mask

    # Apply a basic floor first
    cleaned = prices.copy()
    floor_mask = pd.DataFrame(False, index=cleaned.index, columns=cleaned.columns)
    if min_price is not None:
        floor_mask = (cleaned < min_price) & cleaned.notna()
        if floor_mask.any().any():
            cleaned[floor_mask] = float(min_price)

    # Robust capping
    capped, cap_mask = cap_abnormal_returns(
        cleaned,
        max_daily_move=max_daily_move,
        zscore_threshold=zscore_threshold,
        min_price=min_price,
    )

    cap_mask = cap_mask.astype(bool).reindex_like(cleaned).fillna(False)
    floor_mask = floor_mask.astype(bool)
    combined_mask = (cap_mask | floor_mask).astype(bool)

    fixes = int(combined_mask.to_numpy().sum())
    if callable(info) and fixes > 0:
        info(f"🧹 Data cleaning: Fixed {fixes} extreme price moves across all stocks")

    return capped, combined_mask

def fill_missing_data(
    prices: pd.DataFrame,
    *,
    max_gap_days: int = 3,
    info: Optional[Callable[[str], None]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill short gaps via *linear interpolation on the price level* and return (filled, mask)."""
    if prices is None or prices.empty:
        empty_mask = pd.DataFrame(False, index=getattr(prices, "index", []), columns=getattr(prices, "columns", []))
        return prices.copy() if hasattr(prices, "copy") else pd.DataFrame(), empty_mask

    # Assumes `linear_interpolate_short_gaps` is defined elsewhere
    filled, mask = linear_interpolate_short_gaps(prices, max_gap=max_gap_days)
    n = int(mask.to_numpy().sum())
    if callable(info) and n > 0:
        info(f"🔧 Data filling: Filled {n} missing data points with interpolation")
    return filled, mask.astype(bool)

def validate_and_clean_market_data(
    prices: pd.DataFrame,
    *,
    max_missing_ratio: float = 0.20,
    max_daily_move: float = 0.30,
    min_price: float = 0.5,
    zscore_threshold: float | None = 4.0,
    max_gap_days: int = 3,
    info: Optional[Callable[[str], None]] = None,
) -> tuple[pd.DataFrame, List[str], pd.DataFrame, pd.DataFrame]:
    """
    Full validation + cleaning pipeline:
      1) Drop tickers with too much missing data
      2) Cap extreme moves (robust z/MAD + optional hard cap)
      3) Interpolate short gaps on levels
    Returns: (cleaned_df, alerts_list, cap_mask, interp_mask)
    """
    if prices is None or prices.empty:
        empty = pd.DataFrame(index=getattr(prices, "index", []))
        empty_mask = pd.DataFrame(False, index=empty.index, columns=[])
        return empty, [], empty_mask.copy(), empty_mask.copy()

    data = prices.copy()
    alerts: List[str] = []
    original_shape = data.shape

    # 1) Drop very sparse tickers
    if max_missing_ratio is not None and data.shape[1] > 0:
        miss_frac = data.isna().mean()
        drop_cols = miss_frac[miss_frac > max_missing_ratio].index.tolist()
        if drop_cols:
            data = data.drop(columns=drop_cols)
            alerts.append(f"Removed {len(drop_cols)} stocks with >{int(round(max_missing_ratio*100))}% missing data")

    if data.shape != original_shape:
        alerts.append(f"Data shape: {original_shape} → {data.shape}")

    if data.empty:
        mask = pd.DataFrame(False, index=prices.index, columns=data.columns)
        return data, alerts, mask.copy(), mask.copy()

    # 2) Cap extreme moves
    cleaned, mask_clean = clean_extreme_moves(
        data,
        max_daily_move=max_daily_move,
        min_price=min_price,
        zscore_threshold=zscore_threshold,
        info=info,
    )

    # 3) Fill short gaps
    filled, mask_fill = fill_missing_data(
        cleaned,
        max_gap_days=max_gap_days,
        info=info,
    )

    cap_mask = mask_clean.reindex_like(filled).fillna(False).astype(bool)
    interp_mask = mask_fill.reindex_like(filled).fillna(False).astype(bool)

    return filled, alerts, cap_mask, interp_mask
  
# =========================
# NEW: Enhanced Position Sizing (Fixes the 28.99% bug)
# =========================

def generate_td1_targets_asof(
    daily_close: pd.DataFrame,
    sectors_map: dict,
    preset: dict,
    asof: Optional[pd.Timestamp] = None,
    use_enhanced_features: bool = True,
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
    weights = _build_isa_weights_fixed(
        hist, preset, sectors_map, use_enhanced_features=use_enhanced_features
    )  # returns weights summing to equity exposure
    weights = weights / weights.sum() if weights.sum() > 0 else weights
    # normalize → enforce caps → final renorm before exposure scaling

    regime_metrics: Dict[str, float] = {}
    # If TD1 applies regime exposure (not the drawdown scaler on *returns*, but the weight scaler), apply it here too
    if use_enhanced_features and len(hist) > 0 and len(weights) > 0:
        try:
            regime_metrics = compute_regime_metrics(hist)
            regime_exposure = get_regime_adjusted_exposure(regime_metrics)
            weights = weights * float(regime_exposure)   # leaves implied cash = 1 - sum(weights)
        except Exception:
            regime_metrics = {}

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
    We *do not* force re-distribution; trimmed weight becomes cash (i.e., sum <= 1).
    This residual cash persists until any later exposure scaling step.
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
    idx_sector = pd.Index(ser_sector.values)
    ser_top = ser_sector.map(lambda s: s.split(":")[0])
    idx_top = pd.Index(ser_top.values)
    parent_labels = (
        [k for k in group_caps.keys() if ":" not in k]
        if group_caps
        else []
    )

    def _apply_name_caps(w: pd.Series) -> tuple[pd.Series, bool]:
        over = w[w > name_cap]
        if over.empty:
            return w, False
        w.loc[over.index] = name_cap
        return w, True

    def _apply_sector_caps(w: pd.Series) -> tuple[pd.Series, bool]:
        changed = False
        sums = w.groupby(idx_sector).sum()
        for sec, s in sums.items():
            cap = (group_caps.get(sec) if group_caps and sec in group_caps else sector_cap)
            if s > cap + 1e-12:
                f = cap / s
                mask = idx_sector == sec
                w.loc[mask] = w.loc[mask] * f
                changed = True
        return w, changed

    def _apply_parent_caps(w: pd.Series) -> tuple[pd.Series, bool]:
        if not parent_labels:
            return w, False
        changed = False
        sums = w.groupby(idx_top).sum()
        for parent in parent_labels:
            if parent in sums.index and sums[parent] > group_caps[parent] + 1e-12:
                f = group_caps[parent] / sums[parent]
                mask = idx_top == parent
                w.loc[mask] = w.loc[mask] * f
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
        "PLTR","SNOW","MDB","DDOG","AI"  # keep PLTR here
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
def apply_signal_decay(
    momentum_scores: pd.Series,
    signal_age_days: int | float | pd.Series | dict[str, int] = 0,
    half_life: int = 45,
) -> pd.Series:
    """Apply exponential decay to momentum signals based on age

    Parameters
    ----------
    momentum_scores : pd.Series
        Series of momentum scores indexed by ticker.
    signal_age_days : Union[int, pd.Series, Dict]
        Age of each signal in days.  Can be a scalar applied to all
        tickers or a Series/Dict providing ages per ticker.
    half_life : int, default 45
        Half-life in days for the exponential decay.
    """

    if isinstance(signal_age_days, (int, float)):
        if signal_age_days <= 0:
            return momentum_scores
        decay_factor = 0.5 ** (signal_age_days / half_life)
        return momentum_scores * decay_factor

    ages = pd.Series(signal_age_days).reindex(momentum_scores.index).fillna(0)
    if (ages <= 0).all():
        return momentum_scores

    decay_factors = 0.5 ** (ages / half_life)
    return momentum_scores * decay_factors

# =========================
# NEW: Regime-Adjusted Position Sizing
# =========================
def get_regime_adjusted_exposure(regime_metrics: Dict[str, float]) -> float:
    """Scale overall portfolio exposure based on market regime"""
    breadth = _metric_or_default(regime_metrics, 'breadth_pos_6m', 0.5)
    vol_regime = _metric_or_default(regime_metrics, 'qqq_vol_10d', 0.02)
    qqq_above_ma = _metric_or_default(regime_metrics, 'qqq_above_200dma', 1.0)
    vix_ts = _metric_or_default(regime_metrics, 'vix_term_structure', 1.0)
    hy_oas = _metric_or_default(regime_metrics, 'hy_oas', 4.0)
    
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

######################################################################
# Parameter mapping updater
######################################################################
def update_parameter_mapping(log: pd.DataFrame | None = None) -> None:
    """Recalibrate the mapping from metrics to parameter defaults.

    The function ingests historical assessment/outcome pairs and uses a
    simple grid-search (gradient-free optimisation) to determine
    volatility breakpoints and recommended caps.  Results are stored in
    ``st.session_state['param_map_defaults']``.

    Parameters
    ----------
    log : pd.DataFrame, optional
        Assessment log with at least the columns ``metrics``,
        ``portfolio_ret`` and ``benchmark_ret``.  If ``None`` the log is
        loaded via :func:`load_assess_log`.
    """

    if log is None:
        log = load_assess_log()

    required = {"metrics", "portfolio_ret", "benchmark_ret"}
    if log.empty or not required.issubset(log.columns):
        return

    # Extract volatility metric and compute portfolio alpha as outcome
    def _extract_vol(x: Any) -> float:
        try:
            return float(json.loads(x).get("qqq_vol_10d", np.nan))
        except Exception:
            return np.nan

    vols = log["metrics"].apply(_extract_vol).astype(float)
    outcomes = log["portfolio_ret"].astype(float) - log["benchmark_ret"].astype(float)
    mask = vols.notna() & outcomes.notna()
    vols, outcomes = vols[mask], outcomes[mask]

    if len(vols) < 5:
        # Not enough data to recalibrate
        return

    # Candidate thresholds for grid search
    candidates = np.linspace(vols.quantile(0.1), vols.quantile(0.9), 8)
    best_low = PARAM_MAP_DEFAULTS["low_vol"]
    best_high = PARAM_MAP_DEFAULTS["high_vol"]
    best_score = -np.inf

    for low in candidates:
        for high in candidates:
            if high <= low:
                continue
            low_mask = vols < low
            high_mask = vols > high
            if low_mask.sum() < 1 or high_mask.sum() < 1:
                continue
            score = outcomes[low_mask].mean() - outcomes[high_mask].mean()
            if score > best_score:
                best_score = score
                best_low, best_high = float(low), float(high)

    # Adjust caps proportional to regime separation strength
    delta = float(np.tanh(best_score))  # bounded adjustment factor
    name_cap_low = np.clip(PARAM_MAP_DEFAULTS["name_cap_low"] + 0.05 * delta, 0.15, 0.35)
    name_cap_high = np.clip(PARAM_MAP_DEFAULTS["name_cap_high"] - 0.05 * delta, 0.15, 0.35)
    sector_cap_low = np.clip(PARAM_MAP_DEFAULTS["sector_cap_low"] + 0.05 * delta, 0.20, 0.40)
    sector_cap_high = np.clip(PARAM_MAP_DEFAULTS["sector_cap_high"] - 0.05 * delta, 0.20, 0.40)
    top_n_low = int(np.clip(PARAM_MAP_DEFAULTS["top_n_low"] + 2 * delta, 5, 12))
    top_n_high = int(np.clip(PARAM_MAP_DEFAULTS["top_n_high"] - 2 * delta, 3, 10))

    st.session_state["param_map_defaults"] = {
        "low_vol": best_low,
        "high_vol": best_high,
        "top_n_low": top_n_low,
        "top_n_mid": PARAM_MAP_DEFAULTS["top_n_mid"],
        "top_n_high": top_n_high,
        "name_cap_low": float(name_cap_low),
        "name_cap_mid": PARAM_MAP_DEFAULTS["name_cap_mid"],
        "name_cap_high": float(name_cap_high),
        "sector_cap_low": float(sector_cap_low),
        "sector_cap_mid": PARAM_MAP_DEFAULTS["sector_cap_mid"],
        "sector_cap_high": float(sector_cap_high),
    }

# =========================
# NEW: Regime-Based Parameter Mapping
# =========================
def map_metrics_to_settings(metrics: Dict[str, float]) -> Dict[str, float]:
    """Map regime metrics to strategy parameter settings.

    This function also triggers a periodic update of the parameter
    mapping on the first day of each month using
    :func:`update_parameter_mapping`.

    Parameters
    ----------
    metrics : dict
        Dictionary containing regime metrics such as ``qqq_vol_10d``.

    Returns
    -------
    dict
        Dictionary with keys ``top_n``, ``name_cap``, and ``sector_cap``.
    """

    today = date.today()
    if today.day == 1 and st.session_state.get("_param_map_last_update") != today:
        try:
            update_parameter_mapping()
        except Exception as exc:  # pragma: no cover - best effort only
            st.warning(f"Parameter update failed: {exc}")
        st.session_state["_param_map_last_update"] = today

    mapping = st.session_state.get("param_map_defaults", PARAM_MAP_DEFAULTS)
    vol = metrics.get("qqq_vol_10d", np.nan)

    top_n = mapping["top_n_mid"]
    name_cap = mapping["name_cap_mid"]
    sector_cap = mapping["sector_cap_mid"]

    if pd.notna(vol):
        if vol > mapping["high_vol"]:  # High volatility -> be more defensive
            top_n = mapping["top_n_high"]
            name_cap = mapping["name_cap_high"]
            sector_cap = mapping["sector_cap_high"]
        elif vol < mapping["low_vol"]:  # Low volatility -> allow broader exposure
            top_n = mapping["top_n_low"]
            name_cap = mapping["name_cap_low"]
            sector_cap = mapping["sector_cap_low"]

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
def calculate_portfolio_correlation_to_market(
    portfolio_returns: pd.Series,
    market_returns: pd.Series = None,
) -> float:
    """Calculate correlation between portfolio and benchmark.

    """

    def _to_monthly_returns(r: pd.Series) -> pd.Series:
        """Convert an arbitrary-return series to monthly returns if possible.
        If index isn't datetime-like and can't be coerced, return as-is."""
        r = pd.Series(r).astype(float).dropna()
        if r.empty:
            return r

        idx = r.index
        # Coerce to DatetimeIndex when possible
        if not isinstance(idx, (pd.DatetimeIndex, pd.PeriodIndex)):
            try:
                coerced = pd.to_datetime(idx, errors="coerce")
                mask = ~coerced.isna()
                if mask.any():
                    r = r.loc[mask]
                    r.index = coerced[mask]
                else:
                    # Cannot coerce -> return without resampling
                    return r
            except Exception:
                return r

        # Normalize PeriodIndex to Timestamp for resampling
        if isinstance(r.index, pd.PeriodIndex):
            r.index = r.index.to_timestamp()

        # Now safe to resample monthly
        return (1.0 + r).groupby(pd.Grouper(freq="M")).prod() - 1.0

    # Portfolio series
    port = pd.Series(portfolio_returns).astype(float).dropna()
    if port.empty:
        return np.nan

    # Market series (fetch QQQ if not provided)
    if market_returns is None:
        try:
            end = datetime.now()
            start = end - relativedelta(months=9)
            qqq_px = get_benchmark_series("QQQ", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            mkt = qqq_px.pct_change().dropna()
        except Exception:
            return np.nan
    else:
        mkt = pd.Series(market_returns).astype(float).dropna()

    # Downsample to monthly when possible
    port_m = _to_monthly_returns(port)
    mkt_m  = _to_monthly_returns(mkt)

    if port_m.empty or mkt_m.empty:
        return np.nan

    # Align and require enough overlap
    common = port_m.index.intersection(mkt_m.index)
    if len(common) < 3:
        return np.nan

    corr = port_m.reindex(common).corr(mkt_m.reindex(common))
    return float(corr) if pd.notna(corr) else np.nan

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

    qqq_above = _metric_or_default(regime_metrics, "qqq_above_200dma", 1.0)
    breadth = _metric_or_default(regime_metrics, "breadth_pos_6m", 1.0)
    slope = _metric_or_default(regime_metrics, "qqq_50dma_slope_10d", 0.0)

    bearish = (
        qqq_above < 1.0 or
        breadth < 0.40 or
        slope < 0.0
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

MIN_COVERAGE_FRAC = 0.60  # keep ticker if >=60% non-null closes
MAX_FFILL_DAYS = 5        # max gap tolerance for forward/back fill
VOL_CLIP_Q = 0.995        # clip extreme volume spikes


def _ensure_unique_sorted_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with datetime index coerced, deduped (keep last), and sorted."""
    if df is None or not hasattr(df, "empty") or df.empty:
        return df

    out = df.copy()
    try:
        out.index = pd.to_datetime(out.index, errors="coerce")
    except Exception:
        return out.sort_index()

    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()


def _keep_if_sufficient_history(df: pd.DataFrame, min_days: int = 120) -> pd.DataFrame:
    """Soft gate that keeps columns with at least ``min_days`` observations."""
    if df is None or not hasattr(df, "empty") or df.empty:
        return df

    non_na = df.notna().sum(axis=0)
    keep_cols = non_na[non_na >= int(min_days)].index
    if len(keep_cols) == 0:
        # Fall back to top-N by coverage to avoid returning an empty frame
        keep_cols = non_na.sort_values(ascending=False).head(min(20, len(non_na))).index
    return df.loc[:, keep_cols]


def _normalize_ticker_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Upper-case, trim, and dedupe ticker columns (keep last)."""
    if df is None or not hasattr(df, "empty") or df.empty:
        return df

    out = df.copy()
    try:
        cols = [str(col).strip().upper() for col in out.columns]
        out.columns = pd.Index(cols)
    except Exception:
        return out

    return out.loc[:, ~out.columns.duplicated(keep="last")]


def _coverage_filter(close: pd.DataFrame, min_frac: float = MIN_COVERAGE_FRAC) -> pd.DataFrame:
    if close is None or close.empty:
        return close

    non_null = close.notna().sum(axis=0)
    thresh = int(np.ceil(min_frac * len(close))) if len(close) else 0
    keep = non_null[non_null >= thresh].index.tolist()
    return close.loc[:, keep]


def _log_stage(name: str, close: pd.DataFrame, vol: pd.DataFrame | None = None) -> None:
    close_cols = 0 if close is None or getattr(close, "empty", True) else close.shape[1]
    vol_cols = 0
    if vol is not None and hasattr(vol, "empty") and not vol.empty:
        vol_cols = vol.shape[1]
    rows = 0 if close is None or getattr(close, "empty", True) else len(close)
    logging.info("[pv:%s] close_cols=%s  vol_cols=%s  rows=%s", name, close_cols, vol_cols, rows)


def _safe_subset(prices: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    have = [t for t in tickers if t in prices.columns]
    missing = [t for t in tickers if t not in prices.columns]
    if not have:
        return prices.iloc[:, :0], missing
    return prices.loc[:, have], missing


def _soft_prune_history(
    df: pd.DataFrame,
    *,
    min_days: int = 180,
    max_missing_frac: float = 0.60,
) -> pd.DataFrame:
    """Leniently drop sparse tickers; keep top history coverage if all pruned."""
    if df is None or not hasattr(df, "empty") or df.empty:
        return df

    total = df.shape[0]
    valid = df.notna().sum()
    miss_frac = 1.0 - (valid / max(total, 1))
    keep = (valid >= min_days) & (miss_frac <= max_missing_frac)
    kept = df.loc[:, keep]

    if kept.empty and df.shape[1] > 0:
        top = valid.sort_values(ascending=False).head(min(50, len(valid))).index
        kept = df.loc[:, top]

    return kept

def _chunk_tickers(tickers: List[str], size: Optional[int] = None):
    chunk_size = int(size) if size else int(PERF["yf_batch"])
    if chunk_size <= 0:
        chunk_size = 1
    for i in range(0, len(tickers), chunk_size):
        yield tickers[i : i + chunk_size]

def _yf_download(tickers, **kwargs):
    params = dict(auto_adjust=True, progress=False, group_by="column", timeout=10)
    params.update(kwargs)
    try:
        return yf.download(tickers, **params)
    except TypeError:
        params.pop("group_by", None)
        params.pop("timeout", None)
        return yf.download(tickers, **params)

def parallel_yf_download(tickers, start, end, slices: int = 2):
    """Download Yahoo Finance data in parallel date slices and merge the result."""

    download_module = getattr(yf.download, "__module__", "")
    if download_module and not download_module.startswith("yfinance"):
        return _yf_download(tickers, start=start, end=end)

    try:
        slices = int(slices)
    except Exception:
        slices = 2

    try:
        if len(tickers) <= 1:
            slices = 1
    except TypeError:
        slices = max(1, slices)

    download_mod = getattr(getattr(yf, "download", None), "__module__", "")

    if (
        slices <= 1
        or Parallel is None
        or delayed is None
        or "yfinance" not in str(download_mod)
    ):
        return _yf_download(tickers, start=start, end=end)

    try:
        boundaries = pd.date_range(start=start, end=end, periods=slices + 1)
    except Exception:
        return _yf_download(tickers, start=start, end=end)

    pairs = list(zip(boundaries[:-1], boundaries[1:]))
    if not pairs:
        return _yf_download(tickers, start=start, end=end)

    n_jobs_cfg = PERF.get("n_jobs", 1) or 1
    try:
        n_jobs_cfg = int(n_jobs_cfg)
    except Exception:
        n_jobs_cfg = 1
    n_jobs = max(1, min(len(pairs), n_jobs_cfg))

    parts = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_yf_download)(tickers, start=s, end=e) for s, e in pairs
    )

    frames: List[pd.DataFrame] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, pd.Series):
            part = part.to_frame()
        frames.append(part)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames).sort_index()
    merged = merged.loc[~merged.index.duplicated(keep="last")]
    return merged.ffill()

# =========================
# Universe builders & sectors (Enhanced with validation)
# =========================
@st.cache_data(ttl=86400)
def fetch_sp500_constituents() -> List[str]:
    """Get current S&P 500 tickers with fallback to static list."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        tables = pd.read_html(StringIO(resp.text))
        df = next(
            t for t in tables
            if any(col.lower() in {"symbol", "ticker"} for col in map(str, t.columns))
        )
        col = "Symbol" if "Symbol" in df.columns else "Ticker"
        tickers = (
            df[col]
            .astype(str)
            .str.replace(".", "-", regex=False)
            .str.upper()
            .str.strip()
            .tolist()
        )
        return sorted(set(tickers))
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

_SECTOR_OVERRIDE_PATH = Path(__file__).with_name("sector_overrides.csv")

# Directory for parquet-based caching of downloaded price data
PARQUET_CACHE_DIR = Path(".parquet_cache")

def _load_sector_overrides() -> Dict[str, str]:
    if _SECTOR_OVERRIDE_PATH.exists():
        try:
            df = pd.read_csv(_SECTOR_OVERRIDE_PATH)
            df["Ticker"] = (
                df["Ticker"]
                .astype(str)
                .str.upper()
                .str.replace(".", "-", regex=False)
                .str.strip()
            )
            return {row["Ticker"]: str(row["Sector"]) for _, row in df.iterrows()}
        except Exception:
            return {}
    return {}

_SECTOR_OVERRIDES = _load_sector_overrides()

def _parquet_cache_path(prefix: str, tickers: List[str], start_date: str, end_date: str) -> Path:
    """Return a filesystem path for caching data as parquet.

    A short hash based on the prefix, tickers and date range is used so the
    filenames remain manageable regardless of the number of tickers provided.
    The cache directory is created on demand so tests can monkeypatch
    ``PARQUET_CACHE_DIR`` to a temporary location.
    """

    PARQUET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key_obj = {
        "p": prefix,
        "t": sorted(tickers),
        "s": start_date,
        "e": end_date,
    }
    key = json.dumps(key_obj, sort_keys=True)
    fname = hashlib.sha256(key.encode("utf-8")).hexdigest() + ".parquet"
    return PARQUET_CACHE_DIR / fname

def _read_cached_dataframe(path: Path) -> Optional[pd.DataFrame]:
    """Read a cached DataFrame with graceful fallback respecting ``PERF`` toggles."""

    if not path.exists():
        return None

    cache_days = int(PERF.get("cache_days", 0) or 0)
    if cache_days > 0:
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            if datetime.utcnow() - mtime > timedelta(days=cache_days):
                return None
        except Exception:
            return None

    if PERF.get("fast_io", True):
        if PERF.get("use_polars", False):
            try:
                import polars as pl  # type: ignore

                return pl.read_parquet(path).to_pandas()
            except Exception:
                pass
        try:
            return pd.read_parquet(path)
        except Exception:
            pass

    try:
        return pd.read_pickle(path)
    except Exception:
        return None

def _write_cached_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Persist ``df`` to ``path`` honouring ``PERF`` hints for the storage backend."""

    if PERF.get("fast_io", True):
        if PERF.get("use_polars", False):
            try:
                import polars as pl  # type: ignore

                pl.from_pandas(df).write_parquet(path)
                return
            except Exception:
                pass
        try:
            df.to_parquet(path)
            return
        except Exception:
            pass

    try:
        df.to_pickle(path)
    except Exception:
        pass

def _safe_get_info(ticker: yf.Ticker, timeout: float = 5.0) -> Dict[str, Any]:
    """Fetch ``ticker.info`` with a timeout and graceful fallback."""

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ticker.get_info)
        try:
            return future.result(timeout=timeout) or {}
        except TimeoutError:
            return {}
        except Exception:
            return {}

@st.cache_data(ttl=_SECTOR_CACHE_TTL)
def get_sector_map(tickers: List[str]) -> Dict[str, str]:
    """Return a mapping from ticker to sector name.

    Each ticker is fetched from :mod:`yfinance` at most once. Results are cached
    per ticker in ``_SECTOR_CACHE`` and the full mapping is cached by Streamlit
    for 24 hours. Local overrides from ``sector_overrides.csv`` take precedence,
    ``Ticker.fast_info`` is used when possible, and any remaining calls to
    ``ticker.info`` are wrapped with a timeout to avoid hangs. Tickers with
    missing sector information are stored as ``"Unknown"``.
    """

    tickers = list(tickers)
    unique_tickers = list(dict.fromkeys(tickers))

    # Ensure the synthetic hedge ticker is always mapped without hitting yfinance
    for alias in HEDGE_TICKER_ALIASES:
        if alias in unique_tickers and alias not in _SECTOR_CACHE:
            _SECTOR_CACHE[alias] = "Hedge Overlay"

    for t in unique_tickers:
        if t in _SECTOR_OVERRIDES:
            _SECTOR_CACHE[t] = _SECTOR_OVERRIDES[t]

    new_tickers = [
        t
        for t in unique_tickers
        if t not in _SECTOR_CACHE and t not in _SECTOR_OVERRIDES
    ]
    for t in new_tickers:
        sector = "Unknown"
        try:
            tkr = yf.Ticker(t)
            try:
                fast_info = getattr(tkr, "fast_info", {}) or {}
                sector = fast_info.get("sector")
            except Exception:
                sector = None
            if not sector:
                sector = _safe_get_info(tkr).get("sector")
            if not sector:
                sector = "Unknown"
        except Exception:
            sector = "Unknown"
        _SECTOR_CACHE[t] = sector

    return {
        t: _SECTOR_OVERRIDES.get(t, _SECTOR_CACHE.get(t, "Unknown"))
        for t in unique_tickers
    }

def get_nasdaq_100_plus_tickers() -> List[str]:
    """Get NASDAQ 100+ tickers with fallback list"""
    try:
        resp = requests.get(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=10,
        )
        tables = pd.read_html(StringIO(resp.text))
        df = next(
            t for t in tables
            if any(col.lower() in {"ticker", "symbol"} for col in map(str, t.columns))
        )
        col = "Ticker" if "Ticker" in df.columns else "Symbol"
        nasdaq_100 = (
            df[col].astype(str).str.upper().str.strip().tolist()
        )
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
    tickers = _sc_sanitize_tickers(base_tickers)
    if "QQQ" not in tickers:
        tickers.append("QQQ")
    base_sectors = {t: base_sectors.get(t, "Unknown") for t in tickers if t != "QQQ"}
    if not base_tickers:
        return pd.DataFrame(), pd.DataFrame(), {}, label

    close, vol = fetch_price_volume(tickers, start_date, end_date)
    if close.empty or "QQQ" not in close.columns:
        return pd.DataFrame(), pd.DataFrame(), {}, label

    # Enhanced data validation is now built into fetch_price_volume
    cols = close.columns.intersection(tickers)
    if len(cols) == 0:
        cols = close.columns
    close = close.loc[:, cols]
    if isinstance(vol, pd.DataFrame) and not vol.empty:
        v_cols = vol.columns.intersection(tickers)
        if len(v_cols) == 0:
            v_cols = vol.columns
        vol = vol.loc[:, v_cols]
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
from typing import List, Tuple

def _resolve_fetch_start(start_date: str, end_date: Optional[str]) -> str:
    """Return the actual start date to use when downloading data."""
    months_back = max(14, 6 if PERF.get("fast_io") else 14)

    try:
        start_ts = pd.to_datetime(start_date)
    except Exception:
        return start_date

    extend = False
    if end_date:
        try:
            end_ts = pd.to_datetime(end_date)
            if pd.notna(end_ts):
                extend = (end_ts - start_ts).days >= REGIME_MA
        except Exception:
            extend = False

    if extend:
        start_ts = start_ts - pd.DateOffset(months=months_back)

    return start_ts.strftime("%Y-%m-%d")

@st.cache_data(ttl=43200)
def fetch_market_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Download adj-close prices for tickers, validate/clean, and cache to disk & Streamlit."""
    safe = _sc_sanitize_tickers(tickers)
    if "QQQ" not in safe:
        safe.append("QQQ")

    cache_path = _parquet_cache_path("market", safe, start_date, end_date)

    def _maybe_add_qqq(df: pd.DataFrame) -> pd.DataFrame:
        if "QQQ" in df.columns:
            return df
        try:
            qqq_df = strategy_core.fetch_market_data(["QQQ"], start_date, end_date)
            if not qqq_df.empty and "QQQ" in qqq_df.columns:
                df = pd.concat([df, qqq_df[["QQQ"]]], axis=1)
        except Exception:
            pass
        return df

    cached = _read_cached_dataframe(cache_path)
    if cached is not None:
        cached = _ensure_unique_sorted_index(cached)
        cols = cached.columns.intersection(safe)
        if len(cols) == 0:
            cols = cached.columns
        cached = cached.loc[:, cols]
        cached = cached.sort_index().ffill(limit=5)
        universe_size = cached.shape[1]
        min_days = 90 if universe_size <= 200 else 150
        cached = _keep_if_sufficient_history(cached, min_days=min_days)
        return _maybe_add_qqq(cached)

    try:
        fetch_start = _resolve_fetch_start(start_date, end_date)

        frames: List[pd.DataFrame] = []
        for batch in _chunk_tickers(safe):
            try:
                raw = parallel_yf_download(batch, start=fetch_start, end=end_date)
            except Exception:
                raw = _yf_download(batch, start=fetch_start, end=end_date)

            df = raw["Close"]
            if isinstance(df, pd.Series):
                df = df.to_frame()
                df.columns = [batch[0]]

            df = _ensure_unique_sorted_index(df)
            frames.append(df)

        if not frames:
            return pd.DataFrame()
            
        data = pd.concat(frames, axis=1)
        data = _ensure_unique_sorted_index(data)
        result = data.dropna(axis=1, how="all")
        cols = result.columns.intersection(safe)
        if len(cols) == 0:
            cols = result.columns
        result = result.loc[:, cols]

        if not result.empty:
            cleaned_result, cleaning_alerts, _, _ = validate_and_clean_market_data(
                result, info=logging.info
            )
            cleaned_result = _ensure_unique_sorted_index(cleaned_result)
            cols = cleaned_result.columns.intersection(safe)
            if len(cols) == 0:
                cols = cleaned_result.columns
            cleaned_result = cleaned_result.loc[:, cols]

            universe_size = cleaned_result.shape[1]
            min_days = 90 if universe_size <= 200 else 150
            cleaned_result = _keep_if_sufficient_history(cleaned_result, min_days=min_days)
            cleaned_result = cleaned_result.sort_index().ffill(limit=5)

            dropped = set(result.columns) - set(cleaned_result.columns)
            if dropped:
                logging.info(
                    "Pruned %d tickers for insufficient history (min_days=%d). Examples: %s",
                    len(dropped),
                    min_days,
                    ", ".join(list(dropped)[:10]),
                )

            if cleaning_alerts:
                for alert in cleaning_alerts[:2]:
                    logging.info("Data cleaning: %s", alert)

            _write_cached_dataframe(cleaned_result, cache_path)
            return _maybe_add_qqq(cleaned_result)

        _write_cached_dataframe(result, cache_path)
        return _maybe_add_qqq(result)

    except Exception as e:
        st.error(f"Failed to download market data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=43200)
def fetch_price_volume(tickers: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download Close & Volume with gentle pruning and robust alignment."""

    safe = _sc_sanitize_tickers(tickers)
    if "QQQ" not in safe:
        safe.append("QQQ")

    cache_path = _parquet_cache_path("price_volume", safe, start_date, end_date)

    def _maybe_add_qqq(close_df: pd.DataFrame, vol_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        close_out = _normalize_ticker_columns(_ensure_unique_sorted_index(close_df))
        vol_out = _normalize_ticker_columns(_ensure_unique_sorted_index(vol_df)) if vol_df is not None else vol_df

        if "QQQ" not in getattr(close_out, "columns", []):
            try:
                qqq_close = strategy_core.fetch_market_data(["QQQ"], start_date, end_date)
                qqq_close = _normalize_ticker_columns(_ensure_unique_sorted_index(qqq_close))
                if isinstance(qqq_close, pd.DataFrame) and not qqq_close.empty and "QQQ" in qqq_close.columns:
                    close_out = close_out.join(qqq_close[["QQQ"]], how="left")
            except Exception:
                logging.exception("Could not add QQQ close")

        if vol_out is not None and "QQQ" not in getattr(vol_out, "columns", []):
            try:
                raw = yf.download(
                    "QQQ",
                    start=start_date,
                    end=end_date,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                if isinstance(raw, pd.DataFrame) and "Volume" in raw.columns:
                    qqq_vol = raw["Volume"].rename("QQQ").to_frame()
                    qqq_vol = _normalize_ticker_columns(_ensure_unique_sorted_index(qqq_vol))
                    vol_out = vol_out.join(qqq_vol, how="left")
            except Exception:
                logging.exception("Could not add QQQ volume")

        close_out = _normalize_ticker_columns(_ensure_unique_sorted_index(close_out))
        if vol_out is not None:
            vol_out = _normalize_ticker_columns(_ensure_unique_sorted_index(vol_out))
        return close_out, vol_out if vol_out is not None else pd.DataFrame()

    combined = _read_cached_dataframe(cache_path)
    if combined is not None:
        try:
            combined = _ensure_unique_sorted_index(combined)
            if isinstance(combined.columns, pd.MultiIndex):
                levels = set(combined.columns.get_level_values(0))
                if {"Close", "Volume"}.issubset(levels):
                    close = _normalize_ticker_columns(_ensure_unique_sorted_index(combined["Close"]))
                    vol = _normalize_ticker_columns(_ensure_unique_sorted_index(combined["Volume"]))
                    c_cols = close.columns.intersection(safe)
                    if len(c_cols) == 0:
                        c_cols = close.columns
                    v_cols = vol.columns.intersection(safe)
                    if len(v_cols) == 0:
                        v_cols = vol.columns
                    close = close.loc[:, c_cols]
                    close = close.sort_index().ffill(limit=5)
                    universe_size = close.shape[1]
                    min_days = 90 if universe_size <= 200 else 150
                    close = _keep_if_sufficient_history(close, min_days=min_days)
                    vol = vol.loc[:, v_cols]
                    vol = vol.reindex_like(close).fillna(0)
                    for col in vol.columns:
                        s = vol[col]
                        if s.notna().any():
                            vol[col] = s.clip(lower=0, upper=s.quantile(0.99))
                    close, vol = _maybe_add_qqq(close, vol)
                    vol = vol.reindex_like(close).fillna(0)
                    _log_stage("cache-return", close, vol)
                    return close, vol
            else:
                flat = set(map(str, combined.columns))
                if {"Close", "Volume"}.issubset(flat):
                    close = combined["Close"]
                    vol = combined["Volume"]
                    if isinstance(close, pd.Series):
                        close = close.to_frame()
                    if isinstance(vol, pd.Series):
                        vol = vol.to_frame()
                    close = _normalize_ticker_columns(_ensure_unique_sorted_index(close))
                    vol = _normalize_ticker_columns(_ensure_unique_sorted_index(vol))
                    c_cols = close.columns.intersection(safe)
                    if len(c_cols) == 0:
                        c_cols = close.columns
                    v_cols = vol.columns.intersection(safe)
                    if len(v_cols) == 0:
                        v_cols = vol.columns
                    close = close.loc[:, c_cols]
                    close = close.sort_index().ffill(limit=5)
                    universe_size = close.shape[1]
                    min_days = 90 if universe_size <= 200 else 150
                    close = _keep_if_sufficient_history(close, min_days=min_days)
                    vol = vol.loc[:, v_cols]
                    vol = vol.reindex_like(close).fillna(0)
                    for col in vol.columns:
                        s = vol[col]
                        if s.notna().any():
                            vol[col] = s.clip(lower=0, upper=s.quantile(0.99))
                    close, vol = _maybe_add_qqq(close, vol)
                    vol = vol.reindex_like(close).fillna(0)
                    _log_stage("cache-return", close, vol)
                    return close, vol
        except Exception:
            logging.exception("Cached read failed; will refetch")

    try:
        fetch_start = _resolve_fetch_start(start_date, end_date)

        close_parts: list[pd.DataFrame] = []
        vol_parts: list[pd.DataFrame] = []

        for batch in _chunk_tickers(safe):
            try:
                raw = parallel_yf_download(batch, start=fetch_start, end=end_date)
            except Exception:
                raw = _yf_download(batch, start=fetch_start, end=end_date)

            if not isinstance(raw, (pd.DataFrame, pd.Series)):
                continue
            if isinstance(raw, pd.Series):
                raw = raw.to_frame()

            if isinstance(raw.columns, pd.MultiIndex):
                fields = set(raw.columns.get_level_values(0))
                if not {"Close", "Volume"}.issubset(fields):
                    continue
                close_part = raw["Close"]
                vol_part = raw["Volume"]
            else:
                if "Close" not in raw.columns or "Volume" not in raw.columns:
                    continue
                close_part = raw[["Close"]].copy()
                vol_part = raw[["Volume"]].copy()
                if len(batch) == 1:
                    close_part.columns = [batch[0]]
                    vol_part.columns = [batch[0]]

            close_part = _normalize_ticker_columns(_ensure_unique_sorted_index(close_part))
            vol_part = _normalize_ticker_columns(_ensure_unique_sorted_index(vol_part))
            close_parts.append(close_part)
            vol_parts.append(vol_part)

        if not close_parts:
            logging.error("No close parts downloaded.")
            return pd.DataFrame(), pd.DataFrame()

        close = _normalize_ticker_columns(
            _ensure_unique_sorted_index(pd.concat(close_parts, axis=1).dropna(axis=1, how="all"))
        )
        vol = _normalize_ticker_columns(
            _ensure_unique_sorted_index(pd.concat(vol_parts, axis=1))
        )

        vol = vol.reindex_like(close).fillna(0)

        c_cols = close.columns.intersection(safe)
        v_cols = vol.columns.intersection(safe)
        if len(c_cols) == 0:
            c_cols = close.columns
        if len(v_cols) == 0:
            v_cols = vol.columns
        close = close.loc[:, c_cols]
        vol = vol.loc[:, v_cols]

        _log_stage("raw-combined", close, vol)

        try:
            cleaned_close, close_alerts, _, _ = validate_and_clean_market_data(close, info=logging.info)
            cleaned_close = _normalize_ticker_columns(_ensure_unique_sorted_index(cleaned_close))
        except Exception:
            logging.exception("validate_and_clean_market_data failed; using pre-cleaned close")
            cleaned_close = close
            close_alerts = []

        cleaned_close = cleaned_close.sort_index().ffill(limit=5)
        universe_size = cleaned_close.shape[1]
        min_days = 90 if universe_size <= 200 else 150
        cleaned_close = _keep_if_sufficient_history(cleaned_close, min_days=min_days)

        dropped = set(close.columns) - set(cleaned_close.columns)
        if dropped:
            logging.info(
                "Pruned %d tickers for insufficient history (min_days=%d). Examples: %s",
                len(dropped),
                min_days,
                ", ".join(list(dropped)[:10]),
            )
        if close_alerts:
            for alert in close_alerts[:3]:
                logging.info("Price/Volume cleaning: %s", alert)

        vol_aligned = vol.reindex_like(cleaned_close).fillna(0)
        for col in vol_aligned.columns:
            s = vol_aligned[col]
            if s.notna().any():
                vol_aligned[col] = s.clip(lower=0, upper=s.quantile(0.99))

        if cleaned_close.shape[1] < 8:
            logging.warning(
                "Few tickers survived (%d). Relaxing history gate to min_days=60.",
                cleaned_close.shape[1],
            )
            cleaned_close = _keep_if_sufficient_history(cleaned_close, min_days=60)
            vol_aligned = vol.reindex_like(cleaned_close).fillna(0)

        cleaned_close, vol_aligned = _maybe_add_qqq(cleaned_close, vol_aligned)
        vol_aligned = vol_aligned.reindex_like(cleaned_close).fillna(0)

        combined_out = _ensure_unique_sorted_index(
            pd.concat({"Close": cleaned_close, "Volume": vol_aligned}, axis=1)
        )
        _write_cached_dataframe(combined_out, cache_path)

        _log_stage("final-return", cleaned_close, vol_aligned)
        return cleaned_close, vol_aligned

    except Exception as e:
        logging.exception("fetch_price_volume failed")
        st.error(f"Failed to download price/volume: {e}")
        return pd.DataFrame(), pd.DataFrame()


# =========================
# Persistence (Gist + Local) - Unchanged
# =========================
def save_portfolio_to_gist(portfolio_df: pd.DataFrame) -> None:
    if not GIST_API_URL or not GITHUB_TOKEN:
        if _HAS_ST:
            st.sidebar.warning("Gist secrets not configured; skipping Gist save.")
        else:
            logging.warning("Gist secrets not configured; skipping Gist save.")
        return
    try:
        json_content = portfolio_df.to_json(orient="index")
        payload = {"files": {GIST_PORTF_FILE: {"content": json_content}}}
        resp = requests.patch(GIST_API_URL, headers=HEADERS, json=payload, timeout=10)
        resp.raise_for_status()
        if _HAS_ST:
            st.sidebar.success("✅ Successfully saved portfolio to Gist.")
        else:
            logging.info("Successfully saved portfolio to Gist.")
    except Exception as e:
        if _HAS_ST:
            st.sidebar.error(f"Gist save failed: {e}")
        else:
            logging.error("Gist save failed: %s", e)

def load_previous_portfolio() -> Optional[pd.DataFrame]:
    def _process(df: pd.DataFrame, source: str) -> Optional[pd.DataFrame]:
        """Normalize weights and check constraints. Return None if invalid."""
        if "Weight" not in df.columns:
            if _HAS_ST:
                st.sidebar.warning(f"Discarded {source} portfolio: missing 'Weight' column")
            else:
                logging.warning("Discarded %s portfolio: missing 'Weight' column", source)
            return None

        try:
            weights = pd.to_numeric(df["Weight"], errors="coerce")
        except Exception:
            if _HAS_ST:
                st.sidebar.warning(f"Discarded {source} portfolio: non-numeric weights")
            else:
                logging.warning("Discarded %s portfolio: non-numeric weights", source)
            return None

        if weights.isna().any():
            if _HAS_ST:
                st.sidebar.warning(f"Discarded {source} portfolio: non-numeric weights")
            else:
                logging.warning("Discarded %s portfolio: non-numeric weights", source)
            return None

        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            if _HAS_ST:
                st.sidebar.warning(f"Discarded {source} portfolio: normalization failed")
            else:
                logging.warning("Discarded %s portfolio: normalization failed", source)
            return None

        weights = weights / total
        df = df.copy()
        df["Weight"] = weights

        # Constraint check if sector map available
        try:
            base_map = get_sector_map(list(df.index))
            enhanced_map = get_enhanced_sector_map(list(df.index), base_map=base_map)
            if enhanced_map:
                preset = STRATEGY_PRESETS.get("ISA Dynamic (0.75)", {})
                name_cap = float(preset.get("mom_cap", 0.25))
                sector_cap = float(preset.get("sector_cap", 0.30))
                group_caps = build_group_caps(enhanced_map)
                violations = check_constraint_violations(
                    weights,
                    enhanced_map,
                    name_cap,
                    sector_cap,
                    group_caps=group_caps,
                )
                if violations:
                    msg = f"Discarded {source} portfolio: constraint violations {violations}"
                    if _HAS_ST:
                        st.sidebar.warning(msg)
                    else:
                        logging.warning(msg)
                    return None
        except Exception as e:
            # If anything goes wrong, just proceed without constraint check
            logging.warning("P01 constraint check failed", exc_info=True)

        return df

    # Gist first
    if GIST_API_URL and GITHUB_TOKEN:
        try:
            resp = requests.get(GIST_API_URL, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            files = resp.json().get("files", {})
            content = files.get(GIST_PORTF_FILE, {}).get("content", "")
            if content and content != "{}":
                df = pd.read_json(io.StringIO(content), orient="index")
                processed = _process(df, "Gist")
                if processed is not None:
                    return processed
        except Exception as e:
            logging.warning("P02 failed to fetch portfolio from gist", exc_info=True)

    # Local fallback
    if os.path.exists(LOCAL_PORTF_FILE):
        try:
            df = pd.read_csv(LOCAL_PORTF_FILE)
            if "Weight" in df.columns and "Ticker" in df.columns:
                df = df.set_index("Ticker")
                return _process(df, "local")
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
        if _HAS_ST:
            st.sidebar.warning(f"Could not save local portfolio: {e}")
        else:
            logging.warning("Could not save local portfolio: %s", e)

def save_portfolio_if_rebalance(
    df: pd.DataFrame, price_index: Optional[pd.DatetimeIndex]
) -> bool:
    """Save portfolio only on rebalance days.

    Returns True if the save routines executed, otherwise False.
    """
    if not is_rebalance_today(date.today(), price_index):
        next_window = None
        if price_index is not None and len(price_index) > 0:
            idx = pd.to_datetime(price_index).normalize()
            latest = idx.max()
            current_window = first_trading_day(latest, idx)
            if latest <= current_window:
                next_window = current_window
            else:
                next_month = pd.Timestamp(latest) + pd.offsets.MonthBegin(1)
                next_window = first_trading_day(next_month, None)
        if next_window is not None:
            msg = f"Not a rebalance day – next window opens {next_window.date()}"
            if _HAS_ST:
                st.sidebar.info(msg)
            else:
                logging.info(msg)
        else:
            if _HAS_ST:
                st.sidebar.info("Not a rebalance day – skipping save")
            else:
                logging.info("Not a rebalance day – skipping save")
        return False

    # Proceed with standard save routines
    save_current_portfolio(df)
    save_portfolio_to_gist(df)
    return True

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
    if price_index is None or len(price_index) == 0:
        return False

    idx = pd.to_datetime(price_index).normalize()
    ts = pd.Timestamp(today).normalize()

    if ts in idx:
        reference = ts
        ftd = first_trading_day(ts, idx)
        return reference == ftd

    latest = idx.max()
    if ts.year != latest.year or ts.month != latest.month:
        ftd = first_trading_day(ts, idx)
        return ts == ftd

    ftd = first_trading_day(latest, idx)
    return latest == ftd

# =========================
# Math utils & KPIs (Enhanced)
# =========================
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
      - Turnover uses 0.5 × L1 weight change per rebalance
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
def _compute_signal_panels_polars(
    daily: pd.DataFrame,
) -> Dict[str, pd.DataFrame | pd.Series]:
    daily = daily.dropna(how="all", axis=1)
    if daily.shape[1] == 0:
        empty_df = pd.DataFrame(index=daily.index)
        return {
            "monthly": empty_df,
            "r3": empty_df,
            "r6": empty_df,
            "r12": empty_df,
            "vol63": empty_df,
            "ma200": empty_df,
            "dist200": empty_df,
        }

    pl_daily, value_cols = _prepare_polars_daily_frame(daily)

    if not value_cols:
        empty_df = pd.DataFrame(index=daily.index)
        return {
            "monthly": empty_df,
            "r3": empty_df,
            "r6": empty_df,
            "r12": empty_df,
            "vol63": empty_df,
            "ma200": empty_df,
            "dist200": empty_df,
        }

    monthly_pl = (
        pl_daily.groupby_dynamic(
            "date", every="1mo", period="1mo", label="right", closed="right"
        )
        .agg([pl.col(col).last().alias(col) for col in value_cols])
        .sort("date")
    )

    def _shift_returns(period: int) -> "pl.DataFrame":
        return pl_daily.select(
            [pl.col("date")]
            + [
                ((pl.col(col) / pl.col(col).shift(period)) - 1).alias(col)
                for col in value_cols
            ]
        )

    r3_pl = _shift_returns(63)
    r6_pl = _shift_returns(126)
    r12_pl = _shift_returns(252)

    daily_returns_pl = pl_daily.select(
        [pl.col("date")]
        + [((pl.col(col) / pl.col(col).shift(1)) - 1).alias(col) for col in value_cols]
    )
    vol63_pl = daily_returns_pl.select(
        [pl.col("date")]
        + [pl.col(col).rolling_std(window_size=63).alias(col) for col in value_cols]
    )
    ma200_pl = pl_daily.select(
        [pl.col("date")]
        + [pl.col(col).rolling_mean(window_size=200).alias(col) for col in value_cols]
    )
    dist200_pl = pl_daily.select(
        [pl.col("date")]
        + [
            ((pl.col(col) / pl.col(col).rolling_mean(window_size=200)) - 1).alias(col)
            for col in value_cols
        ]
    )

    monthly = _polars_to_pandas_indexed(monthly_pl)
    r3 = _polars_to_pandas_indexed(r3_pl)
    r6 = _polars_to_pandas_indexed(r6_pl)
    r12 = _polars_to_pandas_indexed(r12_pl)
    vol63 = _polars_to_pandas_indexed(vol63_pl)
    ma200 = _polars_to_pandas_indexed(ma200_pl)
    dist200 = _polars_to_pandas_indexed(dist200_pl).replace([np.inf, -np.inf], np.nan)

    return {
        "monthly": monthly,
        "r3": r3,
        "r6": r6,
        "r12": r12,
        "vol63": vol63,
        "ma200": ma200,
        "dist200": dist200,
    }

@st.cache_data(ttl=43200)
def compute_signal_panels(daily: pd.DataFrame) -> Dict[str, pd.DataFrame | pd.Series]:
    """Cache common signal inputs derived from daily close data."""
    if daily is None or daily.empty:
        empty_df = pd.DataFrame()
        return {
            "monthly": empty_df,
            "r3": empty_df,
            "r6": empty_df,
            "r12": empty_df,
            "vol63": empty_df,
            "ma200": empty_df,
            "dist200": empty_df,
        }

    daily = daily.dropna(how="all", axis=1)
    if daily.shape[1] == 0:
        empty_df = pd.DataFrame()
        return {
            "monthly": empty_df,
            "r3": empty_df,
            "r6": empty_df,
            "r12": empty_df,
            "vol63": empty_df,
            "ma200": empty_df,
            "dist200": empty_df,
        }

    daily = _ensure_unique_sorted_index(daily)

    # If your codebase has a Polars engine switch, respect it; otherwise fallback to pandas.
    if "_use_polars_engine" in globals() and callable(globals()["_use_polars_engine"]) and _use_polars_engine():
        try:
            return _compute_signal_panels_polars(daily)
        except Exception:
            logging.exception("Falling back to pandas signal panel computation")

    # --- pandas fallback ---
    daily_sorted = daily.sort_index()
    monthly = daily_sorted.resample("M").last()
    r3 = daily_sorted.pct_change(63)
    r6 = daily_sorted.pct_change(126)
    r12 = daily_sorted.pct_change(252)
    daily_returns = daily_sorted.pct_change()
    vol63 = daily_returns.rolling(63).std()
    ma200 = daily_sorted.rolling(200).mean()
    dist200 = (daily_sorted / ma200 - 1).replace([np.inf, -np.inf], np.nan)

    return {
        "monthly": monthly,
        "r3": r3,
        "r6": r6,
        "r12": r12,
        "vol63": vol63,
        "ma200": ma200,
        "dist200": dist200,
    }


# ---------- Factor z-score helpers (Pandas-first, optional Polars input) ----------

def _to_pandas(obj):
    """Convert Polars DataFrame/Series to pandas if needed; otherwise return as-is."""
    if "_HAS_POLARS" in globals() and _HAS_POLARS:
        try:
            if isinstance(obj, pl.DataFrame):
                return obj.to_pandas()
            if isinstance(obj, pl.Series):
                return obj.to_pandas()
        except Exception:
            # Fall through to return original object
            pass
    return obj


def blended_momentum_z(monthly: pd.DataFrame | "pl.DataFrame") -> pd.Series:
    """
    Blended momentum z-score using 3/6/12M horizons with weights 0.2/0.4/0.4.
    Accepts pandas or polars DataFrame (prices at month end).
    """
    if monthly is None:
        return pd.Series(dtype=float)

    monthly = _to_pandas(monthly)
    if not isinstance(monthly, pd.DataFrame) or monthly.shape[0] < 13:
        return pd.Series(dtype=float)

    # Ensure chronological order
    try:
        monthly = monthly.sort_index()
    except Exception:
        pass

    cols = monthly.columns
    r3 = monthly.pct_change(3).iloc[-1]
    r6 = monthly.pct_change(6).iloc[-1]
    r12 = monthly.pct_change(12).iloc[-1]

    z3 = _zscore_series(r3, cols)
    z6 = _zscore_series(r6, cols)
    z12 = _zscore_series(r12, cols)

    return 0.2 * z3 + 0.4 * z6 + 0.4 * z12


def lowvol_z(
    daily: pd.DataFrame | "pl.DataFrame",
    vol_series: pd.Series | "pl.Series" | None = None,
) -> pd.Series:
    """
    Low-volatility factor as a z-score of 63-day rolling std of daily returns.
    If 'vol_series' is given (precomputed vol), z-score that instead.
    Accepts pandas or polars inputs.
    """
    if daily is None:
        return pd.Series(dtype=float)

    daily = _to_pandas(daily)
    vol_series = _to_pandas(vol_series)

    if not isinstance(daily, pd.DataFrame):
        return pd.Series(dtype=float)
    cols = daily.columns

    # Use precomputed vol if provided
    if vol_series is not None:
        try:
            return _zscore_series(pd.Series(vol_series), cols)
        except Exception:
            pass

    if daily.shape[0] < 80:  # need enough data for a stable 63d window
        return pd.Series(0.0, index=cols)

    vol = (
        daily.pct_change()
        .rolling(63)
        .std()
        .iloc[-1]
        .replace([np.inf, -np.inf], np.nan)
    )
    return _zscore_series(vol, cols)


def trend_z(
    daily: pd.DataFrame | "pl.DataFrame",
    dist_series: pd.Series | "pl.Series" | None = None,
) -> pd.Series:
    """
    Trend factor as z-score of distance from 200DMA: (last / MA200 - 1).
    If 'dist_series' is given, z-score that instead.
    Accepts pandas or polars inputs.
    """
    if daily is None:
        return pd.Series(dtype=float)

    daily = _to_pandas(daily)
    dist_series = _to_pandas(dist_series)

    if not isinstance(daily, pd.DataFrame):
        return pd.Series(dtype=float)
    cols = daily.columns

    # Use precomputed distance if provided
    if dist_series is not None:
        try:
            return _zscore_series(pd.Series(dist_series), cols)
        except Exception:
            pass

    if daily.shape[0] < 220:  # ensure we have >=200d + buffer
        return pd.Series(0.0, index=cols)

    ma200 = daily.rolling(200).mean().iloc[-1]
    last = daily.iloc[-1]
    dist = (last / ma200 - 1.0).replace([np.inf, -np.inf], np.nan)
    return _zscore_series(dist, cols)
  
def composite_score(daily: pd.DataFrame, panels: Dict[str, pd.DataFrame | pd.Series] | None = None) -> pd.Series:
    panels = compute_signal_panels(daily) if panels is None else panels
    monthly = panels.get("monthly", pd.DataFrame())
    momz = blended_momentum_z(monthly)

    vol_panel = panels.get("vol63")
    vol_series = None
    if isinstance(vol_panel, pd.DataFrame) and not vol_panel.empty:
        vol_series = vol_panel.iloc[-1]
    elif isinstance(vol_panel, pd.Series):
        vol_series = vol_panel
    lvz  = lowvol_z(daily, vol_series=vol_series)

    dist_panel = panels.get("dist200")
    dist_series = None
    if isinstance(dist_panel, pd.DataFrame) and not dist_panel.empty:
        dist_series = dist_panel.iloc[-1]
    elif isinstance(dist_panel, pd.Series):
        dist_series = dist_panel
    tz   = trend_z(daily, dist_series=dist_series)
    return (0.6*momz.add(0.0, fill_value=0.0) + 0.2*(-lvz) + 0.2*tz).dropna()

def momentum_stable_names(
    daily: pd.DataFrame,
    top_n: int,
    days: int,
    panels: Dict[str, pd.DataFrame | pd.Series] | None = None,
) -> List[str]:
    if daily.shape[0] < (days + 260): return []

    panels = compute_signal_panels(daily) if panels is None else panels
    r3 = panels.get("r3", pd.DataFrame())
    r6 = panels.get("r6", pd.DataFrame())
    r12 = panels.get("r12", pd.DataFrame())

    if any(not isinstance(df, pd.DataFrame) or df.empty for df in (r3, r6, r12)):
        return []

    def zrow(df: pd.DataFrame) -> pd.DataFrame:
        mu = df.mean(axis=1)
        sd = df.std(axis=1).replace(0, np.nan)
        return (df.sub(mu, axis=0)).div(sd, axis=0).fillna(0.0)

    mscore = 0.2*zrow(r3) + 0.4*zrow(r6) + 0.4*zrow(r12)
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

    monthly = daily.resample("M").last()
    fwd = monthly.pct_change().shift(-1)  # next-month returns
    rets = pd.Series(index=monthly.index, dtype=float)
    tno  = pd.Series(index=monthly.index, dtype=float)
    prev_w = pd.Series(dtype=float)
    signal_start_dates: Dict[str, pd.Timestamp] = {}

    for m in monthly.index:
        hist = daily.loc[:m]

        # Composite momentum score -> 1-D vector for the current month
        comp_all = composite_score(hist)
        if isinstance(comp_all, pd.DataFrame):
            comp = comp_all.iloc[-1].dropna()
        else:
            comp = pd.Series(comp_all).dropna()
        comp_full = comp.copy()

        if comp.empty:
            rets.loc[m] = 0.0
            tno.loc[m]  = 0.0
            prev_w = pd.Series(dtype=float)
            continue

        # Restrict to momentum names using adaptive cutoff and fallbacks
        momz = blended_momentum_z(hist.resample("M").last())
        base_comp = comp_full
        if isinstance(momz, pd.Series) and not momz.empty:
            comp_aligned = comp_full.reindex(momz.index).dropna()
            aligned_momz = momz.reindex(comp_aligned.index)
            valid_momz = aligned_momz.dropna()
            cutoff = 0.0
            if not valid_momz.empty:
                median_val = float(valid_momz.median())
                cutoff = max(median_val, 0.0) if np.isfinite(median_val) else 0.0
            sel = comp_aligned[aligned_momz > cutoff].dropna()
            base_comp = comp_aligned if not comp_aligned.empty else comp_full
        else:
            sel = base_comp.copy()

        if sel.empty:
            sel = base_comp.dropna()

        if sel.empty:
            sel = base_comp.nlargest(top_n)

        if sel.empty:
            rets.loc[m] = 0.0
            tno.loc[m]  = 0.0
            prev_w = pd.Series(dtype=float)
            continue

        # Pick top-N by score
        picks = sel.nlargest(top_n)

        # Stickiness filter (prefer persistent names)
        stable = set(momentum_stable_names(hist, top_n=top_n, days=stickiness_days))
        if stable:
            filtered = picks.reindex([t for t in picks.index if t in stable]).dropna()
            if not filtered.empty:
                picks = filtered
            else:
                picks = sel.nlargest(top_n)  # fallback if stickiness empties set

        if picks.empty:
            picks = base_comp.nlargest(top_n)

        # Optional: signal decay shaping
        if use_enhanced_features:
            signal_ages: Dict[str, int] = {}
            for t in picks.index:
                if t not in signal_start_dates:
                    signal_start_dates[t] = m
                    signal_ages[t] = 0
                else:
                    signal_ages[t] = (m - signal_start_dates[t]).days

            # Remove tickers no longer selected to reset their age if re-enter later
            for t in list(signal_start_dates.keys()):
                if t not in picks.index:
                    del signal_start_dates[t]

            picks = apply_signal_decay(picks, pd.Series(signal_ages))

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

        w = w / w.sum() if w.sum() > 0 else w
        # normalize → enforce caps → final renorm before exposure scaling

        # Regime-based exposure scaling (keeps exposure < 1 when risk-off)
        if use_enhanced_features and len(hist) > 0 and len(w) > 0:
            try:
                regime_metrics  = compute_regime_metrics(hist)
                regime_exposure = get_regime_adjusted_exposure(regime_metrics)
                w = w * regime_exposure
                w = w / w.sum() if w.sum() > 0 else w
            except Exception as e:
                logging.warning("R01 regime exposure scaling failed in simulation", exc_info=True)

        if w is None or len(w) == 0 or np.isclose(w.sum(), 0.0) or m not in fwd.index:
            rets.loc[m] = 0.0
            tno.loc[m]  = 0.0
            prev_w = pd.Series(dtype=float)
            continue

        # Next-month return with these weights
        valid = [t for t in w.index if t in fwd.columns]
        rets.loc[m] = float((fwd.loc[m, valid] * w.reindex(valid).fillna(0.0)).sum())

        # Turnover (0.5 * L1 distance over union of tickers)
        tno.loc[m] = float(l1_turnover(prev_w, w))
        prev_w = w

    return rets.fillna(0.0), tno.fillna(0.0)

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
            tkr = yf.Ticker(t)
            fi = tkr.fast_info or {}
            info = _safe_get_info(tkr)
            profitability = info.get("returnOnAssets") or info.get("profitMargins")
            leverage = info.get("debtToEquity")
            if leverage is None:
                leverage = fi.get("debtToEquity")
            if profitability is None:
                profitability = fi.get("returnOnAssets") or fi.get("profitMargins")
            if leverage is not None and leverage > 10:
                leverage = leverage / 100.0
        except Exception:
            logging.warning("E01 fundamental data fetch failed", exc_info=True)
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
    sectors_map: Dict[str, str],
    use_enhanced_features: bool = True,
) -> pd.Series:
    """Apply position sizing + hierarchical caps (name/sector + Software sub-caps)
    to the final combined portfolio.

    When ``use_enhanced_features`` is True, the raw sleeve weights are blended
    with risk-parity weights and adjusted by volatility-aware name caps before
    hierarchical cap enforcement. Cap trimming does **not** redistribute weight;
    any residual cash is returned to the caller to handle separately.
    """
    panels = compute_signal_panels(daily_close)
    monthly = panels.get("monthly", pd.DataFrame())

    # --- Momentum Component (NO CAPS YET) ---
    try:
        comp_all = composite_score(daily_close, panels=panels)
    except TypeError as exc:
        if "panels" in str(exc):
            comp_all = composite_score(daily_close)
        else:
            raise
    comp_vec = comp_all.iloc[-1].dropna() if isinstance(comp_all, pd.DataFrame) else pd.Series(comp_all).dropna()
    comp_vec_full = comp_vec.copy()

    momz = blended_momentum_z(monthly)
    mom_pool = comp_vec_full.copy()
    if isinstance(momz, pd.Series) and not momz.empty:
        aligned_momz = momz.reindex(mom_pool.index)
        valid_momz = aligned_momz.dropna()
        cutoff = 0.0
        if not valid_momz.empty:
            median_val = float(valid_momz.median())
            cutoff = max(median_val, 0.0) if np.isfinite(median_val) else 0.0
        mom_pool = mom_pool[aligned_momz > cutoff].dropna()
    if mom_pool.empty:
        mom_pool = comp_vec_full.dropna()
    _debug_stage("positive-mom pool", mom_pool.index)

    try:
        stable_iter = momentum_stable_names(
            daily_close,
            top_n=preset["mom_topn"],
            days=preset.get("stability_days", preset.get("stickiness_days", 7)),
            panels=panels,
        )
    except TypeError as exc:
        if "panels" in str(exc):
            stable_iter = momentum_stable_names(
                daily_close,
                top_n=preset["mom_topn"],
                days=preset.get("stability_days", preset.get("stickiness_days", 7)),
            )
        else:
            raise
    stable_names = set(stable_iter)
    if stable_names and not mom_pool.empty:
        sticky_pool = mom_pool.reindex([t for t in mom_pool.index if t in stable_names]).dropna()
        if not sticky_pool.empty:
            mom_pool = sticky_pool
    _debug_stage("stickiness/monthly lock", mom_pool.index)

    if mom_pool.empty:
        mom_pool = comp_vec_full.dropna()

    mom_cap = float(preset.get("mom_cap", 0.25))
    mom_weight = float(preset.get("mom_w", 0.0))
    w_mom_core = _safe_momentum_weights(mom_pool, top_n=int(preset["mom_topn"]), name_cap=mom_cap)
    mom_raw = w_mom_core * mom_weight if not w_mom_core.empty else pd.Series(dtype=float)

    # --- Mean Reversion Component (NO CAPS YET) ---
    st_ret  = daily_close.pct_change(preset["mr_lb"]).iloc[-1]
    ma200_df = panels.get("ma200")
    if preset.get("mr_ma") == 200 and isinstance(ma200_df, pd.DataFrame) and not ma200_df.empty:
        long_ma = ma200_df.iloc[-1]
    else:
        long_ma = daily_close.rolling(preset["mr_ma"]).mean().iloc[-1]
    quality = (daily_close.iloc[-1] > long_ma)
    _debug_stage("quality/MA gate", [t for t, ok in quality.items() if ok])
    w_mr_core = _safe_mr_weights(st_ret, quality.astype(bool), top_n=int(preset["mr_topn"]))
    mr_raw = w_mr_core * float(preset.get("mr_w", 0.0)) if not w_mr_core.empty else pd.Series(dtype=float)

    # --- Combine Components BEFORE Applying Caps ---
    combined_raw = mom_raw.add(mr_raw, fill_value=0.0)
    if combined_raw.empty or combined_raw.sum() <= 0:
        return combined_raw

    if use_enhanced_features:
        rp = risk_parity_weights(daily_close, combined_raw.index.tolist())
        rp = rp / rp.sum() if rp.sum() > 0 else rp
        lam = 0.4
        combined_raw = lam * combined_raw + (1 - lam) * (combined_raw.sum() * rp)

        vol_caps = get_volatility_adjusted_caps(
            combined_raw, daily_close, base_cap=preset.get("mom_cap", 0.25)
        )
        combined_raw = cap_weights(
            combined_raw, cap=preset.get("mom_cap", 0.25), vol_adjusted_caps=vol_caps
        )
    else:
        combined_raw = combined_raw / combined_raw.sum() if combined_raw.sum() > 0 else combined_raw

    # Enhanced sector map (uses your base sectors_map) + hierarchical caps for Software sub-buckets
    enhanced_sectors = get_enhanced_sector_map(list(combined_raw.index), base_map=sectors_map)
    group_caps = build_group_caps(enhanced_sectors)  # <- adds Software parent (30%) + sub-caps (e.g., 18%)

    # --- Enforce caps on the COMPLETE portfolio ---
    final_weights = enforce_caps_iteratively(
        combined_raw.astype(float),
        enhanced_sectors,
        name_cap=mom_cap,
        sector_cap=preset.get("sector_cap", 0.30),
        group_caps=group_caps,             # <- IMPORTANT: turns on the sub-caps
    )

    target_equity = float(preset.get("target_equity", 1.0))
    final_weights = _rescale_and_floor(
        final_weights.astype(float),
        sectors_map,
        name_cap=mom_cap,
        sector_cap=preset.get("sector_cap", 0.30),
        target_equity=target_equity,
    )

    return final_weights

def check_constraint_violations(
    weights: pd.Series,
    sectors_map: Dict[str, str],
    name_cap: float,
    sector_cap: float,
    group_caps: dict[str, float] | None = None,
) -> List[str]:
    """
    Check for constraint violations in final portfolio
    Returns list of violation descriptions
    """
    violations = []
    
    # Check name caps
    tol = 1e-6

    for ticker, weight in weights.items():
        if weight > name_cap + tol:
            violations.append(f"{ticker}: {weight:.1%} > {name_cap:.1%}")
    
    # Check sector caps
    sectors = pd.Series({t: sectors_map.get(t, "Unknown") for t in weights.index})
    sector_sums = weights.groupby(sectors).sum()

    for sector, total_weight in sector_sums.items():
        if total_weight > sector_cap + tol:
            violations.append(f"{sector}: {total_weight:.1%} > {sector_cap:.1%}")

    # Optional hierarchical/group caps (e.g., Software sub-buckets)
    if group_caps:
        # Build enhanced sector mapping so group labels match potential sub-buckets
        enhanced_map = get_enhanced_sector_map(list(weights.index), base_map=sectors_map)
        ser_group = pd.Series(enhanced_map)
        group_sums = weights.groupby(ser_group).sum()
        parent_sums = weights.groupby(ser_group.map(lambda s: s.split(":")[0])).sum()

        for group, cap in group_caps.items():
            if ":" in group:
                w = group_sums.get(group, 0.0)
            else:
                w = parent_sums.get(group, 0.0)
            if w > cap + tol:
                violations.append(f"{group}: {w:.1%} > {cap:.1%}")
    
    return violations

def _format_display(weights: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    display_df = pd.DataFrame({"Weight": weights}).sort_values("Weight", ascending=False)
    display_fmt = display_df.copy()
    display_fmt["Weight"] = display_fmt["Weight"].map("{:.2%}".format)
    return display_fmt, display_df


def _store_live_prune_meta(meta: Dict[str, Any]) -> None:
    """Persist the latest price-history pruning metadata for UI diagnostics."""

    try:
        eligible_raw = list(meta.get("eligible_tickers") or [])
        dropped_raw = list(meta.get("dropped_tickers") or [])

        def _filter(items: List[str]) -> List[str]:
            seen: dict[str, None] = {}
            out: List[str] = []
            for item in items:
                if item in {"QQQ", HEDGE_TICKER_LABEL}:
                    continue
                if item not in seen:
                    seen[item] = None
                    out.append(item)
            return out

        eligible = _filter(eligible_raw)
        dropped = _filter(dropped_raw)
        st.session_state["live_prune_meta"] = {
            "eligible_tickers": list(eligible),
            "eligible_count": int(len(eligible)),
            "dropped_tickers": list(dropped),
        }
    except Exception:
        # Streamlit session state may be unavailable in some non-UI contexts
        return


def _debug_stage(label: str, obj) -> None:
    """Record eligibility counts for diagnostics without raising on failure."""

    try:
        n = len(obj)  # type: ignore[arg-type]
    except Exception:
        n = None
    logging.info("ELIGIBILITY • %-22s -> %s", label, n)
    try:
        st.session_state.setdefault("eligibility_debug", []).append((label, n))
    except Exception:
        pass


def _prune_price_history_with_meta(
    prices: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Drop unusable price columns and return (pruned_prices, metadata)."""

    if prices is None:
        empty = pd.DataFrame()
        meta = {"eligible_tickers": [], "eligible_count": 0, "dropped_tickers": []}
        return empty, meta

    original_cols = list(getattr(prices, "columns", []))
    if not original_cols:
        meta = {"eligible_tickers": [], "eligible_count": 0, "dropped_tickers": []}
        return prices.copy(), meta

    pruned = prices.copy()
    # Drop columns that are entirely missing; retain the remainder unchanged
    valid_cols = pruned.columns[pruned.notna().any()].tolist()
    pruned = pruned.loc[:, valid_cols]

    dropped = sorted({c for c in original_cols if c not in valid_cols})
    meta = {
        "eligible_tickers": valid_cols,
        "eligible_count": len(valid_cols),
        "dropped_tickers": dropped,
    }
    return pruned, meta


def _safe_momentum_weights(scores: pd.Series, top_n: int, name_cap: float) -> pd.Series:
    """Return non-empty weights even if all scores are non-positive."""

    if scores is None or scores.empty:
        return pd.Series(dtype=float)

    ranked = scores.dropna().sort_values(ascending=False)
    top = ranked.head(max(1, int(top_n)))

    raw = top[top > 0]
    if raw.empty or raw.sum() <= 0:
        raw = top.clip(lower=0)

    if raw.sum() <= 0:
        raw = pd.Series(1.0 / len(top), index=top.index)
    else:
        raw = raw / raw.sum()

    w = cap_weights(raw, cap=float(name_cap))
    if w.sum() <= 0:
        w = pd.Series(1.0 / len(raw), index=raw.index)

    return (w / w.sum()).astype(float)


def _safe_mr_weights(
    short_window_returns: pd.Series,
    quality_mask: pd.Series | None,
    top_n: int,
) -> pd.Series:
    pool = (
        quality_mask[quality_mask].index
        if isinstance(quality_mask, pd.Series)
        else short_window_returns.index
    )
    candidates = short_window_returns.reindex(pool).dropna()
    if candidates.empty:
        candidates = short_window_returns.dropna()
    picks = candidates.nsmallest(max(1, int(top_n)))
    if picks.empty:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(picks), index=picks.index)


def _rescale_and_floor(
    weights: pd.Series,
    sector_map: dict[str, str],
    name_cap: float,
    sector_cap: float,
    target_equity: float = 0.85,
    *,
    respect_capacity: bool = False,
) -> pd.Series:
    if weights is None or weights.empty:
        return pd.Series(dtype=float)

    capped = cap_weights(weights.clip(lower=0).fillna(0.0), cap=float(name_cap))

    try:
        sectors = pd.Series({t: sector_map.get(t, "Other") for t in capped.index})
        by_sector = capped.groupby(sectors).sum()
        scale = {t: 1.0 for t in capped.index}
        for s, tot in by_sector.items():
            if tot > float(sector_cap):
                shrink = float(sector_cap) / max(1e-12, float(tot))
                for t in sectors.index[sectors == s]:
                    scale[t] = shrink
        capped = capped * pd.Series(scale)
    except Exception:
        pass

    capped = capped.clip(lower=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    total = float(capped.sum())
    if total <= 0:
        capped = pd.Series(1.0 / len(weights), index=weights.index)
        total = float(capped.sum())

    normalized = capped / total if total > 0 else capped
    target = float(target_equity)
    if respect_capacity and total > 0 and target > total:
        target = total

    return (normalized * target).astype(float)

def generate_live_portfolio_isa_monthly(
    preset: Dict,
    prev_portfolio: Optional[pd.DataFrame],
    min_dollar_volume: float = 0.0,
    as_of: date | None = None,
    use_enhanced_features: bool = True,
    use_incremental: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """
    Enhanced ISA Dynamic live weights with MONTHLY LOCK + composite + stability + sector caps.
    Now includes regime awareness, volatility adjustments, and an optional `as_of` date.
    Cap trimming leaves residual cash until the final exposure scaling performed
    within this routine.
    """
    _store_live_prune_meta({"eligible_tickers": [], "dropped_tickers": []})
    try:
        st.session_state["eligibility_debug"] = []
    except Exception:
        pass

    universe_choice = st.session_state.get("universe", "Hybrid Top150")
    base_params = STRATEGY_PRESETS["ISA Dynamic (0.75)"]
    params = dict(preset)
    for key, value in base_params.items():
        params.setdefault(key, value)

    stickiness_days = st.session_state.get("stickiness_days", params.get("stability_days", 7))
    sector_cap      = st.session_state.get("sector_cap", params.get("sector_cap", 0.30))
    mom_cap         = st.session_state.get("name_cap", params.get("mom_cap", 0.25))

    # build params from preset, then override with UI/session values
    params["stability_days"] = int(stickiness_days)
    params["stickiness_days"] = int(stickiness_days)
    params["sector_cap"]     = float(sector_cap)
    params["mom_cap"]        = float(mom_cap)
    params.setdefault("target_equity", 0.85)

    # Universe base tickers + sectors
    base_tickers, base_sectors, label = get_universe(universe_choice)
    base_tickers = _sc_sanitize_tickers(base_tickers)
    if "QQQ" not in base_tickers:
        base_tickers.append("QQQ")
    base_sectors = {t: base_sectors.get(t, "Unknown") for t in base_tickers}
    if not base_tickers:
        _store_live_prune_meta({"eligible_tickers": [], "dropped_tickers": []})
        return None, None, "No universe available."

    today = as_of or date.today()
    # Fetch prices
    start = (today - relativedelta(months=max(preset["mom_lb"], 12) + 8)).strftime("%Y-%m-%d")
    end   = today.strftime("%Y-%m-%d")
    close_raw, vol_raw = fetch_price_volume(base_tickers, start, end)
    if close_raw.empty:
        _store_live_prune_meta({
            "eligible_tickers": [],
            "dropped_tickers": [t for t in base_tickers if t != "QQQ"],
        })
        return None, None, "No price data."

    # Special: Hybrid Top150 → reduce by 60d median dollar volume
    sectors_map = base_sectors.copy()
    if label == "Hybrid Top150":
        med = median_dollar_volume(close_raw, vol_raw, window=60).sort_values(ascending=False)
        top_list = med.head(150).index.tolist()
        close_raw = close_raw[top_list]
        vol_raw   = vol_raw[top_list]
        sectors_map = {t: base_sectors.get(t, "Unknown") for t in close_raw.columns}

    close_base = close_raw.copy()
    vol_base = vol_raw.copy()
    sectors_base = sectors_map.copy()

    min_prof = float(st.session_state.get("min_profitability", 0.0))
    max_lev = float(st.session_state.get("max_leverage", 2.0))

    def _apply_primary_filters(
        min_dollar_volume_val: float,
        min_profitability_val: float,
    ) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
        local_close = close_base.copy()
        local_vol = vol_base.copy()
        local_sectors = {t: sectors_base.get(t, "Unknown") for t in local_close.columns}

        equities_initial = [t for t in local_close.columns if t != "QQQ"]
        _debug_stage("universe (input)", equities_initial)

        liq_pool = equities_initial
        if min_dollar_volume_val > 0:
            keep = filter_by_liquidity(
                local_close.drop(columns=["QQQ"], errors="ignore"),
                local_vol.drop(columns=["QQQ"], errors="ignore"),
                float(min_dollar_volume_val),
            )
            keep = [t for t in keep if t in local_close.columns]
            liq_pool = [t for t in keep if t != "QQQ"]
            _debug_stage("liquidity gate", liq_pool)
            if keep:
                keep_ordered = list(dict.fromkeys(keep))
                if "QQQ" in local_close.columns and "QQQ" not in keep_ordered:
                    keep_ordered.append("QQQ")
                local_close = local_close.loc[:, [c for c in keep_ordered if c in local_close.columns]]
                local_vol = local_vol.reindex(columns=local_close.columns)
                local_sectors = {t: sectors_base.get(t, "Unknown") for t in local_close.columns}
            else:
                local_close = local_close.iloc[:, :0]
        else:
            _debug_stage("liquidity gate", liq_pool)

        if local_close.empty:
            meta = {"eligible_tickers": [], "eligible_count": 0, "dropped_tickers": equities_initial}
            return local_close, local_sectors, meta

        fundamentals = fetch_fundamental_metrics([t for t in local_close.columns if t != "QQQ"])
        keep_quality = fundamental_quality_filter(
            fundamentals,
            min_profitability=float(min_profitability_val),
            max_leverage=max_lev,
        )
        quality_pool = [t for t in keep_quality if t in local_close.columns and t != "QQQ"]
        _debug_stage("quality/MA gate", quality_pool if keep_quality else [t for t in local_close.columns if t != "QQQ"])
        if keep_quality:
            keep_ordered = list(dict.fromkeys(keep_quality))
            if "QQQ" in local_close.columns and "QQQ" not in keep_ordered:
                keep_ordered.append("QQQ")
            local_close = local_close.loc[:, [c for c in keep_ordered if c in local_close.columns]]
            local_vol = local_vol.reindex(columns=local_close.columns)
            local_sectors = {t: sectors_base.get(t, "Unknown") for t in local_close.columns}
        elif min_profitability_val > 0:
            local_close = local_close.iloc[:, :0]

        if local_close.empty:
            meta = {"eligible_tickers": [], "eligible_count": 0, "dropped_tickers": equities_initial}
            return local_close, local_sectors, meta

        local_close = _soft_prune_history(local_close, min_days=180, max_missing_frac=0.60)
        if local_close.empty:
            meta = {"eligible_tickers": [], "eligible_count": 0, "dropped_tickers": equities_initial}
            _debug_stage("data survivors", [])
            return local_close, local_sectors, meta

        pruned_close, prune_meta = _prune_price_history_with_meta(local_close)
        if pruned_close.empty:
            _debug_stage("data survivors", [])
            return pruned_close, local_sectors, prune_meta

        local_sectors = {t: sectors_base.get(t, "Unknown") for t in pruned_close.columns}
        survivors = [t for t in pruned_close.columns if t != "QQQ"]
        _debug_stage("data survivors", survivors)
        return pruned_close, local_sectors, prune_meta

    filtered_close, filtered_sectors, prune_meta = _apply_primary_filters(min_dollar_volume, min_prof)
    survivors = [t for t in filtered_close.columns if t != "QQQ"]
    survivor_cnt = len(survivors)
    if survivor_cnt < 120 and (min_dollar_volume > 0 or min_prof > 0):
        logging.info(
            "Relaxing gates: %d survivors < 120 — removing liquidity/quality floors.",
            survivor_cnt,
        )
        try:
            st.session_state["eligibility_debug"] = []
        except Exception:
            pass
        min_dollar_volume = 0
        min_prof = 0.0
        filtered_close, filtered_sectors, prune_meta = _apply_primary_filters(min_dollar_volume, min_prof)
        survivors = [t for t in filtered_close.columns if t != "QQQ"]

    close = filtered_close
    sectors_map = filtered_sectors
    _store_live_prune_meta(prune_meta)

    survivors_index = close.columns
    want = pd.Index(sorted(set(base_tickers)))
    have = survivors_index.intersection(want)
    if have.empty:
        logging.warning(
            "Universe filter removed everything; using survivors only (%d names).",
            len(survivors_index),
        )
        have = survivors_index
    close = close.loc[:, have]
    sectors_map = {t: sectors_map.get(t, "Unknown") for t in close.columns}
    if close.empty:
        return None, None, "No eligible tickers after price history pruning."

    # Expose the price index for downstream processes (e.g. save gating)
    st.session_state["latest_price_index"] = close.index

    regime_metrics: Dict[str, float] = {}
    if len(close) > 0:
        try:
            regime_metrics = compute_regime_metrics(close)
        except Exception:
            logging.warning("R02 regime metric calculation failed in allocation", exc_info=True)
            regime_metrics = {}

    # Monthly lock check – always compute new weights
    is_monthly = is_rebalance_today(today, close.index)
    decision = "Preview only – portfolio not saved" if not is_monthly else ""

    # Build candidate weights (enhanced) – overrides applied above
    new_w = _build_isa_weights_fixed(
        close, params, sectors_map, use_enhanced_features=use_enhanced_features
    )

    target_equity = float(params.get("target_equity", 0.85))

    # Apply regime-based exposure scaling to final weights (respecting floor)
    if (
        use_enhanced_features
        and len(close) > 0
        and isinstance(new_w, pd.Series)
        and not new_w.empty
    ):
        try:
            regime_exposure = get_regime_adjusted_exposure(regime_metrics)
            regime_target = max(float(regime_exposure), target_equity)
            new_w = _rescale_and_floor(
                new_w,
                sectors_map,
                params["mom_cap"],
                params.get("sector_cap", 0.30),
                target_equity=regime_target,
            )
        except Exception:
            logging.warning("R02 regime exposure scaling failed in allocation", exc_info=True)

    # Validate and re-enforce caps if scaling introduced violations
    if len(new_w) > 0:
        enhanced_sectors = get_enhanced_sector_map(list(new_w.index), base_map=sectors_map)
        group_caps = build_group_caps(enhanced_sectors)
        new_w = enforce_caps_iteratively(
            new_w.astype(float),
            enhanced_sectors,
            name_cap=params["mom_cap"],
            sector_cap=params.get("sector_cap", 0.30),
            group_caps=group_caps,
        )
        violations = check_constraint_violations(
            new_w,
            sectors_map,
            params["mom_cap"],
            params.get("sector_cap", 0.30),
            group_caps=group_caps,
        )
        if violations:
            raise ValueError(
                f"Constraint violations after re-enforcing caps: {violations}"
            )

    final_weights = new_w.astype(float) if isinstance(new_w, pd.Series) else pd.Series(dtype=float)

    # Stickiness / lock fallback: ensure we carry forward previous allocation when needed
    if (final_weights.empty or final_weights.sum() <= 0) and prev_portfolio is not None and not prev_portfolio.empty and "Weight" in prev_portfolio.columns:
        prev_w = prev_portfolio["Weight"].astype(float).drop(index=HEDGE_TICKER_LABEL, errors="ignore")
        final_weights = _rescale_and_floor(
            prev_w,
            sectors_map,
            params["mom_cap"],
            params.get("sector_cap", 0.30),
            target_equity=float(prev_w.sum() or target_equity),
            respect_capacity=True,
        )
        decision = decision or "Fallback to previous allocation due to empty candidate set."

    if not is_monthly and prev_portfolio is not None and not prev_portfolio.empty and "Weight" in prev_portfolio.columns:
        prev_w = prev_portfolio["Weight"].astype(float).drop(index=HEDGE_TICKER_LABEL, errors="ignore")
        final_weights = _rescale_and_floor(
            prev_w,
            sectors_map,
            params["mom_cap"],
            params.get("sector_cap", 0.30),
            target_equity=float(prev_w.sum() or target_equity),
            respect_capacity=True,
        )

    hedge_weight = 0.0
    corr = np.nan
    portfolio_recent = pd.Series(dtype=float)
    try:
        cfg_live = HybridConfig(
            momentum_top_n=int(params["mom_topn"]),
            momentum_cap=float(params["mom_cap"]),
            mr_top_n=int(params["mr_topn"]),
            mom_weight=float(params["mom_w"]),
            mr_weight=float(params["mr_w"]),
            mr_lookback_days=int(params["mr_lb"]),
            mr_long_ma_days=int(params["mr_ma"]),
        )
        cache_key = ""
        if use_incremental and len(close) > 0:
            cache_key = _build_hybrid_cache_key(close.columns, cfg_live, close.index.min(), False)
        bt_res = _run_hybrid_backtest_with_cache(
            close,
            cfg_live,
            cache_key,
            apply_vol_target=False,
            use_incremental=use_incremental,
        )
        portfolio_recent = bt_res.get("hybrid_rets", pd.Series(dtype=float)).tail(6)
        hedge_size = (
            st.session_state.get("max_hedge", HEDGE_MAX_DEFAULT)
            if _HAS_ST else HEDGE_MAX_DEFAULT
        )
        hedge_weight = build_hedge_weight(portfolio_recent, regime_metrics, hedge_size)
        corr = calculate_portfolio_correlation_to_market(portfolio_recent)
    except Exception:
        logging.warning("H02 hedge overlay failed in allocation", exc_info=True)

    # Trigger vs previous portfolio (health of current)
    if is_monthly and prev_portfolio is not None and not prev_portfolio.empty and "Weight" in prev_portfolio.columns:
        monthly = close.resample("M").last()
        mom_scores = blended_momentum_z(monthly)
        if not mom_scores.empty and len(new_w) > 0:
            top_m = mom_scores.nlargest(params["mom_topn"])
            top_score = float(top_m.iloc[0]) if len(top_m) > 0 else 1e-9
            prev_w = prev_portfolio["Weight"].astype(float)
            prev_w = prev_w.drop(index=HEDGE_TICKER_LABEL, errors="ignore")
            held_scores = mom_scores.reindex(prev_w.index).fillna(0.0)
            health = float((held_scores * prev_w).sum() / max(top_score, 1e-9))
            if health >= params["trigger"]:
                enhanced_map = get_enhanced_sector_map(list(prev_w.index), base_map=sectors_map)
                group_caps = build_group_caps(enhanced_map)
                prev_w = enforce_caps_iteratively(
                    prev_w,
                    enhanced_map,
                    mom_cap,
                    sector_cap,
                    group_caps=group_caps,
                )
                prev_w = _rescale_and_floor(
                    prev_w,
                    sectors_map,
                    mom_cap,
                    sector_cap,
                    target_equity=target_equity,
                    respect_capacity=True,
                )
                violations = check_constraint_violations(
                    prev_w, sectors_map, mom_cap, sector_cap, group_caps=group_caps
                )
                if not violations:
                    decision = f"Health {health:.2f} ≥ trigger {params['trigger']:.2f} — holding existing portfolio."
                    final_weights = prev_w.astype(float)
                else:
                    decision = (
                        f"Health {health:.2f} ≥ trigger {params['trigger']:.2f}"
                        " — constraints violated, rebalancing to new targets."
                    )
            else:
                decision = f"Health {health:.2f} < trigger {params['trigger']:.2f} — rebalancing to new targets."

    def _inject_hedge(weights: pd.Series) -> pd.Series:
        w = weights.copy() if weights is not None else pd.Series(dtype=float)
        if not isinstance(w, pd.Series):
            w = pd.Series(dtype=float)
        w = w.drop(index=HEDGE_TICKER_LABEL, errors="ignore")
        if hedge_weight > 0:
            w.loc[HEDGE_TICKER_LABEL] = -hedge_weight
        return w.astype(float)

    final_weights = _inject_hedge(final_weights if isinstance(final_weights, pd.Series) else pd.Series(dtype=float))

    state = "active" if hedge_weight > 0 else "inactive"
    corr_txt = "n/a" if pd.isna(corr) else f"{corr:.2f}"
    _emit_info(
        f"Live allocation QQQ hedge {state} (weight={hedge_weight:.1%}, corr={corr_txt})"
    )
    _record_hedge_state("live", hedge_weight, corr, regime_metrics)

    hedge_note = (
        f"QQQ hedge active at {hedge_weight:.1%} short."
        if hedge_weight > 0
        else "QQQ hedge inactive."
    )
    decision = (decision or "").strip()
    decision = f"{decision}\n{hedge_note}" if decision else hedge_note

    disp, raw = _format_display(final_weights)
    return disp, raw, decision

def optimize_hybrid_strategy(prices: Optional[pd.DataFrame] = None,
                             start_date: str = "2017-07-01",
                             end_date: Optional[str] = None,
                             universe: str = "Hybrid Top150",
                             n_jobs: Optional[int] = None) -> Tuple[HybridConfig, float]:
    """Autonomously search for a Sharpe-maximizing ``HybridConfig``.

    Parameters
    ----------
    prices : DataFrame, optional
        Pre-fetched daily price data.  If ``None`` the universe is fetched
        using :func:`_prepare_universe_for_backtest`.
    start_date, end_date : str
        Date range used when ``prices`` is ``None``.
    universe : str
        Universe name passed to ``_prepare_universe_for_backtest`` when
        fetching data internally.
    n_jobs : int, optional
        Overrides the worker count used during grid-search evaluation.
        Defaults to the value specified in :data:`PERF` when available.

    Returns
    -------
    cfg : HybridConfig
        Configuration found by :func:`optimizer.grid_search_hybrid`.
    sector_cap : float
        Currently selected sector cap value.  This parameter is not part of
        ``HybridConfig`` so it is returned separately.
    """
    if prices is None or prices.empty:
        if end_date is None:
            end_date = date.today().strftime("%Y-%m-%d")
        close, _, _, _ = _prepare_universe_for_backtest(universe, start_date, end_date)
        if close.empty:
            return HybridConfig(), 0.30
        prices = close.drop(columns=["QQQ"], errors="ignore")

    param_grid = {
        "top_n": [5, 8, 12],
        "name_cap": [0.20, 0.25, 0.30],
        "sector_cap": [0.25, 0.30],
        "mom_weight": [0.7, 0.8, 0.9],
        "mr_weight": [0.1, 0.2, 0.3],
    }

    search_grid = {
        "momentum_top_n": param_grid["top_n"],
        "momentum_cap": param_grid["name_cap"],
        "mom_weight": param_grid["mom_weight"],
        "mr_weight": param_grid["mr_weight"],
    }

    perf_jobs = PERF.get("n_jobs")
    best_cfg, _ = optimizer.grid_search_hybrid(
        prices,
        search_grid,
        n_jobs=n_jobs or perf_jobs,
    )
    # ``sector_cap`` is not part of ``HybridConfig``; we simply choose the
    # first value in the grid (callers may override as needed).
    sector_cap = param_grid["sector_cap"][0]
    return best_cfg, sector_cap

def run_backtest_isa_dynamic(
    roundtrip_bps: float = 0.0,
    min_dollar_volume: float = 0.0,
    show_net: bool = True,
    start_date: str = "2017-07-01",
    end_date: Optional[str] = None,
    universe_choice: Optional[str] = "Hybrid Top150",
    top_n: Optional[int] = None,
    name_cap: Optional[float] = None,
    sector_cap: Optional[float] = None,
    stickiness_days: int = 7,
    mr_topn: int = 3,
    mom_weight: Optional[float] = None,
    mr_weight: Optional[float] = None,
    use_enhanced_features: bool = True,
    apply_quality_filter: bool = False,
    auto_optimize: bool = False,
    target_vol_annual: Optional[float] = None,
    apply_vol_target: bool = False,
    use_incremental: bool = True,
    n_jobs: Optional[int] = None,
) -> Tuple[
    Optional[pd.Series],
    Optional[pd.Series],
    Optional[pd.Series],
    Optional[pd.Series],
    Optional[HybridConfig],
    pd.DataFrame,
]:
    """
    Enhanced ISA-Dynamic hybrid backtest with stickiness, sector caps, and optional extras.

    Parameters
    ----------
    roundtrip_bps : float, default 0.0
        Round-trip transaction cost in basis points used to model trading drag.
    min_dollar_volume : float, default 0.0
        Liquidity floor for universe filtering (ignored if your pipeline
        doesn't compute dollar volume).
    show_net : bool, default True
        If True, downstream displays prefer net returns; both gross and net are computed.
    start_date, end_date : str
        Backtest window (YYYY-MM-DD). end_date defaults to today.
    universe_choice : {"Hybrid Top150","NASDAQ100+","S&P500 (All)"}, optional
        Universe selector used by your data/universe builder.
    top_n, name_cap, sector_cap : optional
        Overrides for momentum sleeve breadth and caps. Pass None to use preset
        or optimisation.
    stickiness_days : int, default 7
        Holding “stickiness” (days) to reduce churn in the live portfolio logic.
    mr_topn : int, default 3
        Breadth of the mean-reversion sleeve.
    mom_weight, mr_weight : optional
        Hybrid blend weights. Pass None to use preset or optimisation.
    use_enhanced_features : bool, default True
        Enable volatility-adjusted caps / regime awareness / signal decay (if implemented).
    apply_quality_filter : bool, default False
        If True, apply a current-fundamentals quality screen. Keep False for historical
        tests to avoid look-ahead bias.
    auto_optimize : bool, default False
        If True, run parameter optimisation (e.g., grid/Bayesian) and return
        any best-found configuration alongside diagnostics for inspection.
    target_vol_annual : float, optional
        Annualised volatility target (e.g., 0.15) for optional vol targeting.
    apply_vol_target : bool, default False
        If True and `target_vol_annual` is set, apply volatility targeting to the
        combined series.
    use_incremental : bool, default True
        When True, reuse cached sleeve results to accelerate repeat runs.
    n_jobs : int, optional
        Worker count forwarded to optimisation helpers.  Defaults to
        :data:`PERF["n_jobs"] <PERF>` when unspecified.

    Returns
    -------
    strat_cum_gross : Series or None
        Strategy cumulative equity (gross of costs).
    strat_cum_net : Series or None
        Strategy cumulative equity (net of costs / vol targeting as configured).
    qqq_cum : Series or None
        Benchmark cumulative equity (e.g., QQQ).
    hybrid_tno : Series or None
        Monthly turnover (0.5 × L1) for the hybrid portfolio.
    optimization_cfg : HybridConfig or None
        Best configuration identified during optimisation (if any search was run).
    search_diagnostics : DataFrame
        Diagnostics or search history from optimisation routines (empty if unused).
    """
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    # Universe & data (helper)
    close, vol, sectors_map, label = _prepare_universe_for_backtest(universe_choice, start_date, end_date)
    if close.empty or "QQQ" not in close.columns:
        return None, None, None, None, None, pd.DataFrame()

    # Liquidity floor (optional)
    if min_dollar_volume > 0:
        keep = filter_by_liquidity(
            close.drop(columns=["QQQ"], errors="ignore"),
            vol.drop(columns=["QQQ"], errors="ignore"),
            min_dollar_volume
        )
        keep_cols = [c for c in keep if c in close.columns]
        if not keep_cols:
            return None, None, None, None, None, pd.DataFrame()
        close = close[keep_cols + ["QQQ"]]
        sectors_map = {t: sectors_map.get(t, "Unknown") for t in keep_cols}

    daily = close.drop(columns=["QQQ"])
    qqq  = close["QQQ"]

    # Fundamental quality filter (optional to avoid look-ahead bias in backtests)
    if apply_quality_filter:
        min_prof = st.session_state.get("min_profitability", 0.0)
        max_lev = st.session_state.get("max_leverage", 2.0)
        fundamentals = fetch_fundamental_metrics(daily.columns.tolist())
        keep = fundamental_quality_filter(
            fundamentals, min_profitability=min_prof, max_leverage=max_lev
        )
        if not keep:
            return None, None, None, None, None, pd.DataFrame()
        daily = daily[keep]
        sectors_map = {t: sectors_map.get(t, "Unknown") for t in keep}

    daily = _soft_prune_history(daily, min_days=180, max_missing_frac=0.60)
    if daily.empty:
        raise ValueError("No tickers have sufficient price history after applying filters.")
    sectors_map = {t: sectors_map.get(t, "Unknown") for t in daily.columns}

    optimization_cfg: Optional[HybridConfig] = None
    search_diagnostics = pd.DataFrame()

    if auto_optimize:
        base_defaults = HybridConfig()
        base_cfg = HybridConfig(
            momentum_lookback_m=base_defaults.momentum_lookback_m,
            momentum_top_n=top_n or base_defaults.momentum_top_n,
            momentum_cap=name_cap or base_defaults.momentum_cap,
            mr_lookback_days=base_defaults.mr_lookback_days,
            mr_top_n=mr_topn or base_defaults.mr_top_n,
            mr_long_ma_days=base_defaults.mr_long_ma_days,
            mom_weight=mom_weight or base_defaults.mom_weight,
            mr_weight=mr_weight or base_defaults.mr_weight,
            predictive_weight=0.0,
            target_vol_annual=base_defaults.target_vol_annual,
        )
        search_space = {
            "momentum_top_n": range(6, 21),
            "momentum_cap": [0.15, 0.2, 0.25, 0.3, 0.35],
            "momentum_lookback_m": [3, 6, 9, 12],
            "mr_top_n": [2, 3, 4, 5, 6],
            "mr_lookback_days": [10, 15, 21, 30],
            "mr_long_ma_days": [150, 180, 200, 220, 250],
            "mom_weight": [0.55, 0.65, 0.75, 0.85, 0.9],
            "mr_weight": [0.05, 0.15, 0.25, 0.35, 0.45],
        }
        optimization_cfg, search_diagnostics = optimizer.bayesian_optimize_hybrid(
            daily,
            search_space=search_space,
            base_cfg=base_cfg,
            tc_bps=roundtrip_bps if show_net else 0.0,
            apply_vol_target=False,
        )
        top_n = optimization_cfg.momentum_top_n
        name_cap = optimization_cfg.momentum_cap
        mr_topn = optimization_cfg.mr_top_n
        mom_weight = optimization_cfg.mom_weight
        mr_weight = optimization_cfg.mr_weight
    elif any(p is None for p in (top_n, name_cap, sector_cap, mom_weight, mr_weight)):
        cfg, opt_sector_cap = optimize_hybrid_strategy(daily, n_jobs=n_jobs)
        top_n = top_n or cfg.momentum_top_n
        name_cap = name_cap or cfg.momentum_cap
        mom_weight = mom_weight or cfg.mom_weight
        mr_weight = mr_weight or cfg.mr_weight
        sector_cap = sector_cap or opt_sector_cap

    # --- Build HybridConfig (works with/without an optimizer-provided cfg) ---
    tc_for_sim = roundtrip_bps if show_net else 0.0

    # Only pass non-None overrides so dataclass defaults remain intact
    base_kwargs = {
        "momentum_top_n": top_n,
        "momentum_cap": name_cap,
        "mr_top_n": mr_topn,
        "mom_weight": mom_weight,
        "mr_weight": mr_weight,
        "mr_lookback_days": 21,
        "mr_long_ma_days": 200,
        "tc_bps": tc_for_sim,
        # apply vol target only if requested
        "target_vol_annual": (target_vol_annual if apply_vol_target else None),
    }

    # Drop keys whose values are None to avoid overriding dataclass defaults
    base_kwargs = {k: v for k, v in base_kwargs.items() if v is not None}

    # If an optimisation step produced a config, start from it and overlay overrides
    if "optimization_cfg" in locals() and optimization_cfg is not None:
        cfg = replace(optimization_cfg, **base_kwargs)
    else:
        cfg = HybridConfig(**base_kwargs)

    cache_key = ""
    if use_incremental and not daily.empty:
        cache_key = _build_hybrid_cache_key(daily.columns, cfg, daily.index.min(), apply_vol_target)
    res = _run_hybrid_backtest_with_cache(
        daily,
        cfg,
        cache_key,
        apply_vol_target=apply_vol_target,
        use_incremental=use_incremental,
    )
    hybrid_gross = res["hybrid_rets"]
    hybrid_tno = (
        cfg.mom_weight * res["mom_turnover"].reindex(hybrid_gross.index).fillna(0)
        + cfg.mr_weight * res["mr_turnover"].reindex(hybrid_gross.index).fillna(0)
    )

    qqq_monthly = qqq.resample("M").last().pct_change()

    # Apply drawdown-based exposure adjustment (walk-forward)
    if use_enhanced_features:
        hybrid_gross = apply_dynamic_drawdown_scaling(
            hybrid_gross, qqq_monthly, threshold_fraction=0.8
        )

    hedge_weight = 0.0
    corr = np.nan
    regime_metrics: Dict[str, float] = {}
    portfolio_recent = hybrid_gross.tail(6)

    try:
        lookback_daily = daily.iloc[-252:] if len(daily) > 252 else daily
        if not lookback_daily.empty:
            regime_metrics = compute_regime_metrics(lookback_daily)

        hedge_size = (
            st.session_state.get("max_hedge", HEDGE_MAX_DEFAULT)
            if _HAS_ST else HEDGE_MAX_DEFAULT
        )
        hedge_weight = build_hedge_weight(portfolio_recent, regime_metrics, hedge_size)
        corr = calculate_portfolio_correlation_to_market(portfolio_recent)

        if hedge_weight > 0 and not qqq_monthly.empty:
            qqq_aligned = qqq_monthly.reindex(hybrid_gross.index).fillna(0.0)
            hedge_returns = -hedge_weight * qqq_aligned
            hybrid_gross = hybrid_gross.add(hedge_returns, fill_value=0.0)
    except Exception:
        logging.warning("H01 hedge overlay failed in backtest", exc_info=True)

    state = "active" if hedge_weight > 0 else "inactive"
    corr_txt = "n/a" if pd.isna(corr) else f"{corr:.2f}"
    _emit_info(
        f"Backtest QQQ hedge {state} (weight={hedge_weight:.1%}, corr={corr_txt})"
    )
    _record_hedge_state("backtest", hedge_weight, corr, regime_metrics)

    hybrid_net = apply_costs(hybrid_gross, hybrid_tno, roundtrip_bps) if show_net else hybrid_gross

    # Cum curves
    strat_cum_gross = (1 + hybrid_gross.fillna(0)).cumprod()
    strat_cum_net   = (1 + hybrid_net.fillna(0)).cumprod() if show_net else None
    qqq_cum = (1 + qqq_monthly).cumprod().reindex(strat_cum_gross.index, method="ffill")

    return (
        strat_cum_gross,
        strat_cum_net,
        qqq_cum,
        hybrid_tno,
        optimization_cfg,
        search_diagnostics,
    )
    
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
    monthly = daily_prices.resample("M").last()
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

def _build_change_reason(action: str, delta_bps: int, mom_rank: float,
                         mom_score: float, st_ret: float,
                         above_ma: float) -> str:
    """Generate a short natural language explanation for a portfolio change."""
    verb = {"Buy": "Buying", "Sell": "Selling", "Rebalance": "Rebalancing"}.get(action, action)
    parts: List[str] = []
    if pd.notna(mom_rank):
        parts.append(f"momentum rank {int(mom_rank)}")
    if pd.notna(mom_score):
        parts.append(f"score {mom_score:.2f}")
    if pd.notna(st_ret):
        parts.append(f"{st_ret:+.1%} short-term return")
    if pd.notna(above_ma):
        parts.append("above 200DMA" if above_ma else "below 200DMA")
    metrics = ", ".join(parts)
    prefix = f"{verb} {delta_bps:+d} bps" if delta_bps else verb
    if metrics:
        return f"{prefix} due to {metrics}"
    return prefix

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
            "Mom Rank","Mom Score",f"ST Return ({params['mr_lb']}d)",
            f"Above {params['mr_ma']}DMA","Why"
        ])

    safe_cols = [c for c in all_tickers if c in daily_prices.columns]
    missing = [c for c in all_tickers if c not in daily_prices.columns]
    if missing:
        preview = missing[:10]
        if len(missing) > 10:
            preview = preview + ["…"]
        logging.warning("Explainability: missing price columns for %s", preview)

    if safe_cols:
        prices = daily_prices.loc[:, safe_cols].copy()
    else:
        prices = daily_prices.iloc[:, :0].copy()

    prices = prices.dropna(axis=1, how="all")
    if prices.empty:
        msg = "Explainability unavailable: no overlapping price history for selected tickers."
        if _HAS_ST:
            st.warning(msg)
        else:
            logging.warning(msg)
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
        delta_bps = int(round((new_w - old_w) * 10000))
        rows.append({
            "Ticker": t,
            "Action": action,
            "Old Wt": old_w,
            "New Wt": new_w,
            "Δ Wt (bps)": delta_bps,
            "Mom Rank": int(mom_rank) if pd.notna(mom_rank) else np.nan,
            "Mom Score": mom_score,
            f"ST Return ({params['mr_lb']}d)": stv,
            f"Above {params['mr_ma']}DMA": bool(ab) if pd.notna(ab) else None,
            "Why": _build_change_reason(action, delta_bps, mom_rank, mom_score, stv, ab)
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
@st.cache_data(ttl=43200)
def get_benchmark_series(ticker: str, start: str, end: str) -> pd.Series:
    """Fetch a benchmark price series with caching.

    Streamlit caches both successful results and raised exceptions. Instead of
    clearing the cache on failure, log the error and retry once to avoid
    transient outages.
    """
    for attempt in range(2):
        try:
            try:
                data = parallel_yf_download(ticker, start=start, end=end)
            except Exception:
                data = _yf_download(
                    ticker,
                    start=start,
                    end=end,
                )
            try:
                px = data["Close"]
            except Exception:
                # Fallback: take first column if "Close" not present (e.g., FRED series)
                px = data.iloc[:, 0] if hasattr(data, "iloc") else data
            px = _safe_series(px)
            return pd.Series(px).dropna()
        except Exception as e:
            st.warning(f"Failed to download {ticker} data: {e}")
            if attempt == 1:
                raise e

def _safe_last(series: pd.Series) -> float:
    if series is None:
        return np.nan
    ser = pd.Series(series).dropna()
    if ser.empty:
        return np.nan
    last = ser.iloc[-1]
    return float(last) if pd.notna(last) else np.nan


def _safe_last_vol(series: pd.Series, window: int = 10) -> float:
    if series is None:
        return np.nan
    ser = pd.Series(series).dropna()
    if ser.empty or ser.shape[0] <= window:
        return np.nan
    vol = ser.pct_change().rolling(window).std().iloc[-1]
    return float(vol) if pd.notna(vol) and vol > 0 else np.nan


def _vix_term_structure(vix1m: pd.Series, vix3m: pd.Series) -> float:
    v1 = _safe_last(vix1m)
    v3 = _safe_last(vix3m)
    if not (pd.notna(v1) and pd.notna(v3) and v1 > 0 and v3 > 0):
        return np.nan
    ratio = v3 / v1
    return ratio if 0.5 <= ratio <= 2.0 else np.nan


def _above_200dma(px: pd.Series) -> float:
    if px is None:
        return np.nan
    ser = pd.Series(px).dropna()
    if ser.shape[0] < REGIME_MA:
        return np.nan
    ma = ser.rolling(REGIME_MA).mean().iloc[-1]
    last = ser.iloc[-1]
    if not (pd.notna(ma) and pd.notna(last)):
        return np.nan
    return float(last > ma)


def _metric_or_default(metrics: Dict[str, float], key: str, default: float) -> float:
    val = metrics.get(key, np.nan)
    if pd.notna(val):
        try:
            return float(val)
        except Exception:
            return default
    return default


def compute_regime_metrics(universe_prices_daily: pd.DataFrame) -> Dict[str, float]:
    """Enhanced regime metrics calculation"""
    if universe_prices_daily.empty:
        return {}

    start = (universe_prices_daily.index.min() - pd.DateOffset(days=5)).strftime("%Y-%m-%d")
    end = (universe_prices_daily.index.max() + pd.DateOffset(days=5)).strftime("%Y-%m-%d")

    def _safe_fetch(ticker: str) -> pd.Series:
        try:
            return get_benchmark_series(ticker, start, end).reindex(universe_prices_daily.index).ffill()
        except Exception:
            logging.info("Benchmark fetch failed for %s", ticker)
            return pd.Series(dtype=float)

    qqq = _safe_fetch("QQQ").dropna()
    if qqq.empty:
        logging.info("Regime metrics missing QQQ series; attempting SPY fallback")
        qqq = _safe_fetch("SPY").dropna()
    if qqq.empty:
        logging.warning("Regime metrics could not source QQQ or SPY; hedge inputs will be NaN")

    vix = _safe_fetch("^VIX")
    vix3m = _safe_fetch("^VIX3M")
    hy_oas = _safe_fetch("BAMLH0A0HYM2")

    vix_ts = _vix_term_structure(vix, vix3m)
    hy_oas_last = _safe_last(hy_oas)

    # Universe breadth above 200DMA (ignore tickers without sufficient history)
    above_flags: list[float] = []
    for col in universe_prices_daily.columns:
        flag = _above_200dma(universe_prices_daily[col])
        if pd.notna(flag):
            above_flags.append(flag)
    pct_above_ma = float(np.mean(above_flags)) if above_flags else np.nan

    qqq_above_ma = _above_200dma(qqq)
    qqq_vol_10d = _safe_last_vol(qqq, window=10)
    qqq_slope_50 = np.nan
    if not qqq.empty:
        ser = qqq.dropna()
        if ser.shape[0] > 60:
            ma50 = ser.rolling(50).mean()
            tail = ma50.iloc[-10:]
            if tail.dropna().shape[0] == 10 and pd.notna(ma50.iloc[-1]) and pd.notna(ma50.iloc[-10]):
                qqq_slope_50 = float(ma50.iloc[-1] / ma50.iloc[-10] - 1)

    monthly = universe_prices_daily.resample("M").last()
    pos_6m = np.nan
    if len(monthly) >= 7:
        valid_cols = monthly.notna().sum() >= 7
        six_m = monthly.pct_change(6).iloc[-1]
        six_m = six_m[valid_cols.reindex(six_m.index).fillna(False)]
        six_m = six_m.dropna()
        if not six_m.empty:
            pos_6m = float((six_m > 0).mean())

    metrics = {
        "universe_above_200dma": pct_above_ma,
        "qqq_above_200dma": qqq_above_ma,
        "qqq_vol_10d": qqq_vol_10d,
        "breadth_pos_6m": pos_6m,
        "qqq_50dma_slope_10d": qqq_slope_50,
        "vix_term_structure": vix_ts,
        "hy_oas": hy_oas_last,
    }

    used_metrics = {k: v for k, v in metrics.items() if pd.notna(v)}
    logging.info(
        "REGIME metrics computed | vol=%s | breadth=%s | qqq>200=%s | vix_ts=%s | hy_oas=%s (used=%d)",
        metrics.get("qqq_vol_10d"),
        metrics.get("breadth_pos_6m"),
        metrics.get("qqq_above_200dma"),
        metrics.get("vix_term_structure"),
        metrics.get("hy_oas"),
        len(used_metrics),
    )

    return metrics

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

        scoring_inputs = {
            "vol": metrics.get("qqq_vol_10d", np.nan),
            "breadth": metrics.get("breadth_pos_6m", np.nan),
            "trend": metrics.get("qqq_above_200dma", np.nan),
            "vix_ts": metrics.get("vix_term_structure", np.nan),
        }
        valid = {k: v for k, v in scoring_inputs.items() if pd.notna(v)}
        logging.info(
            "REGIME inputs | vol=%s | breadth=%s | qqq>200=%s | vix_ts=%s (used=%d)",
            scoring_inputs["vol"],
            scoring_inputs["breadth"],
            scoring_inputs["trend"],
            scoring_inputs["vix_ts"],
            len(valid),
        )

        # Enhanced labeling with more nuanced categories
        breadth = scoring_inputs["breadth"]
        qqq_abv = scoring_inputs["trend"]
        vol10 = scoring_inputs["vol"]

        if len(valid) < 3:
            label = "Neutral"
        else:
            qqq_above_flag = pd.notna(qqq_abv) and qqq_abv >= 1.0
            breadth_gt = lambda threshold: pd.notna(breadth) and breadth > threshold
            vol_gt = lambda threshold: pd.notna(vol10) and vol10 > threshold
            vol_lt = lambda threshold: pd.notna(vol10) and vol10 < threshold

            if qqq_above_flag and breadth_gt(0.65) and vol_lt(0.025):
                label = "Strong Risk-On"
            elif qqq_above_flag and breadth_gt(0.50):
                label = "Risk-On"
            elif qqq_above_flag and breadth_gt(0.35):
                label = "Cautious Risk-On"
            elif (pd.notna(qqq_abv) and qqq_abv < 1.0) and breadth_gt(0.45):
                label = "Mixed"
            elif (pd.notna(qqq_abv) and qqq_abv < 1.0) and breadth_gt(0.35):
                label = "Risk-Off"
            elif vol_gt(0.045):
                label = "High Volatility Risk-Off"
            else:
                label = "Extreme Risk-Off"

        return label, metrics
    except Exception:
        return "Neutral", {}

def select_optimal_universe(as_of: date | None = None) -> str:
    """Automatically choose the trading universe based on recent index momentum.

    Uses 3-month returns of ``SPY`` and ``QQQ`` as proxies for S&P 500 and
    NASDAQ 100 respectively.  When NASDAQ outperforms the S&P by more than 2%,
    the function selects ``"NASDAQ100+"``.  When the S&P is positive but not
    meaningfully lagging, ``"S&P500 (All)"`` is chosen.  Otherwise the more
    defensive ``"Hybrid Top150"`` subset is returned.
    """

    asof_ts = pd.Timestamp(as_of or date.today())
    start = (asof_ts - relativedelta(months=3)).strftime("%Y-%m-%d")
    end = asof_ts.strftime("%Y-%m-%d")

    # Proxy ETFs for each universe
    etfs = {"NASDAQ100+": "QQQ", "S&P500 (All)": "SPY", "Hybrid Top150": "SPY"}
    try:
        try:
            raw = parallel_yf_download(
                list(set(etfs.values())),
                start=start,
                end=end,
            )
        except Exception:
            raw = _yf_download(
                list(set(etfs.values())),
                start=start,
                end=end,
            )

        data = raw["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
    except Exception:
        return "Hybrid Top150"

    returns: Dict[str, float] = {}
    for label, ticker in etfs.items():
        if ticker in data.columns:
            try:
                ret = float(data[ticker].iloc[-1] / data[ticker].iloc[0] - 1)
                returns[label] = ret
            except Exception:
                continue

    qqq_ret = returns.get("NASDAQ100+", -np.inf)
    spy_ret = returns.get("S&P500 (All)", -np.inf)

    if qqq_ret > 0 and (qqq_ret - spy_ret) > 0.02:
        return "NASDAQ100+"
    elif spy_ret > 0:
        return "S&P500 (All)"
    else:
        return "Hybrid Top150"

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

    # Automatically determine optimal universe for the upcoming period
    chosen_universe = select_optimal_universe(asof_ts)
    st.session_state["universe"] = chosen_universe

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
    try:
        log = load_assess_log()
        new_row = pd.DataFrame([
            {
                "date": asof_ts,
                "metrics": json.dumps(metrics, default=float),
                "settings": json.dumps(settings, default=float),
            }
        ])
        log = pd.concat([log, new_row], ignore_index=True)
        log = log.drop_duplicates(subset=["date"], keep="last")
        save_assess_log(log)
    except Exception as e:
        logging.warning("P03 failed to save assessment log", exc_info=True)

    return {"metrics": metrics, "settings": settings, "universe": chosen_universe}

# =========================
# Assessment Logging
# =========================
def load_assess_log() -> pd.DataFrame:
    if GIST_API_URL and GITHUB_TOKEN:
        try:
            resp = requests.get(GIST_API_URL, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            files = resp.json().get("files", {})
            content = files.get(ASSESS_LOG_FILE, {}).get("content", "")
            if content:
                df = pd.read_csv(io.StringIO(content))
                df["date"] = pd.to_datetime(df["date"])
                return df
        except Exception as e:
            logging.warning("P04 failed to load assessment log", exc_info=True)
    return pd.DataFrame(columns=["date", "metrics", "settings", "portfolio_ret", "benchmark_ret"])

def save_assess_log(df: pd.DataFrame) -> None:
    if not GIST_API_URL or not GITHUB_TOKEN:
        return
    try:
        csv_str = df.to_csv(index=False)
        payload = {"files": {ASSESS_LOG_FILE: {"content": csv_str}}}
        resp = requests.patch(GIST_API_URL, headers=HEADERS, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        if _HAS_ST:
            st.sidebar.warning(f"Could not save assessment log: {e}")
        else:
            logging.warning("Could not save assessment log: %s", e)

def record_assessment_outcome(as_of: date | None = None,
                              benchmark: str = "QQQ") -> Dict[str, Any]:
    asof_ts = pd.Timestamp(as_of or date.today()).normalize()
    end_ts = asof_ts + relativedelta(months=1)

    port_df = load_previous_portfolio()
    if port_df is None or "Weight" not in port_df.columns:
        return {"ok": False, "msg": "No portfolio data"}

    weights = port_df["Weight"].astype(float)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    tickers = list(weights.index)

    px = fetch_market_data(tickers + [benchmark],
                           asof_ts.strftime("%Y-%m-%d"),
                           end_ts.strftime("%Y-%m-%d"))
    if px.empty or benchmark not in px.columns or len(px.index) < 2:
        return {"ok": False, "msg": "Insufficient price data"}

    prices = px[[*tickers, benchmark]].dropna()
    if prices.empty or len(prices) < 2:
        return {"ok": False, "msg": "Insufficient price data"}

    port_prices = prices[tickers]
    port_rets = port_prices.iloc[-1] / port_prices.iloc[0] - 1
    portfolio_ret = float((port_rets * weights.reindex(port_prices.columns).fillna(0)).sum())
    benchmark_ret = float(prices[benchmark].iloc[-1] / prices[benchmark].iloc[0] - 1)

    log = load_assess_log()
    mask = log["date"] == asof_ts
    if mask.any():
        log.loc[mask, "portfolio_ret"] = portfolio_ret
        log.loc[mask, "benchmark_ret"] = benchmark_ret
    else:
        new_row = {
            "date": asof_ts,
            "metrics": json.dumps({}, default=float),
            "settings": json.dumps({}, default=float),
            "portfolio_ret": portfolio_ret,
            "benchmark_ret": benchmark_ret,
        }
        log = pd.concat([log, pd.DataFrame([new_row])], ignore_index=True)

    save_assess_log(log)
    return {"ok": True, "portfolio_ret": portfolio_ret, "benchmark_ret": benchmark_ret}

def evaluate_assessment_accuracy(log: pd.DataFrame | None = None) -> Dict[str, Any]:
    """Evaluate accuracy of past market assessments.

    This reads the assessment log where each entry contains the
    portfolio and benchmark returns that followed a given assessment
    date.  It computes whether the portfolio outperformed the benchmark
    ("correct"), the alpha for each assessment, and aggregates summary
    statistics for display.

    Parameters
    ----------
    log : pd.DataFrame, optional
        Preloaded assessment log.  If ``None`` the log is loaded using
        :func:`load_assess_log`.

    Returns
    -------
    dict
        Dictionary containing the original log augmented with ``alpha``
        and ``correct`` columns along with summary metrics:
        ``hit_rate`` (share of assessments where the portfolio
        outperformed), ``avg_alpha`` (average portfolio minus benchmark
        return) and ``confusion_matrix`` (2x2 counts of portfolio
        positive/negative vs. benchmark positive/negative).
    """

    # Load log if not provided
    if log is None:
        log = load_assess_log()

    # Ensure required columns exist
    required_cols = {"portfolio_ret", "benchmark_ret"}
    if log.empty or not required_cols.issubset(log.columns):
        return {
            "history": pd.DataFrame(columns=["date", "portfolio_ret", "benchmark_ret", "alpha", "correct"]),
            "hit_rate": np.nan,
            "avg_alpha": np.nan,
            "confusion_matrix": pd.DataFrame(),
        }

    df = log.dropna(subset=["portfolio_ret", "benchmark_ret"]).copy()
    if df.empty:
        return {
            "history": pd.DataFrame(columns=["date", "portfolio_ret", "benchmark_ret", "alpha", "correct"]),
            "hit_rate": np.nan,
            "avg_alpha": np.nan,
            "confusion_matrix": pd.DataFrame(),
        }

    # Calculate alpha and correctness
    df["alpha"] = df["portfolio_ret"].astype(float) - df["benchmark_ret"].astype(float)
    df["correct"] = df["alpha"] > 0

    # Summary statistics
    hit_rate = df["correct"].mean() if len(df) else np.nan
    avg_alpha = df["alpha"].mean() if len(df) else np.nan

    # Confusion matrix of portfolio vs benchmark positive returns
    port_pos = df["portfolio_ret"] > 0
    bench_pos = df["benchmark_ret"] > 0
    confusion = pd.crosstab(port_pos, bench_pos).reindex(index=[False, True], columns=[False, True], fill_value=0)
    confusion = confusion.rename(index={False: "port_down", True: "port_up"},
                                 columns={False: "bench_down", True: "bench_up"})

    summary = {
        "history": df[["date", "portfolio_ret", "benchmark_ret", "alpha", "correct"]],
        "hit_rate": float(hit_rate) if pd.notna(hit_rate) else np.nan,
        "avg_alpha": float(avg_alpha) if pd.notna(avg_alpha) else np.nan,
        "confusion_matrix": confusion,
    }

    return summary

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
# TRUST CHECKS (Signal, Construction, Health)
# =========================
def summarize_signal_alignment(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Quick pass/fail style view for regime vs market reality."""
    if not metrics:
        return {"ok": False, "reason": "no-metrics", "checks": {}}

    breadth    = float(metrics.get("breadth_pos_6m", np.nan))
    qqq_above  = float(metrics.get("qqq_above_200dma", np.nan))
    vol10      = float(metrics.get("qqq_vol_10d", np.nan))
    vix_ts     = float(metrics.get("vix_term_structure", np.nan))
    regime_lbl = str(metrics.get("regime", "Unknown"))

    checks = {
        "qqq_above_200dma": (qqq_above >= 1.0),
        "breadth_ok": (not np.isnan(breadth) and breadth >= 0.50),
        "vol_ok": (not np.isnan(vol10) and vol10 < 0.03),
        "vix_ts_ok": (not np.isnan(vix_ts) and vix_ts >= 1.0),  # contango-ish
    }

    # If labeled Strong Risk-On, require stricter tests
    if "strong" in regime_lbl.lower():
        checks["breadth_strong"] = (not np.isnan(breadth) and breadth >= 0.60)
        checks["vol_strong"] = (not np.isnan(vol10) and vol10 < 0.025)

    ok = all(v is True for v in checks.values())
    return {"ok": ok, "reason": regime_lbl, "checks": checks}

def summarize_portfolio_construction(
    weights: pd.Series | None,
    sectors_map: Dict[str, str] | None,
    name_cap: float,
    sector_cap: float,
    turnover_series: pd.Series | None = None,
) -> Dict[str, Any]:
    """Caps, exposure, turnover sanity."""
    out = {"ok": True, "issues": [], "stats": {}}
    if weights is None or len(weights) == 0:
        out["ok"] = False
        out["issues"].append("no-weights")
        return out

    w = pd.Series(weights).dropna().astype(float)
    total_exp = float(w.sum())
    max_w = float(w.max()) if len(w) else 0.0

    # Constraint violations (uses your existing checker)
    violations = []
    if sectors_map:
        try:
            violations = check_constraint_violations(
                w, sectors_map, name_cap=name_cap, sector_cap=sector_cap, group_caps=None
            )
        except Exception:
            violations = []
    if violations:
        out["ok"] = False
        out["issues"].extend(violations)

    # Turnover — use last 12M mean if available
    tpy = None
    if turnover_series is not None and len(turnover_series) > 0:
        try:
            ts = pd.Series(turnover_series).dropna()
            if len(ts) >= 6:
                tpy = float(ts.tail(12).sum() / max(1, len(ts.tail(12)) / 1.0))  # monthly sum avg
        except Exception:
            tpy = None

    out["stats"] = {
        "total_equity_exposure": total_exp,
        "max_name_weight": max_w,
        "avg_turnover_last_12m": tpy,
    }

    # Heuristics
    if total_exp <= 0.50:
        out["issues"].append("low-exposure")
        out["ok"] = False
    if max_w > max(0.35, name_cap + 0.10):  # hard stop if crazy
        out["issues"].append(f"max-name-weight {max_w:.2%} too high")
        out["ok"] = False
    if tpy is not None and tpy > 1.0:  # >100% monthly (0.5*L1 def)
        out["issues"].append("excessive-turnover")
        out["ok"] = False

    return out

def summarize_health(
    monthly_returns: pd.Series,
    benchmark_monthly_returns: pd.Series | None = None,
) -> Dict[str, Any]:
    """Wraps your existing health calc + adds rolling correlation guard."""
    if monthly_returns is None or len(monthly_returns) == 0:
        return {"ok": False, "issues": ["no-returns"], "stats": {}}

    stats = get_strategy_health_metrics(monthly_returns, benchmark_monthly_returns)

    corr = stats.get("benchmark_correlation", np.nan)
    rolling_ok = (np.isnan(corr) or corr < 0.85)  # not a closet tracker
    issues = []
    if not rolling_ok:
        issues.append("high-benchmark-correlation")

    # Drawdown guardrails
    dd = stats.get("current_drawdown", 0.0)
    if dd < -0.30:
        issues.append("large-drawdown")

    ok = len(issues) == 0
    return {"ok": ok, "issues": issues, "stats": stats}

def run_trust_checks(
    weights_df: pd.DataFrame | None,
    metrics: Dict[str, float] | None,
    turnover_series: pd.Series | None,
    name_cap: float,
    sector_cap: float,
) -> Dict[str, Any]:
    """Convenience wrapper used by the app tab."""
    # Weights
    weights = None
    sectors_map = None
    try:
        if weights_df is not None and "Weight" in weights_df.columns:
            weights = weights_df["Weight"].astype(float)
            base_map = get_sector_map(list(weights.index))
            sectors_map = get_enhanced_sector_map(list(weights.index), base_map=base_map)
    except Exception:
        pass

    # 1) Signal alignment
    sig = summarize_signal_alignment(metrics or {})

    # 2) Portfolio construction
    constr = summarize_portfolio_construction(weights, sectors_map, name_cap, sector_cap, turnover_series)

    # 3) Health (use the same series you show in Performance tab)
    # Pull what the app saved in session
    base_cum = st.session_state.get("strategy_cum_net")
    if base_cum is None:
        base_cum = st.session_state.get("strategy_cum_gross")
    qqq_cum  = st.session_state.get("qqq_cum")

    def _to_monthly(series):
        if series is None or len(series) == 0:
            return pd.Series(dtype=float)
        r = pd.Series(series).pct_change().dropna()
        r.index = pd.to_datetime(r.index, errors="coerce")
        r = r[~r.index.isna()]
        return (1 + r).resample("M").prod() - 1

    health = summarize_health(
        _to_monthly(base_cum),
        _to_monthly(qqq_cum) if qqq_cum is not None else None
    )

    score = sum(int(x["ok"]) for x in (sig, constr, health))
    return {"score": score, "signal": sig, "construction": constr, "health": health}

# =========================
# Live performance tracking (Enhanced)
# =========================
def load_live_perf() -> pd.DataFrame:
    if GIST_API_URL and GITHUB_TOKEN:
        try:
            resp = requests.get(GIST_API_URL, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            files = resp.json().get("files", {})
            content = files.get(LIVE_PERF_FILE, {}).get("content", "")
            if not content:
                return pd.DataFrame(columns=["date","strat_ret","qqq_ret","note"])
            df = pd.read_csv(io.StringIO(content))
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            logging.warning("P05 failed to load live performance", exc_info=True)
    return pd.DataFrame(columns=["date","strat_ret","qqq_ret","note"])

def save_live_perf(df: pd.DataFrame) -> None:
    if not GIST_API_URL or not GITHUB_TOKEN:
        return
    try:
        csv_str = df.to_csv(index=False)
        payload = {"files": {LIVE_PERF_FILE: {"content": csv_str}}}
        resp = requests.patch(GIST_API_URL, headers=HEADERS, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        if _HAS_ST:
            st.sidebar.warning(f"Could not save live perf: {e}")
        else:
            logging.warning("Could not save live perf: %s", e)

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
                               block_size: int = 3,
                               seed: int | None = 42) -> Dict:
    """Enhanced Monte Carlo projections with regime awareness.

    Parameters
    ----------
    historical_returns : pd.Series
        Series of historical monthly returns.
    n_scenarios : int, optional
        Number of Monte Carlo scenarios to generate.
    horizon_months : int, optional
        Projection horizon in months.
    confidence_levels : List[int], optional
        Percentiles to compute from the simulated distribution.
    block_size : int, optional
        Size of blocks used in block bootstrap.
    seed : int | None, optional
        Seed for the random number generator. If ``None``, the generator is
        initialized without a seed for non-deterministic results.
    """
    
    if len(historical_returns) < 12:
        return {"error": "Insufficient historical data for Monte Carlo"}
    
    # Clean returns
    returns_clean = historical_returns.dropna()
    if len(returns_clean) < 6:
        return {"error": "Insufficient clean returns data"}
    
    # Block bootstrap to preserve short-term correlation
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
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
