"""Shared strategy core for Hybrid Momentum + Mean-Reversion.

This module is *framework-agnostic* (no Streamlit, no Colab-only bits).
Both your Streamlit backend (production) and your Colab sandbox can import
from here to avoid code drift.

Key features
------------
- Universe building (NASDAQ-100 + optional extras)
- Market data fetching with yfinance
- Weight capping via iterative waterfall
- Momentum sleeve (top-N, positive-only)
- Mean-reversion sleeve (quality filter via long-term MA + worst short-term)
- Monthly backtest engines for each sleeve and for hybrid portfolios
- Metrics helper that uses quantstats if available, otherwise a clean fallback
- Optional volatility targeting and transaction-cost modeling

All functions are deterministic and return pandas objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from io import StringIO

# ------------------------------
# 1) Universe & Data
# ------------------------------

def get_nasdaq_100_plus_tickers(
    extras: Optional[Iterable[str]] = None,
    wikipedia_url: str = "https://en.wikipedia.org/wiki/Nasdaq-100",
) -> List[str]:
    """Return current NASDAQ-100 tickers plus optional extras.

    Notes
    -----
    - Uses Wikipedia by default; this implies survivorship bias if used for history.
    - For point-in-time accuracy, replace this with a provider that offers
      historical constituents.
    """
    extras = list(extras) if extras else []
    try:
        resp = requests.get(wikipedia_url, headers={'User-Agent': 'Mozilla/5.0'})
        tables = pd.read_html(StringIO(resp.text))
        df = next(
            t for t in tables
            if any(col.lower() in {"ticker", "symbol"} for col in map(str, t.columns))
        )
        col = "Ticker" if "Ticker" in df.columns else "Symbol"
        tickers = df[col].astype(str).str.upper().str.strip().tolist()
        if "SQ" in extras:
            extras = [x for x in extras if x != "SQ"]  # acquired / renamed
        full = sorted(set(tickers + extras))
        return full
    except Exception:
        # On failure, just return extras (maybe user passes their own universe)
        return sorted(set(extras))


def fetch_market_data(tickers: Iterable[str],
                      start_date: str,
                      end_date: Optional[str] = None) -> pd.DataFrame:
    """Fetch daily split/dividend-adjusted close prices for tickers using yfinance.

    Parameters
    ----------
    tickers : list-like of str
    start_date : YYYY-MM-DD
    end_date : YYYY-MM-DD or None (None = today)

    Returns
    -------
    DataFrame indexed by date (daily), columns=tickers.

    Notes
    -----
    Prices are adjusted for splits and dividends via ``auto_adjust=True``.
    """
    if not tickers:
        return pd.DataFrame()
    data = yf.download(list(tickers), start=start_date, end=end_date, auto_adjust=True, progress=False)
    if isinstance(data, pd.DataFrame) and "Close" in data:
        df = data["Close"]
    else:
        if isinstance(data, pd.Series):
            df = data.to_frame(name="Close")
        else:
            try:
                df = data.xs("Close", axis=1, level=0)
            except Exception:
                return pd.DataFrame()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if df.shape[1] == 1 and len(list(tickers)) == 1:
        df.columns = [list(tickers)[0]]
    return df.dropna(how="all", axis=1)


# ------------------------------
# 2) Portfolio Utilities
# ------------------------------

def cap_weights(weights: pd.Series, cap: float = 0.25, max_iter: int = 100,
                tol: float = 1e-12) -> pd.Series:
    """Iterative waterfall cap. Preserves proportionality below cap.

    If all names are at cap and excess remains, distributes evenly to avoid
    infinite loops. After capping, weights are renormalized if their sum
    deviates from 1 by more than ``tol``.
    """
    w = weights.copy().astype(float)
    if (w < 0).any():
        raise ValueError("Weights must be non-negative.")
    if w.sum() == 0:
        return w
    w = w / w.sum()
    for _ in range(max_iter):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if w[under].sum() > 0:
            w[under] += w[under] / w[under].sum() * excess
        else:
            # All names at cap; spread excess uniformly
            w += excess / len(w)
    if abs(w.sum() - 1.0) > tol:
        w = w / w.sum()
    return w


def volatility_target(returns: pd.Series, target_vol_annual: Optional[float] = None,
                      periods_per_year: int = 12, min_leverage: float = 0.0,
                      max_leverage: float = 1.0, lookback: int = 12) -> pd.Series:
    """Scale monthly returns to a target annualized volatility using rolling estimate.

    If target_vol_annual is None, returns are unchanged.
    """
    if target_vol_annual is None:
        return returns
    rolling_vol = returns.rolling(lookback).std() * np.sqrt(periods_per_year)
    with np.errstate(divide='ignore', invalid='ignore'):
        lev = target_vol_annual / rolling_vol
    lev = lev.clip(lower=min_leverage, upper=max_leverage).fillna(0)
    return returns * lev


def apply_tc(returns: pd.Series, turnover: pd.Series, tc_bps: float = 10.0) -> pd.Series:
    """Apply simple transaction cost model in basis points per 100% turnover per rebalance.

    Turnover is defined as ``0.5 * |w - w_prev|_1`` at each rebalance.

    Example: tc_bps=10 means 10 bps * turnover deducted from that period's return.
    """
    drag = (tc_bps / 1e4) * turnover.fillna(0)
    return returns - drag


# ------------------------------
# 3) Signal Engines (Monthly)
# ------------------------------

def _resample_month_end(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.resample("M").last()


def momentum_signals(monthly_prices: pd.DataFrame, lookback_m: int) -> pd.Series:
    """Return last-available momentum score per ticker at month-end (percentage change over lookback_m)."""
    mom = monthly_prices.pct_change(periods=lookback_m)
    return mom.iloc[-1].dropna()


def build_momentum_weights(monthly_prices: pd.DataFrame, lookback_m: int, top_n: int,
                           cap: float) -> Tuple[pd.Series, pd.Series]:
    """Compute momentum weights at the *last* month-end.

    Returns (weights, selected_scores) where weights sum to 1.
    """
    scores = momentum_signals(monthly_prices, lookback_m)
    scores = scores[scores > 0]
    if scores.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    top = scores.nlargest(top_n)
    raw = top / top.sum()
    capped = cap_weights(raw, cap=cap)
    final_w = capped / capped.sum()
    return final_w, top


def run_backtest_momentum(daily_prices: pd.DataFrame, lookback_m: int = 6,
                          top_n: int = 15, cap: float = 0.25) -> Tuple[pd.Series, pd.Series]:
    """Monthly backtest for the momentum sleeve.

    Returns
    -------
    (monthly_returns, monthly_turnover)

    Turnover uses 0.5 × L1 weight change between rebalances.
    """
    mp = _resample_month_end(daily_prices)
    future = mp.pct_change().shift(-1)
    mom = mp.pct_change(periods=lookback_m).shift(1)

    rets = pd.Series(index=mp.index, dtype=float)
    tno = pd.Series(index=mp.index, dtype=float)
    prev_w = None

    for dt in mp.index:
        scores = mom.loc[dt].dropna()
        scores = scores[scores > 0]
        if scores.empty:
            rets.loc[dt] = 0.0
            tno.loc[dt] = 0.0
            prev_w = None
            continue
        top = scores.nlargest(top_n)
        raw = top / top.sum()
        w = cap_weights(raw, cap=cap)
        w = w / w.sum()

        valid = w.index.intersection(future.columns)
        rets.loc[dt] = (future.loc[dt, valid] * w[valid]).sum()

        # Turnover as 0.5 * L1 weight change (prev_w = 0 if None)
        aligned_prev = prev_w.reindex(w.index).fillna(0) if prev_w is not None else pd.Series(0, index=w.index)
        tno.loc[dt] = 0.5 * (w - aligned_prev).abs().sum()
        prev_w = w

    return rets.fillna(0.0), tno.fillna(0.0)


def predictive_signals(monthly_prices: pd.DataFrame, lookback_m: int) -> pd.Series:
    """Predict next-month returns via simple AR(1) on monthly returns.

    For each ticker we estimate ``r_t = a + b * r_{t-1}`` over the
    ``lookback_m`` most recent returns and use the fitted parameters to
    forecast the next return.

    Parameters
    ----------
    monthly_prices : DataFrame
        Month-end prices for each ticker.
    lookback_m : int
        Number of months of returns to use in the regression.

    Returns
    -------
    Series
        Predicted next-month return for each ticker.  Missing values are
        omitted from the result.
    """
    rets = monthly_prices.pct_change().dropna()
    preds = {}
    if rets.empty:
        return pd.Series(dtype=float)
    for col in rets.columns:
        series = rets[col].dropna().iloc[-lookback_m:]
        if len(series) == 0:
            continue
        x = series.shift(1).dropna()
        y = series.loc[x.index]
        if len(x) == 0:
            continue
        X = np.vstack([np.ones(len(x)), x.values]).T
        try:
            a, b = np.linalg.lstsq(X, y.values, rcond=None)[0]
            preds[col] = a + b * series.iloc[-1]
        except Exception:
            continue
    return pd.Series(preds).dropna()


def build_predictive_weights(monthly_prices: pd.DataFrame, lookback_m: int,
                             top_n: int, cap: float) -> Tuple[pd.Series, pd.Series]:
    """Compute weights based on predicted returns from :func:`predictive_signals`.

    Returns ``(weights, selected_scores)`` where weights sum to 1.
    Only tickers with positive predicted returns are considered.
    """
    scores = predictive_signals(monthly_prices, lookback_m)
    scores = scores[scores > 0]
    if scores.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    top = scores.nlargest(top_n)
    raw = top / top.sum() if top.sum() != 0 else pd.Series(1.0 / len(top), index=top.index)
    capped = cap_weights(raw, cap=cap)
    final_w = capped / capped.sum()
    return final_w, top


def run_backtest_predictive(daily_prices: pd.DataFrame, lookback_m: int = 12,
                            top_n: int = 10, cap: float = 0.25) -> Tuple[pd.Series, pd.Series]:
    """Monthly backtest using predictive stock selection.

    At each month-end an AR(1) model is fit for each ticker using the last
    ``lookback_m`` monthly returns and the top ``top_n`` positive forecasts
    are selected.  Weights are capped via :func:`cap_weights`.

    Returns
    -------
    (monthly_returns, monthly_turnover)

    Turnover uses 0.5 × L1 weight change between rebalances.
    """
    mp = _resample_month_end(daily_prices)
    future = mp.pct_change().shift(-1)

    rets = pd.Series(index=mp.index, dtype=float)
    tno = pd.Series(index=mp.index, dtype=float)
    prev_w = None

    for dt in mp.index:
        hist = mp.loc[:dt]
        scores = predictive_signals(hist, lookback_m)
        scores = scores[scores > 0]
        if scores.empty:
            rets.loc[dt] = 0.0
            tno.loc[dt] = 0.0
            prev_w = None
            continue
        top = scores.nlargest(top_n)
        raw = top / top.sum()
        w = cap_weights(raw, cap=cap)
        w = w / w.sum()

        valid = w.index.intersection(future.columns)
        rets.loc[dt] = (future.loc[dt, valid] * w[valid]).sum()

        aligned_prev = prev_w.reindex(w.index).fillna(0) if prev_w is not None else pd.Series(0, index=w.index)
        tno.loc[dt] = 0.5 * (w - aligned_prev).abs().sum()
        prev_w = w

    return rets.fillna(0.0), tno.fillna(0.0)


def run_backtest_mean_reversion(daily_prices: pd.DataFrame, lookback_days: int = 21,
                                top_n: int = 5, long_ma_days: int = 200) -> Tuple[pd.Series, pd.Series]:
    """Monthly backtest for the mean-reversion sleeve with long-term uptrend filter.

    Returns
    -------
    (monthly_returns, monthly_turnover)
    """
    mp = _resample_month_end(daily_prices)
    future = mp.pct_change().shift(-1)

    short = daily_prices.pct_change(lookback_days).resample("M").last()
    trend = daily_prices.rolling(long_ma_days).mean().resample("M").last()

    rets = pd.Series(index=mp.index, dtype=float)
    tno = pd.Series(index=mp.index, dtype=float)
    prev_w = None

    for dt in mp.index:
        quality = mp.loc[dt] > trend.loc[dt]
        pool = quality[quality].index
        if len(pool) == 0:
            rets.loc[dt] = 0.0
            tno.loc[dt] = 0.0
            prev_w = None
            continue
        candidates = short.loc[dt, pool].dropna()
        if candidates.empty:
            rets.loc[dt] = 0.0
            tno.loc[dt] = 0.0
            prev_w = None
            continue
        picks = candidates.nsmallest(top_n)
        if picks.empty:
            rets.loc[dt] = 0.0
            tno.loc[dt] = 0.0
            prev_w = None
            continue
        w = pd.Series(1.0 / len(picks), index=picks.index)
        valid = w.index.intersection(future.columns)
        rets.loc[dt] = (future.loc[dt, valid] * w[valid]).sum()

        if prev_w is None:
            tno.loc[dt] = w.abs().sum()
        else:
            aligned_prev = prev_w.reindex(w.index).fillna(0)
            tno.loc[dt] = (w - aligned_prev).abs().sum()
        prev_w = w

    return rets.fillna(0.0), tno.fillna(0.0)


# ------------------------------
# 4) Hybrid & Benchmarks
# ------------------------------

def combine_hybrid(mom_rets: pd.Series, mr_rets: pd.Series,
                   mom_weight: float = 0.90, mr_weight: float = 0.10) -> pd.Series:
    """Linear blend of two monthly return series on a matched index."""
    idx = mom_rets.index.union(mr_rets.index)
    mom = mom_rets.reindex(idx).fillna(0)
    mr = mr_rets.reindex(idx).fillna(0)
    return mom * mom_weight + mr * mr_weight


def cumulative_growth(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()


# ------------------------------
# 5) Metrics
# ------------------------------

def get_performance_metrics(returns: pd.Series,
                            periods_per_year: int = 12,
                            use_quantstats: bool = True) -> Dict[str, str]:
    """Compute performance metrics; prefer quantstats if available.

    Returns a dict of formatted strings.
    """
    if returns is None or len(returns) < 2 or returns.isna().all():
        return {"Annual Return": "N/A", "Sharpe Ratio": "N/A", "Sortino Ratio": "N/A",
                "Calmar Ratio": "N/A", "Max Drawdown": "N/A"}

    if use_quantstats:
        try:
            import quantstats as qs
            ann = qs.stats.cagr(returns)
            sharpe = qs.stats.sharpe(returns, periods=periods_per_year)
            sortino = qs.stats.sortino(returns, periods=periods_per_year)
            calmar = qs.stats.calmar(returns)
            mdd = qs.stats.max_drawdown(returns)
            return {
                "Annual Return": f"{ann:.2%}",
                "Sharpe Ratio": f"{sharpe:.2f}",
                "Sortino Ratio": f"{sortino:.2f}",
                "Calmar Ratio": f"{calmar:.2f}",
                "Max Drawdown": f"{mdd:.2%}",
            }
        except Exception:
            # fall back below
            pass

    # Fallback metrics (monthly series expected)
    mean = returns.mean()
    std = returns.std()
    ann_ret = (1 + mean) ** periods_per_year - 1 if pd.notna(mean) else np.nan
    ann_vol = std * np.sqrt(periods_per_year) if pd.notna(std) else np.nan

    # Max drawdown
    curve = cumulative_growth(returns)
    peak = curve.cummax()
    mdd = ((curve - peak) / peak).min()

    # Sortino (downside std)
    downside = returns.copy()
    downside[downside > 0] = 0
    d_std = downside.std()
    sortino = (mean * periods_per_year) / (d_std * np.sqrt(periods_per_year) + 1e-9) if pd.notna(d_std) else np.nan

    # Calmar (ann ret / |mdd|)
    calmar = (ann_ret / abs(mdd)) if (pd.notna(ann_ret) and pd.notna(mdd) and mdd != 0) else np.nan

    return {
        "Annual Return": f"{ann_ret:.2%}" if pd.notna(ann_ret) else "N/A",
        "Sharpe Ratio": f"{(mean * periods_per_year) / (ann_vol + 1e-9):.2f}" if pd.notna(ann_vol) else "N/A",
        "Sortino Ratio": f"{sortino:.2f}" if pd.notna(sortino) else "N/A",
        "Calmar Ratio": f"{calmar:.2f}" if pd.notna(calmar) else "N/A",
        "Max Drawdown": f"{mdd:.2%}" if pd.notna(mdd) else "N/A",
    }


# ------------------------------
# 6) Convenience Runner
# ------------------------------
@dataclass
class HybridConfig:
    momentum_lookback_m: int = 6
    momentum_top_n: int = 15
    momentum_cap: float = 0.25
    mr_lookback_days: int = 21
    mr_top_n: int = 5
    mr_long_ma_days: int = 200
    mom_weight: float = 0.90
    mr_weight: float = 0.10
    target_vol_annual: Optional[float] = None  # e.g., 0.15 for 15% target
    tc_bps: float = 0.0  # per monthly rebalance; 10 = 10 bps


def run_hybrid_backtest(daily_prices: pd.DataFrame, cfg: HybridConfig = HybridConfig(),
                        apply_vol_target: bool = False) -> Dict[str, pd.Series]:
    """Run both sleeves, combine, and return a dict of useful Series.

    Keys: 'mom_rets', 'mom_turnover', 'mr_rets', 'mr_turnover',
          'hybrid_rets', 'hybrid_equity'
    """
    mom_rets, mom_tno = run_backtest_momentum(
        daily_prices,
        lookback_m=cfg.momentum_lookback_m,
        top_n=cfg.momentum_top_n,
        cap=cfg.momentum_cap,
    )

    mr_rets, mr_tno = run_backtest_mean_reversion(
        daily_prices,
        lookback_days=cfg.mr_lookback_days,
        top_n=cfg.mr_top_n,
        long_ma_days=cfg.mr_long_ma_days,
    )

    hybrid = combine_hybrid(mom_rets, mr_rets, cfg.mom_weight, cfg.mr_weight)

    # Apply TC
    if cfg.tc_bps != 0:
        # Approximate portfolio turnover as weighted average of sleeves (not perfect but serviceable)
        turnover = cfg.mom_weight * mom_tno.reindex(hybrid.index).fillna(0) + \
                   cfg.mr_weight * mr_tno.reindex(hybrid.index).fillna(0)
        hybrid = apply_tc(hybrid, turnover, tc_bps=cfg.tc_bps)

    # Optional vol targeting on combined series
    if apply_vol_target and cfg.target_vol_annual is not None:
        hybrid = volatility_target(hybrid, target_vol_annual=cfg.target_vol_annual)

    equity = cumulative_growth(hybrid)

    return {
        "mom_rets": mom_rets,
        "mom_turnover": mom_tno,
        "mr_rets": mr_rets,
        "mr_turnover": mr_tno,
        "hybrid_rets": hybrid,
        "hybrid_equity": equity,
    }
