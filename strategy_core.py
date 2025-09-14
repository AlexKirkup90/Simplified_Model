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
from typing import Iterable, List, Dict, Optional, Tuple, Callable, Any

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from io import StringIO

# ------------------------------
# 1) Universe & Data
# ------------------------------

_YF_BATCH_SIZE = 200


def _chunk_tickers(tickers: List[str], size: int = _YF_BATCH_SIZE):
    tickers = list(tickers)
    for i in range(0, len(tickers), size):
        yield tickers[i : i + size]

_NDX_CONSTITUENT_CACHE: Dict[str, List[str]] = {}


def get_nasdaq_100_plus_tickers(
    extras: Optional[Iterable[str]] = None,
    as_of: Optional[str] = None,
    api_url: str = "https://data.nasdaq.com/api/v3/datatables/NASDAQ/NDX",
) -> List[str]:
    """Return NASDAQ-100 constituents for a given date plus optional extras.

    Parameters
    ----------
    extras : iterable of str, optional
        Additional tickers to include regardless of index membership.
    as_of : str or pandas-compatible date, optional
        Date for which constituents are requested (``YYYY-MM-DD``).  Defaults to
        today's date.
    api_url : str
        Endpoint for NASDAQ Data Link's datatable providing historical
        constituents.

    Notes
    -----
    Results are cached by date to avoid repeated network calls.
    """
    extras = list(extras) if extras else []
    date_str = pd.to_datetime(as_of or pd.Timestamp.today()).strftime("%Y-%m-%d")

    if date_str not in _NDX_CONSTITUENT_CACHE:
        try:
            params = {"date": date_str, "download": "true"}
            resp = requests.get(
                api_url,
                params=params,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))
            tickers = df["ticker"].astype(str).str.upper().str.strip().tolist()
        except Exception:
            tickers = []
        _NDX_CONSTITUENT_CACHE[date_str] = tickers

    tickers = _NDX_CONSTITUENT_CACHE.get(date_str, [])
    if "SQ" in extras:
        extras = [x for x in extras if x != "SQ"]  # acquired / renamed
    return sorted(set(tickers + extras))


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
    tickers = list(tickers)
    if not tickers:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for batch in _chunk_tickers(tickers):
        data = yf.download(batch, start=start_date, end=end_date, auto_adjust=True, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = [batch[0]]
        frames.append(data)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1)
    if df.shape[1] == 1 and len(tickers) == 1:
        df.columns = [tickers[0]]
    return df.dropna(how="all", axis=1)


# ------------------------------
# 2) Portfolio Utilities
# ------------------------------

def cap_weights(weights: pd.Series, cap: float = 0.25, max_iter: int = 100,
                tol: float = 1e-12) -> pd.Series:
    """Iterative waterfall cap. Preserves proportionality below cap.

    If every name breaches the cap (i.e. no "under-cap" names remain), the
    function stops redistributing and leaves residual cash rather than forcing
    further violations.  The final series is renormalized to sum to 1 only when
    doing so will not exceed the available cap capacity.
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
            w[under] += (w[under] / w[under].sum()) * excess
        else:
            # All names at cap -> no feasible redistribution; leave residual cash.
            break

    total_capacity = min(len(w), (w > 0).sum()) * cap
    if total_capacity >= 1 - 1e-12 and abs(w.sum() - 1.0) > tol:
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


def l1_turnover(prev_w: pd.Series | None, w: pd.Series) -> float:
    """0.5 × L1 distance between consecutive weight vectors.

    Aligns on the union of tickers so both additions and deletions are counted.
    """
    if prev_w is None:
        return 0.5 * w.abs().sum()
    union = w.index.union(prev_w.index)
    aligned_w = w.reindex(union, fill_value=0.0)
    aligned_prev = prev_w.reindex(union, fill_value=0.0)
    return 0.5 * (aligned_w - aligned_prev).abs().sum()


def run_backtest_momentum(
    daily_prices: pd.DataFrame,
    lookback_m: int = 6,
    top_n: int = 15,
    cap: float = 0.25,
    get_constituents: Optional[Callable[[pd.Timestamp], Iterable[str]]] = None,
) -> Tuple[pd.Series, pd.Series]:
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
        if get_constituents is not None:
            members = set(get_constituents(dt))
            scores = scores.loc[scores.index.intersection(members)]
        scores = scores[scores > 0]
        if scores.empty:
            rets.loc[dt] = 0.0
            if prev_w is not None:
                tno.loc[dt] = l1_turnover(prev_w, pd.Series(dtype=float))
            else:
                tno.loc[dt] = 0.0
            prev_w = None
            continue
        top = scores.nlargest(top_n)
        raw = top / top.sum()
        w = cap_weights(raw, cap=cap)

        valid = w.index.intersection(future.columns)
        if get_constituents is not None:
            valid = valid.intersection(members)
        rets.loc[dt] = (future.loc[dt, valid] * w[valid]).sum()
        tno.loc[dt] = l1_turnover(prev_w, w)
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


def run_backtest_predictive(
    daily_prices: pd.DataFrame,
    lookback_m: int = 12,
    top_n: int = 10,
    cap: float = 0.25,
    get_constituents: Optional[Callable[[pd.Timestamp], Iterable[str]]] = None,
) -> Tuple[pd.Series, pd.Series]:
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
        if get_constituents is not None:
            members = set(get_constituents(dt))
            scores = scores.loc[scores.index.intersection(members)]
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
        if get_constituents is not None:
            valid = valid.intersection(members)
        rets.loc[dt] = (future.loc[dt, valid] * w[valid]).sum()
        tno.loc[dt] = l1_turnover(prev_w, w)
        prev_w = w

    return rets.fillna(0.0), tno.fillna(0.0)


def run_backtest_mean_reversion(
    daily_prices: pd.DataFrame,
    lookback_days: int = 21,
    top_n: int = 5,
    long_ma_days: int = 200,
    get_constituents: Optional[Callable[[pd.Timestamp], Iterable[str]]] = None,
) -> Tuple[pd.Series, pd.Series]:
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
    prev_w: Optional[pd.Series] = None

    # Local fallback if a shared helper isn't available
    def _l1_turnover(prev_w_: Optional[pd.Series], w_: pd.Series) -> float:
        if prev_w_ is None or prev_w_.empty:
            # first rebalance: 0.5 * ||w - 0||_1 = 0.5 * sum(|w|)
            return float(0.5 * w_.abs().sum())
        union = w_.index.union(prev_w_.index)
        a = w_.reindex(union, fill_value=0.0)
        b = prev_w_.reindex(union, fill_value=0.0)
        return float(0.5 * (a - b).abs().sum())

    for dt in mp.index:
        quality = mp.loc[dt] > trend.loc[dt]
        if get_constituents is not None:
            members = set(get_constituents(dt))
            quality = quality.loc[quality.index.intersection(members)]
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
        if get_constituents is not None:
            valid = valid.intersection(members)
        rets.loc[dt] = float((future.loc[dt, valid] * w[valid]).sum())

        # Use shared helper if you have one; otherwise fallback
        try:
            # If strategy_core.l1_turnover is imported, prefer it
            tno.loc[dt] = l1_turnover(prev_w, w)  # type: ignore[name-defined]
        except Exception:
            tno.loc[dt] = _l1_turnover(prev_w, w)

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
    predictive_lookback_m: int = 12
    predictive_top_n: int = 10
    predictive_cap: float = 0.25
    predictive_weight: float = 0.0
    target_vol_annual: Optional[float] = None  # e.g., 0.15 for 15% target
    tc_bps: float = 0.0  # per monthly rebalance; 10 = 10 bps


def run_hybrid_backtest(
    daily_prices: pd.DataFrame,
    cfg: HybridConfig = HybridConfig(),
    apply_vol_target: bool = False,
    get_constituents: Optional[Callable[[pd.Timestamp], Iterable[str]]] = None,
) -> Dict[str, pd.Series]:
    """Run sleeves, blend, and return useful Series.

    Always runs momentum and mean-reversion sleeves.  If
    ``cfg.predictive_weight`` is greater than zero, the predictive sleeve is
    also evaluated and included in the final blend.
    """
    mom_rets, mom_tno = run_backtest_momentum(
        daily_prices,
        lookback_m=cfg.momentum_lookback_m,
        top_n=cfg.momentum_top_n,
        cap=cfg.momentum_cap,
        get_constituents=get_constituents,
    )

    mr_rets, mr_tno = run_backtest_mean_reversion(
        daily_prices,
        lookback_days=cfg.mr_lookback_days,
        top_n=cfg.mr_top_n,
        long_ma_days=cfg.mr_long_ma_days,
        get_constituents=get_constituents,
    )

    pred_rets = pd.Series(dtype=float)
    pred_tno = pd.Series(dtype=float)
    if cfg.predictive_weight > 0:
        pred_rets, pred_tno = run_backtest_predictive(
            daily_prices,
            lookback_m=cfg.predictive_lookback_m,
            top_n=cfg.predictive_top_n,
            cap=cfg.predictive_cap,
            get_constituents=get_constituents,
        )

    idx = mom_rets.index
    idx = idx.union(mr_rets.index)
    idx = idx.union(pred_rets.index)
    hybrid = (
        cfg.mom_weight * mom_rets.reindex(idx).fillna(0)
        + cfg.mr_weight * mr_rets.reindex(idx).fillna(0)
    )
    if cfg.predictive_weight > 0:
        hybrid += cfg.predictive_weight * pred_rets.reindex(idx).fillna(0)

    # Apply TC
    if cfg.tc_bps != 0:
        turnover = (
            cfg.mom_weight * mom_tno.reindex(idx).fillna(0)
            + cfg.mr_weight * mr_tno.reindex(idx).fillna(0)
        )
        if cfg.predictive_weight > 0:
            turnover += cfg.predictive_weight * pred_tno.reindex(idx).fillna(0)
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
        "pred_rets": pred_rets,
        "pred_turnover": pred_tno,
        "hybrid_rets": hybrid,
        "hybrid_equity": equity,
    }


def walk_forward_backtest(
    prices: pd.DataFrame,
    cfg: HybridConfig,
    train_years: int = 3,
    val_years: int = 1,
) -> Dict[str, Any]:
    """Run a simple walk-forward backtest and return average metrics.

    The data is split into consecutive ``train_years``/``val_years`` windows.
    For each window, :func:`run_hybrid_backtest` is evaluated on the combined
    period but only the validation slice is used to compute performance
    metrics.  Sharpe and Sortino ratios are computed for each validation
    slice and then averaged across all windows.

    Parameters
    ----------
    prices : DataFrame
        Daily price data for the universe.
    cfg : HybridConfig
        Configuration passed to :func:`run_hybrid_backtest`.
    train_years : int, default ``3``
        Length of the in-sample training window in years.
    val_years : int, default ``1``
        Length of the out-of-sample validation window in years.

    Returns
    -------
    dict
        Dictionary containing the average ``sharpe`` and ``sortino`` along
        with per-window metrics under ``windows``.
    """

    def _metrics(returns: pd.Series) -> Tuple[float, float]:
        if returns.empty:
            return float("nan"), float("nan")
        mean = returns.mean()
        std = returns.std()
        sharpe = (
            (mean * 12) / (std * np.sqrt(12) + 1e-9)
            if pd.notna(std) and std != 0
            else float("nan")
        )
        downside = returns.copy()
        downside[downside > 0] = 0
        d_std = downside.std()
        sortino = (
            (mean * 12) / (d_std * np.sqrt(12) + 1e-9)
            if pd.notna(d_std) and d_std != 0
            else float("nan")
        )
        return float(sharpe), float(sortino)

    start = prices.index.min()
    end = prices.index.max()
    start_win = start
    window_metrics: List[Dict[str, float]] = []

    while True:
        train_end = start_win + pd.DateOffset(years=train_years)
        val_end = train_end + pd.DateOffset(years=val_years)
        if val_end > end:
            break

        win_prices = prices.loc[start_win:val_end]
        res = run_hybrid_backtest(win_prices, cfg)
        val_rets = res["hybrid_rets"][res["hybrid_rets"].index >= train_end]
        sharpe, sortino = _metrics(val_rets)
        window_metrics.append({"sharpe": sharpe, "sortino": sortino})

        start_win += pd.DateOffset(years=val_years)

    avg_sharpe = float(np.nanmean([w["sharpe"] for w in window_metrics])) if window_metrics else float("nan")
    avg_sortino = float(np.nanmean([w["sortino"] for w in window_metrics])) if window_metrics else float("nan")

    return {"sharpe": avg_sharpe, "sortino": avg_sortino, "windows": window_metrics}
