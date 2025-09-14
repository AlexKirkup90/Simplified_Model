"""Simple grid-search optimizer for Hybrid strategy parameters.

This module provides a helper to run the existing ``run_hybrid_backtest``
function over a grid of parameters and select the combination that
maximizes the Sharpe ratio of the resulting portfolio.

Example
-------
>>> import pandas as pd
>>> prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
>>> from optimizer import grid_search_hybrid
>>> best_cfg, results = grid_search_hybrid(prices, {
...     "momentum_top_n": [10, 15, 20],
...     "momentum_cap": [0.20, 0.25],
... })
>>> best_cfg.momentum_top_n
10
"""
from __future__ import annotations

import itertools
from dataclasses import replace, fields
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from strategy_core import HybridConfig, run_hybrid_backtest


def _infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """Infer periods/year from a DateTime index."""
    if index is None or len(index) < 3:
        return 12.0  # default to monthly
    try:
        f = pd.infer_freq(index)
    except Exception:
        f = None
    if f:
        F = f.upper()
        if F.startswith(("B", "D")):
            return 252.0
        if F.startswith("W"):
            return 52.0
        if F.startswith("M"):
            return 12.0
        if F.startswith("Q"):
            return 4.0
        if F.startswith(("A", "Y")):
            return 1.0
    # fallback: median day spacing
    d = np.median(np.diff(index.view("i8"))) / 1e9 / 86400.0
    return 252.0 if d <= 2.5 else 52.0 if d <= 9 else 12.0 if d <= 45 else 4.0 if d <= 150 else 1.0


def _annualized_sharpe(returns: pd.Series, periods_per_year: float | None = None) -> float:
    """Annualized Sharpe, robust to NaNs/zero-std."""
    if returns is None or len(returns) == 0:
        return float("-inf")
    r = pd.Series(returns).dropna()
    if r.empty:
        return float("-inf")
    ppyr = periods_per_year or _infer_periods_per_year(r.index)
    std = r.std()
    if std == 0 or not np.isfinite(std):
        return float("-inf")
    mean = r.mean()
    return float(np.sqrt(ppyr) * mean / std)


def grid_search_hybrid(
    daily_prices: pd.DataFrame,
    param_grid: Dict[str, Iterable],
    base_cfg: HybridConfig | None = None,
    tc_bps: float = 0.0,
    apply_vol_target: bool = False,
) -> Tuple[HybridConfig, pd.DataFrame]:
    """Search over ``param_grid`` and return best config and results table."""
    base_cfg = base_cfg or HybridConfig()

    # Only allow fields that exist on HybridConfig
    cfg_fields = {f.name for f in fields(HybridConfig)}
    grid = {k: list(v) for k, v in param_grid.items() if k in cfg_fields}
    if not grid:
        grid = {"momentum_top_n": [base_cfg.momentum_top_n], "momentum_cap": [base_cfg.momentum_cap]}
    keys = list(grid.keys())

    best_score = float("-inf")
    best_cfg = base_cfg
    rows: list[dict] = []

    # Try all combos; skip ones that error gracefully
    for combo in itertools.product(*grid.values()):
        params = dict(zip(keys, combo))
        try:
            cfg = replace(base_cfg, **params, tc_bps=tc_bps)
            res = run_hybrid_backtest(daily_prices, cfg, apply_vol_target=apply_vol_target)
            rets = res.get("hybrid_rets_net", res.get("hybrid_rets"))
            if rets is None:
                raise ValueError("run_hybrid_backtest returned no hybrid returns.")
            ppyr = _infer_periods_per_year(pd.Index(rets.index))
            sharpe = _annualized_sharpe(pd.Series(rets).dropna(), periods_per_year=ppyr)
        except Exception as exc:
            row = {**params, "sharpe": float("-inf"), "error": str(exc)}
            rows.append(row)
            continue

        row = {**params, "sharpe": sharpe, "periods_per_year": ppyr}
        rows.append(row)
        if sharpe > best_score:
            best_score = sharpe
            best_cfg = cfg

    results = pd.DataFrame(rows)
    if not results.empty and "sharpe" in results.columns:
        results = results.sort_values("sharpe", ascending=False).reset_index(drop=True)
    return best_cfg, results
