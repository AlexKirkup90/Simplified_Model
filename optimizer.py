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
from dataclasses import replace
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from strategy_core import HybridConfig, run_hybrid_backtest


def _annualized_sharpe(returns: pd.Series, periods_per_year: int = 12) -> float:
    """Return annualized Sharpe ratio of a return series.

    If the standard deviation is zero or the series is empty, ``-inf`` is
    returned so such configurations are not selected.
    """
    if returns is None or len(returns) == 0:
        return float("-inf")
    std = returns.std()
    if std == 0 or np.isnan(std):
        return float("-inf")
    mean = returns.mean()
    return float(np.sqrt(periods_per_year) * mean / std)


def grid_search_hybrid(
    daily_prices: pd.DataFrame,
    param_grid: Dict[str, Iterable],
    base_cfg: HybridConfig | None = None,
) -> Tuple[HybridConfig, pd.DataFrame]:
    """Search over ``param_grid`` and return best config and results table.

    Parameters
    ----------
    daily_prices : DataFrame
        Daily price data used by :func:`run_hybrid_backtest`.
    param_grid : dict
        Mapping of ``HybridConfig`` field names to iterables of values.
    base_cfg : HybridConfig, optional
        Configuration to start from.  Defaults to ``HybridConfig()``.

    Returns
    -------
    best_cfg : HybridConfig
        Configuration with the highest Sharpe ratio.
    results : DataFrame
        One row per parameter combination with the evaluated Sharpe ratio.
    """
    base_cfg = base_cfg or HybridConfig()
    keys = list(param_grid.keys())
    best_score = float("-inf")
    best_cfg = base_cfg
    rows = []

    for combo in itertools.product(*param_grid.values()):
        params = dict(zip(keys, combo))
        cfg = replace(base_cfg, **params)
        res = run_hybrid_backtest(daily_prices, cfg)
        sharpe = _annualized_sharpe(res["hybrid_rets"])
        row = {**params, "sharpe": sharpe}
        rows.append(row)
        if sharpe > best_score:
            best_score = sharpe
            best_cfg = cfg

    results = pd.DataFrame(rows)
    return best_cfg, results
