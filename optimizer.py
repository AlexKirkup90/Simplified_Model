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
import random
from dataclasses import replace, fields
from typing import Dict, Iterable, Tuple, Any, Sequence

try:
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover - optional dependency fallback
    def delayed(func):
        def wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)

        return wrapper

    class Parallel:  # type: ignore[misc]
        def __init__(self, n_jobs: int | None = None, backend: str | None = None, batch_size: str | None = None):
            self.n_jobs = n_jobs
            self.backend = backend
            self.batch_size = batch_size

        def __call__(self, iterable: Iterable[Any]) -> list[Any]:
            return [task() for task in iterable]

import numpy as np
import pandas as pd

from strategy_core import HybridConfig, run_hybrid_backtest


def _get_perf_settings() -> Dict[str, Any]:
    """Fetch ``PERF`` tuning hints from :mod:`backend` with safe fallbacks."""

    defaults: Dict[str, Any] = {"parallel_grid": True, "n_jobs": 1}
    try:
        import backend as _backend  # type: ignore

        perf = getattr(_backend, "PERF", None)
        if isinstance(perf, dict):
            defaults.update(perf)
    except Exception:
        pass
    return defaults


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
    prices: pd.DataFrame,
    search_grid: Dict[str, Iterable],
    n_jobs: int | None = None,
    base_cfg: HybridConfig | None = None,
    tc_bps: float = 0.0,
    apply_vol_target: bool = False,
) -> Tuple[HybridConfig, pd.DataFrame]:
    """Search over ``search_grid`` and return best config and results table."""

    base_cfg = base_cfg or HybridConfig()

    perf = _get_perf_settings()
    n_jobs = n_jobs or (perf["n_jobs"] if perf.get("parallel_grid", True) else 1)

    cfg_fields = {f.name for f in fields(HybridConfig)}
    normalized_grid = {k: list(v) for k, v in search_grid.items() if k in cfg_fields}
    if not normalized_grid:
        normalized_grid = {
            "momentum_top_n": [base_cfg.momentum_top_n],
            "momentum_cap": [base_cfg.momentum_cap],
        }
    keys = list(normalized_grid.keys())
    grid_values = [normalized_grid[k] for k in keys]

    def _evaluate_combo(combo: Tuple[Any, ...]) -> Tuple[HybridConfig | None, Dict[str, Any]]:
        params = dict(zip(keys, combo))
        try:
            cfg = replace(base_cfg, **params, tc_bps=tc_bps)
            res = run_hybrid_backtest(prices, cfg, apply_vol_target=apply_vol_target)
            rets = res.get("hybrid_rets_net", res.get("hybrid_rets"))
            if rets is None:
                raise ValueError("run_hybrid_backtest returned no hybrid returns.")
            rets_series = pd.Series(rets).dropna()
            rets_index = getattr(rets, "index", rets_series.index)
            periods_per_year = _infer_periods_per_year(pd.Index(rets_index))
            sharpe = _annualized_sharpe(rets_series, periods_per_year=periods_per_year)
            row = {**params, "sharpe": sharpe, "periods_per_year": periods_per_year}
            return cfg, row
        except Exception as exc:  # pragma: no cover - defensive
            row = {**params, "sharpe": float("-inf"), "error": str(exc)}
            return None, row

    evaluations = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
        delayed(_evaluate_combo)(combo) for combo in itertools.product(*grid_values)
    )

    best_score = float("-inf")
    best_cfg = base_cfg
    rows: list[Dict[str, Any]] = []

    for candidate, row in evaluations:
        rows.append(row)
        sharpe = row.get("sharpe", float("-inf"))
        if candidate is not None and sharpe > best_score:
            best_score = sharpe
            best_cfg = candidate

    results = pd.DataFrame(rows)
    if not results.empty and "sharpe" in results.columns:
        results = results.sort_values("sharpe", ascending=False).reset_index(drop=True)
    return best_cfg, results


def _normalize_domain(name: str, values: Iterable[Any] | range | tuple[Any, ...] | None, base_value: Any) -> list[Any]:
    """Normalize a parameter domain to a list of python scalars."""
    if values is None:
        domain: list[Any] = []
    elif isinstance(values, range):
        domain = list(values)
    elif isinstance(values, tuple) and len(values) in (2, 3):
        lo, hi = values[:2]
        if len(values) == 3:
            step = values[2]
            if step == 0:
                domain = [lo]
            else:
                # ``np.arange`` handles float steps gracefully
                domain = list(np.arange(lo, hi + (step if step > 0 else -step), step))
        elif isinstance(lo, int) and isinstance(hi, int):
            step = 1 if hi >= lo else -1
            domain = list(range(lo, hi + step, step))
        else:
            num = max(int(abs(float(hi) - float(lo)) // 0.05), 5)
            domain = list(np.linspace(lo, hi, num=num))
    else:
        domain = list(values)

    if not domain:
        domain = [base_value]

    cleaned: list[Any] = []
    for val in domain:
        if isinstance(base_value, bool):
            cleaned.append(bool(val))
        elif isinstance(base_value, int) and not isinstance(base_value, bool):
            cleaned.append(int(round(float(val))))
        elif base_value is None:
            cleaned.append(val)
        else:
            cleaned.append(float(val))

    if base_value not in cleaned:
        cleaned.append(base_value)

    # preserve insertion order but drop duplicates
    seen: set[Any] = set()
    unique: list[Any] = []
    for val in cleaned:
        key = val if isinstance(val, (int, float, str)) or val is None else repr(val)
        if key in seen:
            continue
        seen.add(key)
        unique.append(val)
    return unique


def bayesian_optimize_hybrid(
    daily_prices: pd.DataFrame,
    search_space: Dict[str, Iterable[Any] | range | tuple[Any, ...]] | None = None,
    base_cfg: HybridConfig | None = None,
    tc_bps: float = 0.0,
    apply_vol_target: bool = False,
    population_size: int = 12,
    n_iter: int = 30,
    elite_fraction: float = 0.3,
    seed: int | None = None,
) -> Tuple[HybridConfig, pd.DataFrame]:
    """Evolutionary search for a Sharpe-maximising ``HybridConfig``.

    Parameters
    ----------
    daily_prices : DataFrame
        Daily price history used for backtesting the candidate configurations.
    search_space : dict, optional
        Mapping of ``HybridConfig`` field names to candidate values.  Values may
        be iterables, ranges, or ``(low, high[, step])`` tuples.  When omitted a
        sensible default search space spanning the key hybrid parameters is used.
    base_cfg : HybridConfig, optional
        Configuration providing defaults for unspecified fields.  By default the
        vanilla :class:`HybridConfig` is used.
    tc_bps : float, optional
        Trading cost applied during optimisation (per rebalance in basis points).
    apply_vol_target : bool, optional
        Whether to apply volatility targeting within ``run_hybrid_backtest``
        during evaluation.  Mirrors the behaviour of :func:`grid_search_hybrid`.
    population_size : int, optional
        Number of candidates maintained in the evolutionary population.
    n_iter : int, optional
        Number of evolutionary iterations to perform after the initial
        population has been evaluated.
    elite_fraction : float, optional
        Fraction of the population treated as the elite pool when generating
        offspring (default 30%).
    seed : int, optional
        Optional random seed for reproducibility.

    Returns
    -------
    best_cfg : HybridConfig
        The best performing configuration discovered.
    diagnostics : DataFrame
        Table containing the evaluated candidates, Sharpe scores and metadata.
    """

    base_cfg = replace(base_cfg or HybridConfig(), tc_bps=tc_bps)

    if daily_prices is None or daily_prices.empty:
        return base_cfg, pd.DataFrame()

    cfg_fields = {f.name for f in fields(HybridConfig)}
    default_space: Dict[str, Sequence[Any]] = {
        "momentum_top_n": range(6, 21),
        "momentum_cap": [0.15, 0.2, 0.25, 0.3, 0.35],
        "momentum_lookback_m": [3, 6, 9, 12],
        "mr_top_n": [2, 3, 4, 5, 6],
        "mr_lookback_days": [10, 15, 21, 30],
        "mr_long_ma_days": [150, 180, 200, 220, 250],
        "mom_weight": [0.55, 0.65, 0.75, 0.85, 0.9],
        "mr_weight": [0.05, 0.15, 0.25, 0.35, 0.45],
        "target_vol_annual": [base_cfg.target_vol_annual] if base_cfg.target_vol_annual is not None else [None],
        "predictive_weight": [0.0],
    }

    raw_space = default_space.copy()
    if search_space:
        for key, domain in search_space.items():
            if key not in cfg_fields:
                continue
            raw_space[key] = domain

    normalized_space: Dict[str, list[Any]] = {}
    for name, domain in raw_space.items():
        if name not in cfg_fields:
            continue
        normalized_space[name] = _normalize_domain(name, domain, getattr(base_cfg, name))

    # Ensure every optimised field includes at least the base value
    for name in cfg_fields:
        if name not in normalized_space:
            normalized_space[name] = [getattr(base_cfg, name)]

    variable_fields = [name for name, domain in normalized_space.items() if len(domain) > 1]
    if not variable_fields:
        return base_cfg, pd.DataFrame()

    rng = random.Random(seed)
    elite_size = max(1, int(population_size * elite_fraction))

    evaluated: Dict[Tuple[Tuple[str, Any], ...], float] = {}
    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_cfg = base_cfg

    def _evaluate(params: Dict[str, Any], iteration: int, origin: str) -> float:
        nonlocal best_score, best_cfg
        key = tuple(sorted(params.items()))
        if key in evaluated:
            return evaluated[key]

        cfg = replace(base_cfg, **params)
        try:
            res = run_hybrid_backtest(daily_prices, cfg, apply_vol_target=apply_vol_target)
            rets = res.get("hybrid_rets_net") or res.get("hybrid_rets")
            if rets is None:
                raise ValueError("run_hybrid_backtest returned no returns")
            rets = pd.Series(rets).dropna()
            if rets.empty:
                raise ValueError("No returns to evaluate")
            periods_per_year = _infer_periods_per_year(pd.Index(rets.index))
            score = _annualized_sharpe(rets, periods_per_year=periods_per_year)
            error = None
        except Exception as exc:  # pragma: no cover - handled gracefully
            periods_per_year = float("nan")
            score = float("-inf")
            error = str(exc)

        row = {name: params.get(name, getattr(base_cfg, name)) for name in variable_fields}
        row.update(
            {
                "iteration": iteration,
                "origin": origin,
                "evaluation": len(history) + 1,
                "sharpe": score,
                "periods_per_year": periods_per_year,
                "error": error,
            }
        )
        history.append(row)

        evaluated[key] = score
        if score > best_score:
            best_score = score
            best_cfg = cfg
        return score

    def _sample_candidate(source: Sequence[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name in variable_fields:
            domain = normalized_space[name]
            if source and rng.random() < 0.5:
                choice = rng.choice(source)[name]
            else:
                choice = rng.choice(domain)
            params[name] = choice
        return params

    population: list[Dict[str, Any]] = []
    scores: list[float] = []

    # Initial random population
    while len(population) < population_size:
        candidate = _sample_candidate()
        score = _evaluate(candidate, iteration=0, origin="init")
        population.append(candidate)
        scores.append(score)

    # Evolutionary loop
    for iteration in range(1, n_iter + 1):
        # Sort population by score descending
        order = np.argsort(scores)[::-1]
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        elite = population[:elite_size]

        parent_pool = elite if elite else population
        parent_a = rng.choice(parent_pool)
        parent_b = rng.choice(parent_pool)
        offspring = {}
        for name in variable_fields:
            domain = normalized_space[name]
            if rng.random() < 0.45:
                offspring[name] = parent_a[name]
            elif rng.random() < 0.9:
                offspring[name] = parent_b[name]
            else:
                offspring[name] = rng.choice(domain)

            # Occasional mutation
            if rng.random() < 0.2:
                offspring[name] = rng.choice(domain)

        score = _evaluate(offspring, iteration=iteration, origin="evolve")
        population.append(offspring)
        scores.append(score)
        # Keep population bounded
        if len(population) > population_size:
            worst = int(np.argmin(scores))
            population.pop(worst)
            scores.pop(worst)

    diagnostics = pd.DataFrame(history)
    if not diagnostics.empty:
        diagnostics = diagnostics.sort_values("evaluation").reset_index(drop=True)
    return best_cfg, diagnostics
