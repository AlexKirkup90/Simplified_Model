"""Shared portfolio utilities used by both the Streamlit backend and core strategies."""
from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd

__all__ = ["cap_weights", "l1_turnover"]


def cap_weights(
    weights: pd.Series,
    cap: float = 0.25,
    *,
    vol_adjusted_caps: Optional[Mapping[str, float]] = None,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> pd.Series:
    """Iteratively cap position sizes while preserving proportionality below the cap.

    Parameters
    ----------
    weights : Series
        Raw, non-negative weights.
    cap : float
        Default maximum weight per name when ``vol_adjusted_caps`` is not supplied.
    vol_adjusted_caps : mapping, optional
        Optional per-ticker caps (e.g., volatility-adjusted limits). Values are
        interpreted as absolute caps (0--1). Any tickers absent from the mapping
        fall back to ``cap``.
    max_iter : int
        Maximum number of redistribution passes.
    tol : float
        Numerical tolerance used when assessing whether a renormalisation step is
        safe. When the aggregate capacity is below ``1 - tol`` the function
        returns residual cash instead of forcing the weights to sum to one.
    """

    if weights.empty:
        return weights

    w = weights.astype(float).copy()
    w[w < 0] = 0.0
    total = w.sum()
    if total == 0:
        return w
    w /= total

    caps = pd.Series(cap, index=w.index, dtype=float)
    if vol_adjusted_caps:
        for ticker, value in vol_adjusted_caps.items():
            if ticker in caps.index and pd.notna(value):
                caps.at[ticker] = float(value)

    for _ in range(max_iter):
        over = w > (caps + tol)
        if not over.any():
            break
        excess = (w[over] - caps[over]).sum()
        w.loc[over] = caps.loc[over]
        under = ~over
        if w[under].sum() > 0:
            w.loc[under] += (w.loc[under] / w.loc[under].sum()) * excess
        else:
            # No remaining capacity; leave residual cash so caps remain satisfied.
            break

    # Numerical cleanup and optional renormalisation when there is sufficient capacity
    w[w < 0] = 0.0
    capacity = caps.loc[w > 0].sum()
    if capacity >= 1 - tol and abs(w.sum() - 1.0) > tol:
        w = w / w.sum()

    return w


def l1_turnover(prev_w: pd.Series | None, w: pd.Series) -> float:
    """Compute 0.5 × L1 turnover between consecutive weight vectors."""

    if prev_w is None or len(prev_w) == 0:
        return float(0.5 * w.abs().sum())

    union = w.index.union(prev_w.index)
    aligned_w = w.reindex(union, fill_value=0.0)
    aligned_prev = prev_w.reindex(union, fill_value=0.0)
    return float(0.5 * (aligned_w - aligned_prev).abs().sum())
