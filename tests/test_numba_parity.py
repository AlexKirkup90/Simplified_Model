import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import backend


COLS = ["AAA", "BBB", "CCC", "DDD"]
RNG = np.random.default_rng(123)

MONTHLY_INDEX = pd.date_range("2015-01-31", periods=24, freq="M")
MONTHLY_VALUES = np.exp(
    np.linspace(0.0, 0.8, len(MONTHLY_INDEX))[:, None]
    + RNG.normal(scale=0.05, size=(len(MONTHLY_INDEX), len(COLS)))
    + np.arange(len(COLS)) * 0.1
)
MONTHLY_DF = pd.DataFrame(100.0 * MONTHLY_VALUES, index=MONTHLY_INDEX, columns=COLS)

DAILY_INDEX = pd.date_range("2020-01-01", periods=260, freq="B")
DAILY_VALUES = np.exp(
    np.linspace(0.0, 1.0, len(DAILY_INDEX))[:, None]
    + RNG.normal(scale=0.02, size=(len(DAILY_INDEX), len(COLS)))
    + np.arange(len(COLS)) * 0.05
)
DAILY_DF = pd.DataFrame(100.0 * DAILY_VALUES, index=DAILY_INDEX, columns=COLS)


def _clone(value):
    return value.copy() if hasattr(value, "copy") else value


def _invoke_both(func, *args, **kwargs):
    original_flag = backend.NUMBA_OK
    try:
        backend.NUMBA_OK = False
        baseline = func(*[_clone(arg) for arg in args], **{k: _clone(v) for k, v in kwargs.items()})
        backend.NUMBA_OK = True
        accelerated = func(*[_clone(arg) for arg in args], **{k: _clone(v) for k, v in kwargs.items()})
    finally:
        backend.NUMBA_OK = original_flag
    return baseline, accelerated


def test_robust_return_stats_numba_parity():
    cases = [
        pd.Series([0.01, 0.02, 0.015, np.nan, -0.03, 0.04, 0.05, 0.01], dtype=float),
        pd.Series([0.01, 0.01, 0.01, 0.01, np.nan], dtype=float),
    ]

    for series in cases:
        baseline, accelerated = _invoke_both(backend._robust_return_stats, series)
        np.testing.assert_allclose(accelerated, baseline, rtol=0.0, atol=1e-12)


def test_blended_momentum_z_numba_parity():
    baseline, accelerated = _invoke_both(backend.blended_momentum_z, MONTHLY_DF)
    pd.testing.assert_series_equal(
        accelerated,
        baseline,
        check_names=False,
        rtol=0.0,
        atol=1e-12,
    )


def test_lowvol_z_numba_parity():
    baseline, accelerated = _invoke_both(backend.lowvol_z, DAILY_DF)
    pd.testing.assert_series_equal(
        accelerated,
        baseline,
        check_names=False,
        rtol=0.0,
        atol=1e-12,
    )


def test_trend_z_numba_parity():
    baseline, accelerated = _invoke_both(backend.trend_z, DAILY_DF)
    pd.testing.assert_series_equal(
        accelerated,
        baseline,
        check_names=False,
        rtol=0.0,
        atol=1e-12,
    )
