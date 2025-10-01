import math
import pathlib
import sys
import types

import numpy as np
import streamlit as st

st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import backend


def test_compute_regime_label_bearish_risk_off():
    metrics = {
        "qqq_above_200dma": 0,
        "breadth_pos_6m": 0.0,
        "vix_term_structure": 1.4,
        "qqq_vol_10d": 0.04,
        "universe_above_200dma": 0.1,
    }

    label, score, components = backend.compute_regime_label(metrics)

    assert label == "Risk-Off"
    assert score <= 33.0
    assert components["vix_ts"] < 10


def test_compute_regime_label_blocks_impossible_risk_on():
    metrics = {
        "qqq_above_200dma": 0,
        "breadth_pos_6m": 0.03,
        "vix_term_structure": 1.3,
        "qqq_vol_10d": np.nan,
        "qqq_50dma_slope_10d": 0.05,
        "universe_above_200dma": 0.2,
    }

    label, score, _ = backend.compute_regime_label(metrics)

    assert label != "Risk-On"
    assert score <= 55.0


def test_compute_regime_label_nan_volatility_neutral():
    metrics = {
        "qqq_above_200dma": 1,
        "breadth_pos_6m": 0.5,
        "vix_term_structure": 1.0,
        "qqq_vol_10d": np.nan,
        "universe_above_200dma": 0.5,
    }

    _, _, components = backend.compute_regime_label(metrics)

    assert math.isclose(components["volatility"], 50.0, abs_tol=1e-6)


def test_compute_regime_label_vix_directionality():
    base = {
        "qqq_above_200dma": 1,
        "breadth_pos_6m": 0.5,
        "qqq_vol_10d": 0.02,
        "universe_above_200dma": 0.5,
    }

    _, _, comps_low = backend.compute_regime_label({**base, "vix_term_structure": 0.9})
    _, _, comps_high = backend.compute_regime_label({**base, "vix_term_structure": 1.1})

    assert comps_low["vix_ts"] > comps_high["vix_ts"]
