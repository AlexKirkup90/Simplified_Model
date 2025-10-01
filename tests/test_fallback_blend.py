import pathlib
import sys
import types

import pandas as pd
import pytest
import streamlit as st

st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend  # noqa: E402


def test_fallback_blend_meets_min_positions_and_caps():
    blended, used, target, cash_weight, meta = backend.compose_graceful_fallback(
        stock_weights=pd.Series(dtype=float),
        regime_metrics={"regime_score": 60.0, "qqq_above_200dma": 0.4},
        regime_label="Neutral",
        base_target=0.8,
        min_names=backend.MIN_ELIGIBLE_FALLBACK,
        eligible_pool=0,
        leadership_slice=backend.LEADERSHIP_SLICE_DEFAULT,
        core_slice=backend.CORE_SPY_SLICE_DEFAULT,
    )

    assert used is True
    assert len(blended[blended > 0]) >= backend.MIN_ELIGIBLE_FALLBACK
    assert float(blended.max()) <= 0.10 + 1e-6
    assert cash_weight >= 0.0


def test_fallback_cap_relaxes_when_target_low():
    blended, used, target, cash_weight, meta = backend.compose_graceful_fallback(
        stock_weights=pd.Series(dtype=float),
        regime_metrics={"regime_score": 20.0, "qqq_above_200dma": 0.2},
        regime_label="Risk-Off",
        base_target=0.4,
        min_names=backend.MIN_ELIGIBLE_FALLBACK,
        eligible_pool=0,
        leadership_slice=backend.LEADERSHIP_SLICE_DEFAULT,
        core_slice=backend.CORE_SPY_SLICE_DEFAULT,
    )

    assert used is True
    assert target <= 0.5
    assert len(blended[blended > 0]) >= backend.MIN_ELIGIBLE_FALLBACK
    assert float(blended.max()) <= 0.20 + 1e-6
    assert cash_weight >= 0.0
    assert meta["components"]
