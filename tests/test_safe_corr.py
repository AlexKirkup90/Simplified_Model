import pathlib
import sys
import types

import pandas as pd
import pytest
import streamlit as st

st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend  # noqa: E402


def test_correlation_none_when_overlap_short():
    idx = pd.date_range("2023-01-31", periods=5, freq="M")
    portfolio = pd.Series([0.01, 0.02, -0.01, 0.0, 0.005], index=idx)
    market = pd.Series([0.015, 0.018, -0.005, 0.002, 0.004], index=idx)

    corr, meta = backend.calculate_portfolio_correlation_to_market(
        portfolio,
        market,
        return_metadata=True,
    )

    assert corr is None
    assert meta["fallback"] is True
    assert meta["status"] == "insufficient_data"
    assert meta["points"] == 5
    assert meta["n"] == 5


def test_correlation_none_when_series_degenerate():
    idx = pd.date_range("2022-01-31", periods=6, freq="M")
    portfolio = pd.Series([0.0] * 6, index=idx)
    market = pd.Series([0.01, 0.02, -0.01, 0.0, 0.005, -0.002], index=idx)

    corr, meta = backend.calculate_portfolio_correlation_to_market(
        portfolio,
        market,
        return_metadata=True,
    )

    assert corr is None
    assert meta["status"] == "degenerate_series"
    assert meta["fallback"] is True
    assert meta["n"] == 6
