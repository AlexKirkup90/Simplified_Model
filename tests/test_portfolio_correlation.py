import numpy as np
import pandas as pd
import pytest
import streamlit as st
import sys, pathlib, types

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


def test_calculate_portfolio_correlation_resamples_to_monthly():
    # Three months of daily returns
    index = pd.date_range("2023-01-01", periods=90, freq="D")
    rng = np.random.default_rng(seed=42)
    market_daily = rng.normal(0.001, 0.01, len(index))
    portfolio_daily = market_daily + rng.normal(0, 0.005, len(index))
    portfolio_returns = pd.Series(portfolio_daily, index=index)
    market_returns = pd.Series(market_daily, index=index)

    expected_port = (1 + portfolio_returns).resample("M").prod() - 1
    expected_market = (1 + market_returns).resample("M").prod() - 1
    expected_corr = expected_port.corr(expected_market)

    corr = backend.calculate_portfolio_correlation_to_market(portfolio_returns, market_returns)

    assert corr == pytest.approx(expected_corr)


def test_calculate_portfolio_correlation_fallback_metadata():
    idx = pd.date_range("2023-01-31", periods=2, freq="M")
    portfolio_returns = pd.Series([0.01, -0.02], index=idx)

    corr, meta = backend.calculate_portfolio_correlation_to_market(
        portfolio_returns,
        portfolio_returns,
        return_metadata=True,
    )

    assert corr == 0.0
    assert meta["fallback"] is True
    assert meta["points"] <= 2
