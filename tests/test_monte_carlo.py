import streamlit as st
import sys, pathlib, types
import pandas as pd
import numpy as np
import pytest

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


def test_run_monte_carlo_deterministic_reproducibility():
    """Monte Carlo projections should be reproducible and include expected keys."""
    returns = pd.Series([0.01] * 24)
    result1 = backend.run_monte_carlo_projections(returns, n_scenarios=100, horizon_months=12)
    result2 = backend.run_monte_carlo_projections(returns, n_scenarios=100, horizon_months=12)

    expected_total = (1 + 0.01) ** 12 - 1
    expected_keys = {
        "scenarios",
        "percentiles",
        "mean_return",
        "std_return",
        "prob_positive",
        "prob_beat_10pct",
        "downside_risk",
        "horizon_months",
    }

    assert set(result1.keys()) == expected_keys
    assert all(result1["percentiles"][f"p{c}"] == pytest.approx(expected_total) for c in [10, 50, 90])
    assert result1["mean_return"] == pytest.approx(expected_total)
    assert result1["std_return"] == pytest.approx(0)
    assert result1["prob_positive"] == pytest.approx(1)
    assert result1["prob_beat_10pct"] == pytest.approx(1)
    assert result1["downside_risk"] == pytest.approx(0)
    assert result1["horizon_months"] == 12
    assert np.array_equal(result1["scenarios"], result2["scenarios"])


def test_run_monte_carlo_short_history():
    """Short return histories should return an informative error."""
    short_returns = pd.Series([0.01] * 6)
    result = backend.run_monte_carlo_projections(short_returns)

    assert "error" in result
    assert "Insufficient historical data" in result["error"]
