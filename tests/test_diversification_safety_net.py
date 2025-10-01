import numpy as np
import pandas as pd
import streamlit as st
import types

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

import backend


def _dummy_prices(columns):
    idx = pd.date_range("2023-01-01", periods=3, freq="B")
    data = np.tile(np.arange(1, len(idx) + 1), (len(columns), 1)).T
    return pd.DataFrame(data, index=idx, columns=columns)


def test_diversification_safety_net_targets_minimum_breadth():
    base_names = [f"T{i}" for i in range(8)]
    weights = pd.Series([0.125] * len(base_names), index=base_names, dtype=float)
    extras = pd.Series(
        np.linspace(1.0, 0.2, 7),
        index=[f"X{i}" for i in range(7)],
    )
    prices = _dummy_prices(base_names + ["QQQ"])

    adjusted, applied, notes = backend.apply_diversification_safety_net(
        weights,
        extras,
        prices,
        min_names=10,
        fallback_min_names=15,
    )

    assert applied is True
    assert adjusted[adjusted > 0].shape[0] >= 15
    assert "secondary_pool" in notes


def test_diversification_safety_net_uses_etf_overlay_when_needed():
    base_names = [f"Z{i}" for i in range(8)]
    weights = pd.Series([0.125] * len(base_names), index=base_names, dtype=float)
    extras = pd.Series([1.0, 0.8], index=["Y1", "Y2"])
    prices = _dummy_prices(base_names + ["QQQ"])

    adjusted, applied, notes = backend.apply_diversification_safety_net(
        weights,
        extras,
        prices,
        min_names=10,
        fallback_min_names=15,
    )

    assert applied is True
    assert "etf_overlay" in notes
    assert "QQQ" in adjusted.index
