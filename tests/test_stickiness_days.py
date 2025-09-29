import pandas as pd
import streamlit as st
import sys
import pathlib
import types

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


def test_momentum_stability_uses_stickiness_fallback(monkeypatch):
    idx = pd.date_range("2024-01-02", periods=6, freq="B")
    data = {
        "AAA": [100, 101, 102, 103, 104, 105],
        "BBB": [50, 49, 50, 51, 52, 53],
    }
    daily_close = pd.DataFrame(data, index=idx)
    sectors_map = {ticker: "Tech" for ticker in daily_close.columns}

    preset = {
        "mom_topn": 2,
        "mom_w": 1.0,
        "mom_cap": 0.3,
        "mr_lb": 1,
        "mr_ma": 2,
        "mr_topn": 1,
        "mr_w": 0.0,
        "stickiness_days": 11,
    }

    monthly = daily_close.resample("M").last().ffill()

    monkeypatch.setattr(backend, "compute_signal_panels", lambda prices: {"monthly": monthly})
    monkeypatch.setattr(backend, "composite_score", lambda prices, panels=None: prices.copy())
    monkeypatch.setattr(backend, "blended_momentum_z", lambda monthly: pd.Series(1.0, index=daily_close.columns))

    captured = {}

    def fake_momentum_stable_names(prices, top_n, days, panels=None):
        captured["days"] = days
        return prices.columns[:top_n]

    monkeypatch.setattr(backend, "momentum_stable_names", fake_momentum_stable_names)

    weights = backend._build_isa_weights_fixed(
        daily_close,
        preset,
        sectors_map,
        use_enhanced_features=False,
    )

    assert not weights.empty
    assert captured["days"] == 11
