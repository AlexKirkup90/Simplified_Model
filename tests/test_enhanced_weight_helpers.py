import pandas as pd
import numpy as np
import streamlit as st
import sys, pathlib, types
import pytest

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


def _setup_prices():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    prices = pd.DataFrame({"AAA": np.linspace(10, 19, len(dates)),
                           "BBB": np.linspace(20, 29, len(dates))}, index=dates)
    return prices


@pytest.fixture
def preset():
    return {
        "mom_topn": 2,
        "mom_w": 1.0,
        "mr_topn": 0,
        "mr_w": 0.0,
        "mr_lb": 5,
        "mr_ma": 200,
        "stickiness_days": 1,
        "mom_cap": 1.0,
        "sector_cap": 1.0,
    }


def _patch_selection(monkeypatch):
    def fake_composite_score(df):
        return pd.Series({"AAA": 1.0, "BBB": 1.0})

    def fake_mom_z(df):
        return pd.Series({"AAA": 1.0, "BBB": 1.0})

    def fake_stable(df, top_n, days):
        return ["AAA", "BBB"]

    monkeypatch.setattr(backend, "composite_score", fake_composite_score)
    monkeypatch.setattr(backend, "blended_momentum_z", fake_mom_z)
    monkeypatch.setattr(backend, "momentum_stable_names", fake_stable)


def test_risk_parity_influences_weights(monkeypatch, preset):
    _patch_selection(monkeypatch)
    prices = _setup_prices()
    sectors = {"AAA": "Tech", "BBB": "Tech"}

    calls = {"rp": 0}

    def fake_rp(prices, tickers, lookback=63):
        calls["rp"] += 1
        return pd.Series({"AAA": 1.0, "BBB": 0.0})

    def fake_vol_caps(weights, daily, lookback=63, base_cap=1.0):
        return {"AAA": 1.0, "BBB": 1.0}

    monkeypatch.setattr(backend, "risk_parity_weights", fake_rp)
    monkeypatch.setattr(backend, "get_volatility_adjusted_caps", fake_vol_caps)

    w_enh = backend._build_isa_weights_fixed(prices, preset, sectors, use_enhanced_features=True)
    w_off = backend._build_isa_weights_fixed(prices, preset, sectors, use_enhanced_features=False)

    assert calls["rp"] == 1
    assert w_enh["AAA"] != pytest.approx(w_off["AAA"])
    assert w_enh.sum() == pytest.approx(1.0)


def test_volatility_caps_influence_weights(monkeypatch, preset):
    _patch_selection(monkeypatch)
    prices = _setup_prices()
    sectors = {"AAA": "Tech", "BBB": "Tech"}

    calls = {"vc": 0}

    def fake_rp(prices, tickers, lookback=63):
        return pd.Series({"AAA": 0.5, "BBB": 0.5})

    def fake_vol_caps(weights, daily, lookback=63, base_cap=1.0):
        calls["vc"] += 1
        return {"AAA": 0.1, "BBB": 1.0}

    monkeypatch.setattr(backend, "risk_parity_weights", fake_rp)
    monkeypatch.setattr(backend, "get_volatility_adjusted_caps", fake_vol_caps)

    w_enh = backend._build_isa_weights_fixed(prices, preset, sectors, use_enhanced_features=True)
    w_off = backend._build_isa_weights_fixed(prices, preset, sectors, use_enhanced_features=False)

    assert calls["vc"] == 1
    assert w_enh["AAA"] < w_off["AAA"]
    assert w_enh["AAA"] == pytest.approx(0.1)
    assert w_enh.sum() == pytest.approx(1.0)
