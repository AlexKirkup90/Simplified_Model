import numpy as np
import pandas as pd
import pytest
import streamlit as st
import sys, pathlib, types
from datetime import date

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend

def setup_module(module):
    st.session_state.clear()

def _base_patch(monkeypatch, metrics, regime_label="Risk-On"):
    def fake_get_universe(choice):
        return ["AAA", "BBB"], {"AAA": "Tech", "BBB": "Tech"}, "label"

    def fake_fetch_market_data(tickers, start, end):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        return pd.DataFrame(100.0, index=dates, columns=tickers)

    def fake_compute_regime_metrics(prices):
        return metrics

    def fake_get_market_regime():
        return regime_label, {}

    monkeypatch.setattr(backend, "get_universe", fake_get_universe)
    monkeypatch.setattr(backend, "fetch_market_data", fake_fetch_market_data)
    monkeypatch.setattr(backend, "compute_regime_metrics", fake_compute_regime_metrics)
    monkeypatch.setattr(backend, "get_market_regime", fake_get_market_regime)


def test_assess_market_conditions_risk_on(monkeypatch):
    metrics = {
        "breadth_pos_6m": 0.7,
        "qqq_vol_10d": 0.02,
        "vix_term_structure": 1.1,
        "hy_oas": 4.0,
        "qqq_above_200dma": 1.0,
    }
    _base_patch(monkeypatch, metrics, "Risk-On")
    result = backend.assess_market_conditions(date(2024, 1, 5))
    settings = result["settings"]
    assert result["metrics"]["regime"] == "Risk-On"
    assert settings["sector_cap"] == pytest.approx(0.35)
    assert settings["name_cap"] == pytest.approx(0.30)
    assert settings["stickiness_days"] == 5


def test_assess_market_conditions_risk_off(monkeypatch):
    metrics = {
        "breadth_pos_6m": 0.3,
        "qqq_vol_10d": 0.05,
        "vix_term_structure": 0.8,
        "hy_oas": 7.0,
        "qqq_above_200dma": 0.0,
    }
    _base_patch(monkeypatch, metrics, "Risk-Off")
    result = backend.assess_market_conditions(date(2024, 1, 5))
    settings = result["settings"]
    assert result["metrics"]["regime"] == "Risk-Off"
    assert settings["sector_cap"] == pytest.approx(0.20)
    assert settings["name_cap"] == pytest.approx(0.20)
    assert settings["stickiness_days"] == 14
