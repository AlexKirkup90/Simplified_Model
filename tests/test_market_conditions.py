import numpy as np
import pandas as pd
import pytest
import streamlit as st
import sys, pathlib, types, json
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


def test_assess_market_conditions_logs(monkeypatch):
    metrics = {
        "breadth_pos_6m": 0.7,
        "qqq_vol_10d": 0.02,
        "vix_term_structure": 1.1,
        "hy_oas": 4.0,
        "qqq_above_200dma": 1.0,
    }
    _base_patch(monkeypatch, metrics, "Risk-On")

    def fake_load():
        return pd.DataFrame(columns=["date", "metrics", "settings", "portfolio_ret", "benchmark_ret"])

    captured = {}

    def fake_save(df):
        captured["df"] = df

    monkeypatch.setattr(backend, "load_assess_log", fake_load)
    monkeypatch.setattr(backend, "save_assess_log", fake_save)

    backend.assess_market_conditions(date(2024, 1, 5))

    assert "df" in captured
    row = captured["df"].iloc[0]
    assert row["date"] == pd.Timestamp("2024-01-05")
    m = json.loads(row["metrics"])
    s = json.loads(row["settings"])
    assert "breadth_pos_6m" in m
    assert "sector_cap" in s


def test_record_assessment_outcome(monkeypatch):
    log_df = pd.DataFrame([
        {
            "date": pd.Timestamp("2024-01-05"),
            "metrics": "{}",
            "settings": "{}",
            "portfolio_ret": np.nan,
            "benchmark_ret": np.nan,
        }
    ])

    monkeypatch.setattr(backend, "load_assess_log", lambda: log_df.copy())
    saved = {}
    monkeypatch.setattr(backend, "save_assess_log", lambda df: saved.setdefault("df", df))

    port_df = pd.DataFrame({"Weight": [0.5, 0.5]}, index=["AAA", "BBB"])
    monkeypatch.setattr(backend, "load_previous_portfolio", lambda: port_df)

    dates = pd.to_datetime(["2024-01-05", "2024-02-05"])
    prices = pd.DataFrame({
        "AAA": [100, 110],
        "BBB": [100, 90],
        "QQQ": [100, 105],
    }, index=dates)

    monkeypatch.setattr(
        backend,
        "fetch_market_data",
        lambda tickers, start, end: prices.reindex(columns=tickers),
    )

    result = backend.record_assessment_outcome(date(2024, 1, 5))
    assert result["ok"]
    assert saved["df"].loc[0, "portfolio_ret"] == pytest.approx(0.0)
    assert saved["df"].loc[0, "benchmark_ret"] == pytest.approx(0.05)
