import pandas as pd
import streamlit as st
import sys, pathlib, types
from datetime import date
import pytest

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

# Make backend importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


def _mock_env(monkeypatch):
    sectors = {"AAA": "Tech", "BBB": "Tech"}
    tickers = list(sectors.keys())

    def fake_get_universe(choice):
        return tickers, sectors, "Label"

    def fake_fetch_price_volume(tickers, start, end):
        idx = pd.date_range("2024-06-01", periods=10, freq="B")
        close = pd.DataFrame(100.0, index=idx, columns=tickers)
        vol = pd.DataFrame(1000.0, index=idx, columns=tickers)
        return close, vol

    def fake_fetch_fundamental_metrics(tickers):
        return pd.DataFrame(index=tickers)

    def fake_fundamental_quality_filter(df, min_profitability, max_leverage):
        return df.index.tolist()

    def fake_build_weights(close, params, sectors_map, use_enhanced_features=True):
        base = pd.Series({"AAA": 0.6, "BBB": 0.4})
        return backend.enforce_caps_iteratively(
            base,
            sectors_map,
            name_cap=params["mom_cap"],
            sector_cap=params["sector_cap"],
        )

    monkeypatch.setattr(backend, "get_universe", fake_get_universe)
    monkeypatch.setattr(backend, "fetch_price_volume", fake_fetch_price_volume)
    monkeypatch.setattr(backend, "fetch_fundamental_metrics", fake_fetch_fundamental_metrics)
    monkeypatch.setattr(backend, "fundamental_quality_filter", fake_fundamental_quality_filter)
    monkeypatch.setattr(backend, "_build_isa_weights_fixed", fake_build_weights)
    monkeypatch.setattr(backend, "compute_regime_metrics", lambda hist: {})
    monkeypatch.setattr(backend, "get_regime_adjusted_exposure", lambda metrics: 1.0)
    monkeypatch.setattr(backend, "is_rebalance_today", lambda today, idx: False)

    return sectors


def test_prev_portfolio_trimmed(monkeypatch):
    st.session_state.clear()
    sectors = _mock_env(monkeypatch)
    preset = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]

    prev = pd.DataFrame({"Weight": [0.6, 0.4]}, index=["AAA", "BBB"])
    disp, raw, decision = backend.generate_live_portfolio_isa_monthly(
        preset, prev, as_of=date(2024, 6, 17)
    )

    raw_w = raw["Weight"].astype(float)
    sector_series = pd.Series(sectors)

    # Previous portfolio violated caps
    assert prev["Weight"].max() > preset["mom_cap"]

    # Generated portfolio trims to caps
    assert raw_w.max() <= preset["mom_cap"] + 1e-12
    assert raw_w.groupby(sector_series).sum().max() <= preset["sector_cap"] + 1e-12


def test_fresh_portfolio_respects_caps(monkeypatch):
    st.session_state.clear()
    sectors = _mock_env(monkeypatch)
    preset = backend.STRATEGY_PRESETS["ISA Dynamic (0.75)"]

    disp, raw, decision = backend.generate_live_portfolio_isa_monthly(
        preset, None, as_of=date(2024, 6, 17)
    )

    raw_w = raw["Weight"].astype(float)
    sector_series = pd.Series(sectors)

    assert raw_w.max() <= preset["mom_cap"] + 1e-12
    assert raw_w.groupby(sector_series).sum().max() <= preset["sector_cap"] + 1e-12
