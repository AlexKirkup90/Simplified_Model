import numpy as np
import pandas as pd
import pytest
import streamlit as st
import types
import sys
import pathlib

st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend  # noqa: E402


def test_compose_graceful_fallback_blends_components():
    stocks = pd.Series(dtype=float)
    metrics = {"qqq_above_200dma": 0.0, "qqq_50dma_slope_10d": -0.01}

    blended, used, target, cash_weight = backend.compose_graceful_fallback(
        stocks,
        metrics,
        "Extreme Risk-Off",
        0.85,
        min_names=backend.MIN_ELIGIBLE_FALLBACK,
        eligible_pool=0,
        leadership_slice=0.3,
        core_slice=0.2,
    )

    assert used is True
    assert pytest.approx(blended.sum() + cash_weight, rel=1e-6) == 1.0
    assert cash_weight > 0
    assert any(ticker in blended.index for ticker in backend.FALLBACK_LEADERSHIP_ETFS + [backend.FALLBACK_CORE_TICKER])
    assert target <= 0.45  # risk-off cap applied


def test_compose_graceful_fallback_scales_existing_weights_without_trigger():
    stocks = pd.Series({"AAPL": 0.5, "MSFT": 0.35}, dtype=float)
    metrics = {"qqq_above_200dma": 1.0, "qqq_50dma_slope_10d": 0.02}

    blended, used, target, cash_weight = backend.compose_graceful_fallback(
        stocks,
        metrics,
        "Risk-On",
        0.85,
        min_names=1,
        eligible_pool=10,
        leadership_slice=0.3,
        core_slice=0.2,
    )

    assert used is False
    assert cash_weight > 0
    assert pytest.approx(blended.sum(), rel=1e-6) == pytest.approx(target, rel=1e-6)


def test_hy_oas_missing_sets_neutral_score(monkeypatch):
    idx = pd.date_range("2020-01-01", periods=260, freq="B")
    prices = pd.DataFrame({"AAPL": np.linspace(100, 150, len(idx))}, index=idx)

    def fake_benchmark(ticker, start, end):
        data = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
        if ticker == "BAMLH0A0HYM2":
            return pd.Series(dtype=float)
        return data

    monkeypatch.setattr(backend, "get_benchmark_series", fake_benchmark)

    metrics = backend.compute_regime_metrics(prices)
    assert metrics["hy_oas_status"] == "missing"
    assert metrics["hy_oas_score"] == 50.0


def test_format_display_filters_zero_weights():
    weights = pd.Series({"AAPL": 0.05, "SPY": 0.0, "CASH": 0.0, backend.HEDGE_TICKER_LABEL: -0.1})
    _, raw = backend._format_display(weights)
    assert "SPY" not in raw.index and "CASH" not in raw.index
    assert backend.HEDGE_TICKER_LABEL in raw.index
