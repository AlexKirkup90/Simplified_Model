import json
import pathlib
import sys
import types

import numpy as np
import pandas as pd
import pytest
import streamlit as st

st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import backend  # noqa: E402
from scripts import run_benchmarks  # noqa: E402


@pytest.fixture(autouse=True)
def _patch_backend(monkeypatch):
    index = pd.date_range("2020-01-31", periods=6, freq="M")
    equity = pd.Series(np.linspace(1.0, 1.6, len(index)), index=index)
    bench = pd.Series(np.linspace(1.0, 1.4, len(index)), index=index)
    turnover = pd.Series(0.5, index=index)

    def fake_run_backtest(**kwargs):
        return equity, equity, bench, turnover, None, pd.DataFrame()

    daily_index = pd.date_range("2020-01-01", "2020-06-30", freq="B")
    daily_prices = pd.DataFrame({"AAA": np.linspace(10, 15, len(daily_index)), "QQQ": np.linspace(50, 55, len(daily_index))}, index=daily_index)

    def fake_prepare(universe, start, end):
        return daily_prices, None, {}, universe

    monthly_prices = daily_prices.resample("M").last()

    def fake_fetch(tickers, start, end):
        cols = {t: np.linspace(10 + i, 12 + i, len(daily_index)) for i, t in enumerate(tickers)}
        return pd.DataFrame(cols, index=daily_index)

    def fake_metrics(window):
        return {"regime_score": 60.0, "regime": "Risk-On"}

    def fake_label(metrics):
        return metrics.get("regime", "Neutral"), metrics.get("regime_score", 50.0), {}

    def fake_fallback(**kwargs):
        weights = pd.Series(1.0 / len(backend.FALLBACK_BROAD_ETFS), index=backend.FALLBACK_BROAD_ETFS)
        return (
            weights,
            True,
            0.8,
            0.2,
            {
                "components": list(weights.index),
                "candidate_pool": list(weights.index),
                "equity_target": 0.8,
                "added": list(weights.index),
            },
        )

    monkeypatch.setattr(backend, "run_backtest_isa_dynamic", lambda **kwargs: fake_run_backtest())
    monkeypatch.setattr(backend, "_prepare_universe_for_backtest", lambda *args, **kwargs: fake_prepare(*args, **kwargs))
    monkeypatch.setattr(backend, "fetch_market_data", lambda *args, **kwargs: fake_fetch(*args, **kwargs))
    monkeypatch.setattr(backend, "compute_regime_metrics", lambda *args, **kwargs: fake_metrics(*args, **kwargs))
    monkeypatch.setattr(backend, "compute_regime_label", lambda metrics: fake_label(metrics))
    monkeypatch.setattr(backend, "compose_graceful_fallback", lambda *args, **kwargs: fake_fallback(**kwargs))

    yield


def test_run_benchmarks_outputs(tmp_path):
    out_dir = tmp_path / "artifacts"
    run_benchmarks.run_benchmarks("2020-01-01", "2020-06-30", str(out_dir), "Hybrid Top150")

    metrics_path = out_dir / "metrics.json"
    equity_path = out_dir / "equity_curves.csv"
    payload_path = out_dir / "payload.json"

    assert metrics_path.exists()
    assert equity_path.exists()
    assert payload_path.exists()

    data = json.loads(metrics_path.read_text())
    assert "scenarios" in data
    assert set(data["scenarios"].keys()) == {"Hybrid150", "SP500_Fallback", "Leadership_ETF_Blend"}
