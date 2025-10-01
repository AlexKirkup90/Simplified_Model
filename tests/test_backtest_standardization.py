import numpy as np
import pandas as pd
import pytest
import streamlit as st
import types
import sys
import pathlib

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


def test_standardize_backtest_payload_creates_canonical_curves():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    gross = pd.Series([1.0, 1.05, 1.02, 1.08], index=idx)
    net = pd.Series([1.0, 1.04, 1.01, 1.06], index=idx)
    bench = pd.Series([1.0, 1.02, 0.99, 1.05], index=idx)
    turnover = pd.Series([0.2, 0.15, np.nan, 0.25], index=idx)

    payload = backend.standardize_backtest_payload(
        strategy_cum_gross=gross,
        strategy_cum_net=net,
        qqq_cum=bench,
        hybrid_tno=turnover,
        show_net=True,
    )

    equity_strategy = backend.deserialize_backtest_series(payload["equity_strategy"])
    assert np.allclose(equity_strategy.values, net.dropna().values)

    rets_strategy = backend.deserialize_backtest_series(payload["rets_strategy"])
    expected = net.pct_change().dropna()
    assert np.allclose(rets_strategy.values, expected.values)

    rets_bench = backend.deserialize_backtest_series(payload["rets_bench"])
    assert np.allclose(rets_bench.values, bench.pct_change().dropna().values)

    turnover_payload = backend.deserialize_backtest_series(payload["turnover"])
    expected_turnover = pd.Series([0.2, 0.15, 0.15, 0.25], index=idx)
    pd.testing.assert_series_equal(turnover_payload, expected_turnover, check_names=False, check_freq=False)

    assert payload["synthetic_flags"] == {
        "strategy_gross": False,
        "strategy_net": False,
        "benchmark": False,
        "turnover": False,
    }


def test_standardize_backtest_payload_builds_synthetic_when_missing():
    payload = backend.standardize_backtest_payload(
        strategy_cum_gross=None,
        strategy_cum_net=None,
        qqq_cum=None,
        hybrid_tno=None,
        show_net=True,
    )

    equity_strategy = backend.deserialize_backtest_series(payload["equity_strategy"])
    assert (equity_strategy == 1.0).all()
    turnover_payload = backend.deserialize_backtest_series(payload["turnover"])
    assert (turnover_payload == 0.0).all()
    assert payload["synthetic_flags"]["strategy_gross"] is True
    assert payload["synthetic_flags"]["strategy_net"] is True
    assert payload["synthetic_flags"]["benchmark"] is True


def test_compute_hedge_metadata_enforces_overlap():
    idx = pd.date_range("2021-01-31", periods=12, freq="ME")
    portfolio = pd.Series(np.linspace(0.01, 0.12, 12), index=idx)
    bench = portfolio.copy()

    meta = backend.compute_hedge_metadata(portfolio, bench, min_overlap=12)
    assert meta["overlap"] == 12
    assert meta["status"] == "ok"
    assert meta["correlation"] == pytest.approx(1.0)

    short_meta = backend.compute_hedge_metadata(portfolio.head(6), bench.head(6), min_overlap=12)
    assert short_meta["correlation"] is None
    assert short_meta["overlap"] == 6
    assert short_meta["status"] == "insufficient_data"
