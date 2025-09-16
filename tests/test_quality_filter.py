import numpy as np
import pandas as pd
import streamlit as st
import sys, pathlib, types

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


def test_fundamental_quality_filter_basic():
    df = pd.DataFrame({
        "profitability": [0.2, -0.1, 0.05],
        "leverage": [0.5, 0.4, 3.0],
    }, index=["AAA", "BBB", "CCC"])
    keep = backend.fundamental_quality_filter(df, min_profitability=0.0, max_leverage=1.0)
    assert keep == ["AAA"]


def test_run_backtest_isa_dynamic_uses_quality_filter(monkeypatch):
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    close = pd.DataFrame({
        "AAA": np.linspace(100, 110, len(dates)),
        "BBB": np.linspace(50, 60, len(dates)),
        "QQQ": np.linspace(200, 210, len(dates)),
    }, index=dates)
    vol = pd.DataFrame(1.0, index=dates, columns=["AAA", "BBB", "QQQ"])
    sectors_map = {"AAA": "Tech", "BBB": "Tech"}

    def fake_prepare(universe_choice, start_date, end_date):
        return close, vol, sectors_map, "Test"

    monkeypatch.setattr(backend, "_prepare_universe_for_backtest", fake_prepare)

    fdf = pd.DataFrame({
        "profitability": [0.2, -0.2],
        "leverage": [0.5, 0.5],
    }, index=["AAA", "BBB"])

    def fake_fetch_fundamental_metrics(tickers):
        return fdf.reindex(tickers)

    monkeypatch.setattr(backend, "fetch_fundamental_metrics", fake_fetch_fundamental_metrics)

    captured = {}

    import strategy_core

    def fake_run_hybrid_backtest(daily_prices, cfg, apply_vol_target=False):
        captured["cols"] = list(daily_prices.columns)
        captured["apply_vol_target"] = apply_vol_target
        idx = pd.date_range("2020-01-31", "2020-03-31", freq="M")
        return {
            "hybrid_rets": pd.Series(0.0, index=idx),
            "mom_turnover": pd.Series(0.0, index=idx),
            "mr_turnover": pd.Series(0.0, index=idx),
        }

    monkeypatch.setattr(strategy_core, "run_hybrid_backtest", fake_run_hybrid_backtest)

    st.session_state["min_profitability"] = 0.0
    st.session_state["max_leverage"] = 1.0

    backend.run_backtest_isa_dynamic(
        min_dollar_volume=0,
        top_n=1,
        name_cap=1.0,
        sector_cap=1.0,
        stickiness_days=1,
        mr_topn=1,
        mom_weight=1.0,
        mr_weight=0.0,
        use_enhanced_features=False,
        apply_quality_filter=True,
    )

    assert captured.get("cols") == ["AAA"]
