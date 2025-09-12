import numpy as np
import pandas as pd
import streamlit as st
import sys, pathlib, types

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


def test_run_backtest_isa_dynamic_uses_optimizer(monkeypatch):
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
        "profitability": [0.2, 0.2],
        "leverage": [0.5, 0.5],
    }, index=["AAA", "BBB"])

    monkeypatch.setattr(backend, "fetch_fundamental_metrics", lambda tickers: fdf.reindex(tickers))

    captured = {}

    def fake_momentum(daily, sectors_map, top_n, name_cap, sector_cap, stickiness_days, use_enhanced_features):
        captured["top_n"] = top_n
        captured["name_cap"] = name_cap
        captured["sector_cap"] = sector_cap
        series = pd.Series(0.0, index=pd.date_range("2020-01-31", "2020-03-31", freq="M"))
        return series, series

    def fake_mr(daily, lookback_period_mr, top_n_mr, long_ma_days):
        series = pd.Series(0.0, index=pd.date_range("2020-01-31", "2020-03-31", freq="M"))
        return series, series

    def fake_combine(mom_rets, mr_rets, mom_tno, mr_tno, mom_w, mr_w):
        captured["mom_weight"] = mom_w
        captured["mr_weight"] = mr_w
        return pd.Series(0.0, index=mom_rets.index), pd.Series(0.0, index=mom_rets.index)

    monkeypatch.setattr(backend, "run_momentum_composite_param", fake_momentum)
    monkeypatch.setattr(backend, "run_backtest_mean_reversion", fake_mr)
    monkeypatch.setattr(backend, "combine_hybrid", fake_combine)

    class DummyCfg:
        momentum_top_n = 2
        momentum_cap = 0.4
        mom_weight = 0.6
        mr_weight = 0.4

    monkeypatch.setattr(backend, "optimize_hybrid_strategy", lambda prices: (DummyCfg(), 0.33))

    st.session_state["min_profitability"] = 0.0
    st.session_state["max_leverage"] = 10.0

    backend.run_backtest_isa_dynamic()

    assert captured["top_n"] == 2
    assert captured["name_cap"] == 0.4
    assert captured["sector_cap"] == 0.33
    assert captured["mom_weight"] == 0.6
    assert captured["mr_weight"] == 0.4
