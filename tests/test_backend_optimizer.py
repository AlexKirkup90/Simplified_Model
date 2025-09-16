import numpy as np
import pandas as pd
import streamlit as st
import sys, pathlib, types

# Provide empty secrets so backend import does not fail
st.secrets = types.SimpleNamespace(get=lambda *args, **kwargs: None)

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import backend


def test_run_backtest_isa_dynamic_uses_bayesian_optimizer(monkeypatch):
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

    captured: dict[str, object] = {}

    import strategy_core

    def fake_run_hybrid_backtest(daily_prices, cfg):
        captured["cfg"] = cfg
        idx = pd.date_range("2020-01-31", "2020-03-31", freq="M")
        return {
            "hybrid_rets": pd.Series(0.0, index=idx),
            "mom_turnover": pd.Series(0.0, index=idx),
            "mr_turnover": pd.Series(0.0, index=idx),
        }

    monkeypatch.setattr(strategy_core, "run_hybrid_backtest", fake_run_hybrid_backtest)

    dummy_cfg = backend.HybridConfig(
        momentum_top_n=2,
        momentum_cap=0.4,
        mr_top_n=4,
        mom_weight=0.6,
        mr_weight=0.4,
    )
    diagnostics = pd.DataFrame(
        [
            {"momentum_top_n": 2, "mom_weight": 0.6, "mr_weight": 0.4, "sharpe": 1.23},
            {"momentum_top_n": 5, "mom_weight": 0.7, "mr_weight": 0.3, "sharpe": 0.9},
        ]
    )

    def fake_bayes_opt(prices, **kwargs):
        captured["optimizer_kwargs"] = kwargs
        return dummy_cfg, diagnostics

    monkeypatch.setattr(backend.optimizer, "bayesian_optimize_hybrid", fake_bayes_opt)

    st.session_state["min_profitability"] = 0.0
    st.session_state["max_leverage"] = 10.0

    (
        _,
        _,
        _,
        _,
        best_cfg,
        search_diag,
    ) = backend.run_backtest_isa_dynamic(auto_optimize=True, roundtrip_bps=15.0)

    cfg = captured["cfg"]
    assert cfg.momentum_top_n == dummy_cfg.momentum_top_n
    assert cfg.momentum_cap == dummy_cfg.momentum_cap
    assert cfg.mr_top_n == dummy_cfg.mr_top_n
    assert cfg.mom_weight == dummy_cfg.mom_weight
    assert cfg.mr_weight == dummy_cfg.mr_weight

    assert isinstance(best_cfg, backend.HybridConfig)
    assert best_cfg == dummy_cfg
    pd.testing.assert_frame_equal(search_diag, diagnostics)

    kwargs = captured.get("optimizer_kwargs", {})
    assert kwargs.get("tc_bps") == 15.0
