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

    def fake_run_hybrid_backtest(daily_prices, cfg, apply_vol_target=False):
        captured["cfg"] = cfg
        captured["apply_vol_target"] = apply_vol_target
        idx = pd.date_range("2020-01-31", "2020-03-31", freq="M")
        return {
            "hybrid_rets": pd.Series(0.0, index=idx),
            "mom_turnover": pd.Series(0.0, index=idx),
            "mr_turnover": pd.Series(0.0, index=idx),
        }

    monkeypatch.setattr(strategy_core, "run_hybrid_backtest", fake_run_hybrid_backtest)

    # Dummy optimizer outputs (Bayesian-style)
    dummy_cfg = backend.HybridConfig(
        momentum_top_n=2,
        momentum_cap=0.40,
        mr_top_n=4,
        mom_weight=0.60,
        mr_weight=0.40,
    )

    diagnostics = pd.DataFrame(
        [
            {"momentum_top_n": 2, "momentum_cap": 0.40, "mom_weight": 0.60, "mr_weight": 0.40, "sharpe": 1.23},
            {"momentum_top_n": 5, "momentum_cap": 0.25, "mom_weight": 0.70, "mr_weight": 0.30, "sharpe": 0.90},
        ]
    )

    def fake_bayes_opt(prices, **kwargs):
        captured["optimizer_kwargs"] = kwargs
        return dummy_cfg, diagnostics

    # Patch the optimizer entry point used by backend
    monkeypatch.setattr(backend.optimizer, "bayesian_optimize_hybrid", fake_bayes_opt, raising=True)

    # Keep ancillary patches from main branch for deterministic tests
    monkeypatch.setattr(backend, "compute_regime_metrics", lambda *args, **kwargs: {}, raising=True)
    monkeypatch.setattr(backend, "build_hedge_weight", lambda *args, **kwargs: 0.0, raising=True)
    monkeypatch.setattr(
        backend,
        "calculate_portfolio_correlation_to_market",
        lambda *args, **kwargs: 0.0,
        raising=True,
    )

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
    assert isinstance(cfg, backend.HybridConfig)
    assert cfg.tc_bps == 15.0
    assert captured.get("apply_vol_target") is False

    assert isinstance(best_cfg, backend.HybridConfig)
    assert best_cfg == dummy_cfg
    pd.testing.assert_frame_equal(search_diag, diagnostics)

    optimizer_kwargs = captured.get("optimizer_kwargs", {})
    assert optimizer_kwargs.get("tc_bps") == 15.0


def test_run_backtest_isa_dynamic_with_auto_opt(monkeypatch):
    # Minimal dummy data for backtest plumbing
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

    monkeypatch.setattr(backend, "_prepare_universe_for_backtest", fake_prepare, raising=True)

    # --- Fake Bayesian optimizer output ---
    dummy_cfg = backend.HybridConfig(
        momentum_top_n=2,
        momentum_cap=0.40,
        mr_top_n=4,
        mom_weight=0.60,
        mr_weight=0.40,
    )
    diagnostics = pd.DataFrame(
        [
            {"momentum_top_n": 2, "momentum_cap": 0.40, "mom_weight": 0.60, "mr_weight": 0.40, "sharpe": 1.23},
            {"momentum_top_n": 5, "momentum_cap": 0.25, "mom_weight": 0.70, "mr_weight": 0.30, "sharpe": 0.90},
        ]
    )

    captured = {}

    def fake_bayes_opt(prices, **kwargs):
        captured["optimizer_kwargs"] = kwargs
        return dummy_cfg, diagnostics

    monkeypatch.setattr(backend.optimizer, "bayesian_optimize_hybrid", fake_bayes_opt, raising=True)

    # Capture cfg passed into strategy_core.run_hybrid_backtest
    import strategy_core

    def fake_run_hybrid_backtest(daily_prices, cfg, apply_vol_target=False, get_constituents=None):
        captured["cfg"] = cfg
        captured["apply_vol_target"] = apply_vol_target
        idx = pd.date_range("2020-01-31", "2020-03-31", freq="M")
        return {
            "hybrid_rets": pd.Series(0.0, index=idx),
            "mom_turnover": pd.Series(0.0, index=idx),
            "mr_turnover": pd.Series(0.0, index=idx),
        }

    monkeypatch.setattr(strategy_core, "run_hybrid_backtest", fake_run_hybrid_backtest, raising=True)

    # Run with auto_optimize + transaction costs on (15 bps)
    strat_cum_gross, strat_cum_net, qqq_cum, hybrid_tno, best_cfg, search_diag = backend.run_backtest_isa_dynamic(
        roundtrip_bps=15.0,
        min_dollar_volume=0,
        top_n=1,              # user inputs (will be overridden by optimizer)
        name_cap=1.0,
        sector_cap=1.0,
        stickiness_days=1,
        mr_topn=1,
        mom_weight=1.0,
        mr_weight=0.0,
        use_enhanced_features=False,
        auto_optimize=True,   # <-- important
    )

    # --- Assertions on optimizer outputs and pass-through cfg ---
    assert isinstance(best_cfg, backend.HybridConfig)
    assert best_cfg == dummy_cfg
    pd.testing.assert_frame_equal(search_diag, diagnostics)

    # Ensure the cfg used in the backtest matches the optimizer cfg + tc_bps applied
    cfg = captured["cfg"]
    assert isinstance(cfg, backend.HybridConfig)
    assert cfg.momentum_top_n == dummy_cfg.momentum_top_n
    assert cfg.momentum_cap == dummy_cfg.momentum_cap
    assert cfg.mr_top_n == dummy_cfg.mr_top_n
    assert cfg.mom_weight == dummy_cfg.mom_weight
    assert cfg.mr_weight == dummy_cfg.mr_weight
    assert cfg.tc_bps == 15.0

    # Ensure optimizer was called with tc_bps propagated
    kwargs = captured.get("optimizer_kwargs", {})
    assert kwargs.get("tc_bps") == 15.0


def test_run_backtest_isa_dynamic_applies_vol_target(monkeypatch):
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

    monkeypatch.setattr(backend, "_prepare_universe_for_backtest", fake_prepare, raising=True)

    captured = {}
    import strategy_core

    def fake_run_hybrid_backtest(daily_prices, cfg, apply_vol_target=False, get_constituents=None):
        captured["apply_vol_target"] = apply_vol_target
        captured["target_vol"] = cfg.target_vol_annual
        idx = pd.date_range("2020-01-31", "2020-03-31", freq="M")
        return {
            "hybrid_rets": pd.Series(0.0, index=idx),
            "mom_turnover": pd.Series(0.0, index=idx),
            "mr_turnover": pd.Series(0.0, index=idx),
        }

    monkeypatch.setattr(strategy_core, "run_hybrid_backtest", fake_run_hybrid_backtest, raising=True)

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
        target_vol_annual=0.1,
        apply_vol_target=True,
    )

    assert captured.get("apply_vol_target") is True
    assert captured.get("target_vol") == 0.1
