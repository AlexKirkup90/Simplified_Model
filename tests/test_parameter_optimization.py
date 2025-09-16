import numpy as np
import pandas as pd

import optimizer


def test_grid_search_hybrid_selects_best_params():
    # Build ~13 months of synthetic daily prices
    dates = pd.date_range('2020-01-01', periods=400, freq='D')
    prices = pd.DataFrame({
        'A': np.linspace(100, 200, len(dates)),  # strong uptrend
        'B': np.linspace(100, 80, len(dates)),   # downtrend
    }, index=dates)

    best_cfg, results = optimizer.grid_search_hybrid(prices, {
        'momentum_top_n': [1, 2],
        'mom_weight': [1.0],
        'mr_weight': [0.0],
    }, tc_bps=5.0, apply_vol_target=True)

    assert set(results['momentum_top_n']) == {1, 2}
    assert best_cfg.momentum_top_n == 1
    assert 'sharpe' in results.columns
    assert results['sharpe'].notna().all()


def test_grid_search_hybrid_passes_tc_and_vol(monkeypatch):
    prices = pd.DataFrame({'A': [1, 2, 3]}, index=pd.date_range('2020-01-01', periods=3))

    captured = {}

    def fake_run_hybrid_backtest(daily_prices, cfg, apply_vol_target=False):
        captured['tc_bps'] = cfg.tc_bps
        captured['apply_vol_target'] = apply_vol_target
        idx = pd.date_range('2020-01-31', periods=3, freq='M')
        net = pd.Series([0.02, -0.01, 0.015], index=idx)
        gross = pd.Series([-0.02, 0.01, -0.015], index=idx)
        return {'hybrid_rets': gross, 'hybrid_rets_net': net}

    monkeypatch.setattr(optimizer, 'run_hybrid_backtest', fake_run_hybrid_backtest)

    best_cfg, results = optimizer.grid_search_hybrid(
        prices,
        {'momentum_top_n': [1]},
        tc_bps=12.0,
        apply_vol_target=True,
    )

    assert captured['tc_bps'] == 12.0
    assert captured['apply_vol_target'] is True
    idx = pd.date_range('2020-01-31', periods=3, freq='M')
    expected = optimizer._annualized_sharpe(pd.Series([0.02, -0.01, 0.015], index=idx))
    assert np.isclose(results['sharpe'].iloc[0], expected)


def test_bayesian_optimize_hybrid_prefers_best_candidate(monkeypatch):
    prices = pd.DataFrame(
        {
            "AAA": [1.0, 1.1, 1.2, 1.3],
            "BBB": [1.0, 0.9, 0.95, 1.0],
        },
        index=pd.date_range("2020-01-01", periods=4, freq="D"),
    )

    returns_map = {
        5: [0.01, -0.005, 0.009],
        10: [0.015, -0.004, 0.016],
        15: [0.005, -0.006, 0.007],
    }

    def fake_run_hybrid_backtest(daily_prices, cfg, apply_vol_target=False):
        idx = pd.date_range("2020-01-31", periods=3, freq="M")
        vals = returns_map[cfg.momentum_top_n]
        series = pd.Series(vals, index=idx)
        return {"hybrid_rets": series}

    monkeypatch.setattr(optimizer, "run_hybrid_backtest", fake_run_hybrid_backtest)

    best_cfg, diagnostics = optimizer.bayesian_optimize_hybrid(
        prices,
        search_space={"momentum_top_n": [5, 10, 15]},
        population_size=4,
        n_iter=2,
        seed=42,
    )

    assert isinstance(best_cfg, optimizer.HybridConfig)
    assert best_cfg.momentum_top_n == 10
    assert not diagnostics.empty
    assert 10 in diagnostics["momentum_top_n"].values
