import numpy as np
import pandas as pd

from optimizer import grid_search_hybrid


def test_grid_search_hybrid_selects_best_params():
    # Build ~13 months of synthetic daily prices
    dates = pd.date_range('2020-01-01', periods=400, freq='D')
    prices = pd.DataFrame({
        'A': np.linspace(100, 200, len(dates)),  # strong uptrend
        'B': np.linspace(100, 80, len(dates)),   # downtrend
    }, index=dates)

    best_cfg, results = grid_search_hybrid(prices, {
        'momentum_top_n': [1, 2],
        'mom_weight': [1.0],
        'mr_weight': [0.0],
    })

    assert set(results['momentum_top_n']) == {1, 2}
    assert best_cfg.momentum_top_n == 1
    assert 'sharpe' in results.columns
    assert results['sharpe'].notna().all()
