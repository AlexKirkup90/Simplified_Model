import numpy as np
import pandas as pd
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import strategy_core as sc


def _make_monthly_prices():
    idx = pd.date_range('2023-01-31', periods=6, freq='M')
    prices = pd.DataFrame({
        'AAA': 100 * (1.01) ** np.arange(6),  # +1% each month
        'BBB': 100 * (0.99) ** np.arange(6),  # -1% each month
    }, index=idx)
    return prices


def test_hybrid_includes_predictive_sleeve():
    monthly = _make_monthly_prices()
    daily = monthly.resample('D').ffill()
    cfg = sc.HybridConfig(
        momentum_lookback_m=2,
        momentum_top_n=1,
        momentum_cap=1.0,
        mr_lookback_days=1,
        mr_top_n=1,
        mr_long_ma_days=5,
        mom_weight=0.0,
        mr_weight=0.0,
        predictive_lookback_m=2,
        predictive_top_n=1,
        predictive_cap=1.0,
        predictive_weight=1.0,
    )
    res = sc.run_hybrid_backtest(daily, cfg)
    pred_rets, _ = sc.run_backtest_predictive(daily, lookback_m=2, top_n=1, cap=1.0)
    assert res['pred_rets'].equals(pred_rets)
    assert res['hybrid_rets'].equals(pred_rets)
