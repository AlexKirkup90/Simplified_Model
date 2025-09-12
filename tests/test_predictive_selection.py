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


def test_predictive_signals_and_weights():
    prices = _make_monthly_prices().iloc[:4]
    signals = sc.predictive_signals(prices, lookback_m=3)
    assert signals['AAA'] > 0
    assert signals['BBB'] < 0

    w, scores = sc.build_predictive_weights(prices, lookback_m=3, top_n=1, cap=1.0)
    assert list(w.index) == ['AAA']
    assert np.isclose(w.iloc[0], 1.0)
    assert list(scores.index) == ['AAA']


def test_run_backtest_predictive_produces_positive_returns():
    monthly = _make_monthly_prices()
    daily = monthly.resample('D').ffill()
    rets, tno = sc.run_backtest_predictive(daily, lookback_m=2, top_n=1, cap=1.0)
    assert rets.sum() > 0
    assert (tno > 0).any()

