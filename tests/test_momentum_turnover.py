import numpy as np
import pandas as pd
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import strategy_core as sc


def test_run_backtest_momentum_turnover_half_l1():
    idx = pd.date_range('2023-01-31', periods=6, freq='M')
    prices = pd.DataFrame({
        'AAA': 100 * (1.05) ** np.arange(6),
        'BBB': 100 * (1.00) ** np.arange(6),
    }, index=idx)
    daily = prices.resample('D').ffill()
    _, tno = sc.run_backtest_momentum(daily, lookback_m=1, top_n=1, cap=1.0)
    # First turnover after weights exist should be 0.5
    assert np.isclose(tno[tno > 0].iloc[0], 0.5)
