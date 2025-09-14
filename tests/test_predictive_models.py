import numpy as np
import pandas as pd
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import predictive_models as pm


def _make_data(n=30):
    idx = pd.date_range('2024-01-01', periods=n, freq='D')
    prices = pd.DataFrame({
        'AAA': 100 * (1 + 0.001) ** np.arange(n),
        'BBB': 100 * (1 - 0.001) ** np.arange(n)
    }, index=idx)
    rets = prices.pct_change()
    factors = {}
    for t in prices.columns:
        momentum = rets[t].rolling(3).mean()
        volatility = rets[t].rolling(5).std()
        quality = 1 / prices[t].rolling(5).mean()
        factors[t] = pd.concat(
            [momentum.rename('momentum'),
             volatility.rename('volatility'),
             quality.rename('quality')], axis=1
        )
    features = pd.concat(factors, axis=1)
    return prices, features


def test_predict_next_returns_outputs_series():
    prices, features = _make_data()
    preds = pm.predict_next_returns(prices, features)
    assert set(preds.index) == {'AAA', 'BBB'}
    assert preds.notna().all()
