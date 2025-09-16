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


def test_predict_next_returns_supports_multiple_feature_sets():
    prices, features = _make_data()
    quality = features.copy() * 0.5
    sentiment = features.copy() * -1.0
    feature_sets = {
        'core': features,
        'quality': quality,
        'sentiment': sentiment
    }
    preds = pm.predict_next_returns(prices, feature_sets, model='random_forest', random_state=123)
    assert set(preds.index) == {'AAA', 'BBB'}
    assert preds.notna().all()


def test_predict_next_returns_can_return_metrics():
    prices, features = _make_data()
    preds, metrics = pm.predict_next_returns(
        prices,
        {'core': features},
        model='linear_boost',
        candidate_params=[{'n_estimators': 40, 'learning_rate': 0.05}],
        return_metrics=True,
    )
    assert set(preds.index) == {'AAA', 'BBB'}
    assert metrics['ticker'].tolist() == ['AAA', 'BBB']
    assert all(status == 'fitted' for status in metrics['status'])
    assert metrics['cv_mse'].notna().all()
    assert metrics['candidate_scores'].apply(lambda rows: isinstance(rows, list) and len(rows) >= 1).all()
