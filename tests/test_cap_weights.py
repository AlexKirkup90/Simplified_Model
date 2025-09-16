import pandas as pd
import numpy as np
import strategy_core
import backend
import portfolio_utils


def test_cap_weights_consistency_and_normalization():
    weights = pd.Series([0.6, 0.3, 0.1], index=['A', 'B', 'C'])
    cap = 0.4
    capped = portfolio_utils.cap_weights(weights, cap=cap)
    assert np.allclose(capped.sum(), 1.0)
    pd.testing.assert_series_equal(strategy_core.cap_weights(weights, cap=cap), capped)
    pd.testing.assert_series_equal(backend.cap_weights(weights, cap=cap), capped)
    assert strategy_core.cap_weights is portfolio_utils.cap_weights
    assert backend.cap_weights is portfolio_utils.cap_weights

