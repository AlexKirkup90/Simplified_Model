import pandas as pd
import numpy as np
import strategy_core
import backend


def test_cap_weights_consistency_and_normalization():
    weights = pd.Series([0.6, 0.3, 0.1], index=['A', 'B', 'C'])
    cap = 0.4
    s = strategy_core.cap_weights(weights, cap=cap, max_iter=10, tol=1e-12)
    b = backend.cap_weights(weights, cap=cap, max_iter=10, tol=1e-12)
    assert np.allclose(s.sum(), 1.0)
    assert np.allclose(b.sum(), 1.0)
    pd.testing.assert_series_equal(s, b)

