"""Machine learning utilities for price prediction.

This module implements a light-weight gradient boosting model that operates on
factor data (e.g., momentum, volatility, quality).  The implementation avoids
third party dependencies to keep the project self-contained.  Features are
standardised before modelling and a simple time-series cross‑validation loop is
executed for each ticker.

The main entry point :func:`predict_next_returns` fits a model per ticker and
returns a forecast for the next period return.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class _StandardScaler:
    """Minimal standard scaler.

    Stores column-wise means and standard deviations and applies a standard
    z-score transformation.  A value of ``1`` is used whenever the standard
    deviation is ``0`` to avoid division errors.
    """

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "_StandardScaler":
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fitted.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class _GradientBoostingLinear:
    """Simple gradient boosting using linear base learners.

    The model iteratively fits linear regressions to the residuals of the
    previous stage.  Although significantly simpler than tree-based gradient
    boosting, it captures non-linear interactions through the boosting
    mechanism and is sufficient for small feature sets.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, ridge_lambda: float = 1e-4):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.ridge_lambda = ridge_lambda
        self.coefs_: list[np.ndarray] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_GradientBoostingLinear":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        pred = np.zeros_like(y)
        self.coefs_ = []
        if X.ndim != 2 or X.shape[1] == 0:
            return self
        eye = np.eye(X.shape[1])
        for _ in range(self.n_estimators):
            residual = y - pred
            xtx = X.T @ X + self.ridge_lambda * eye
            xty = X.T @ residual
            try:
                coef = np.linalg.solve(xtx, xty)
            except np.linalg.LinAlgError:
                coef, *_ = np.linalg.lstsq(X, residual, rcond=None)
            self.coefs_.append(coef)
            pred += self.learning_rate * X.dot(coef)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        pred = np.zeros(X.shape[0])
        for coef in self.coefs_:
            pred += self.learning_rate * X.dot(coef)
        return pred


def _time_series_cv_score(
    X: np.ndarray, y: np.ndarray, n_splits: int, n_estimators: int, learning_rate: float
) -> float:
    """Run a simple rolling-origin cross‑validation and return mean MSE."""

    n_samples = len(X)
    if n_samples < n_splits + 2:
        return float("nan")

    fold_size = n_samples // (n_splits + 1)
    scores = []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end = fold_size * (i + 2)
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]

        scaler = _StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = _GradientBoostingLinear(n_estimators=n_estimators, learning_rate=learning_rate)
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        scores.append(np.mean((pred - y_test) ** 2))

    return float(np.mean(scores)) if scores else float("nan")


def predict_next_returns(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    n_splits: int = 3,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
) -> pd.Series:
    """Forecast next-period returns for each ticker.

    Parameters
    ----------
    prices : DataFrame
        Historical prices indexed by date and with tickers as columns.
    features : DataFrame
        MultiIndex columns (ticker, factor).  Each inner DataFrame contains
        factor values aligned to ``prices``.
    n_splits : int, optional
        Number of cross‑validation splits for the rolling-origin evaluation.
    n_estimators, learning_rate : optional
        Gradient boosting hyper-parameters.

    Returns
    -------
    Series
        Per-ticker forecast of the next period return.  Tickers with
        insufficient data will be omitted.
    """

    if not isinstance(features.columns, pd.MultiIndex):
        raise ValueError("features must have MultiIndex columns (ticker, factor)")

    # Next-period returns serve as the prediction target.
    future_rets = prices.pct_change().shift(-1)
    preds: Dict[str, float] = {}

    for ticker in prices.columns:
        if ticker not in features.columns.get_level_values(0):
            continue

        feat = features[ticker]
        targ = future_rets[ticker].reindex(feat.index)
        data = pd.concat([feat, targ.rename("target")], axis=1).dropna()
        if len(data) < max(3, n_splits + 2):
            continue

        X = data.drop(columns=["target"]).to_numpy(dtype=float)
        y = data["target"].to_numpy(dtype=float)
        latest_feat = feat.iloc[[-1]].to_numpy(dtype=float)

        candidate_grid = {
            (int(n_estimators), float(learning_rate)),
            (max(20, int(n_estimators // 2)), max(0.01, float(learning_rate) / 2)),
            (min(200, int(n_estimators * 2)), min(0.3, float(learning_rate) * 1.5)),
        }

        best_score = np.inf
        best_params = (int(n_estimators), float(learning_rate))
        for n_est, lr in candidate_grid:
            score = _time_series_cv_score(X, y, n_splits, int(n_est), float(lr))
            if np.isnan(score):
                continue
            if score < best_score:
                best_score = score
                best_params = (int(n_est), float(lr))

        scaler = _StandardScaler()
        X_s = scaler.fit_transform(X)
        latest_s = scaler.transform(latest_feat)

        model = _GradientBoostingLinear(n_estimators=best_params[0], learning_rate=best_params[1])
        model.fit(X_s, y)
        prediction = float(model.predict(latest_s)[0])
        preds[ticker] = float(np.clip(prediction, -0.30, 0.30))

    return pd.Series(preds)
