"""Machine learning utilities for price prediction.

This module now supports multiple feature sets (e.g., momentum, quality,
sentiment) and pluggable model classes.  Two reference implementations are
provided: a linear gradient boosting model and a lightweight random forest
ensemble.  Both models avoid third-party dependencies to keep the project
self-contained.  Features can be standardised automatically and a
time-series cross-validation loop evaluates candidate hyper-parameters per
ticker.  Model-selection metrics are logged for downstream monitoring.

The main entry point :func:`predict_next_returns` fits a model per ticker and
returns a forecast for the next period return.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Dict

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


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


@dataclass
class _IdentityScaler:
    """No-op scaler used for models that operate on raw features."""

    def fit(self, X: np.ndarray) -> "_IdentityScaler":  # pragma: no cover - trivial
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - trivial
        return np.asarray(X, dtype=float)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - trivial
        return self.transform(X)


class _GradientBoostingLinear:
    """Simple gradient boosting using linear base learners.

    The model iteratively fits linear regressions to the residuals of the
    previous stage.  Although significantly simpler than tree-based gradient
    boosting, it captures non-linear interactions through the boosting
    mechanism and is sufficient for small feature sets.
    """

    default_scaler = _StandardScaler

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


@dataclass
class _TreeNode:
    feature_index: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None
    value: float | None = None


class _DecisionTreeRegressor:
    """Minimal regression tree used for the random forest ensemble."""

    def __init__(
        self,
        max_depth: int | None = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | float | None = "sqrt",
        random_state: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root_: _TreeNode | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_DecisionTreeRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            self.root_ = _TreeNode(value=float(np.nanmean(y) if len(y) else 0.0))
            return self
        rng = np.random.default_rng(self.random_state)
        self.root_ = self._grow_tree(X, y, depth=0, rng=rng)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None:
            raise ValueError("The decision tree has not been fitted.")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_row(row, self.root_) for row in X], dtype=float)

    # ------------------------------------------------------------------
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator) -> _TreeNode:
        if len(y) == 0:
            return _TreeNode(value=0.0)
        if self.max_depth is not None and depth >= self.max_depth:
            return _TreeNode(value=float(np.mean(y)))
        if len(y) < self.min_samples_split or np.allclose(y, y[0]):
            return _TreeNode(value=float(np.mean(y)))

        feature_indices = np.arange(X.shape[1])
        max_features = self._resolve_max_features(X.shape[1])
        if max_features < len(feature_indices):
            feature_indices = rng.choice(feature_indices, size=max_features, replace=False)

        best_feat, best_thresh = self._best_split(X, y, feature_indices)
        if best_feat is None or best_thresh is None:
            return _TreeNode(value=float(np.mean(y)))

        mask = X[:, best_feat] <= best_thresh
        left_X, right_X = X[mask], X[~mask]
        left_y, right_y = y[mask], y[~mask]
        if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
            return _TreeNode(value=float(np.mean(y)))

        left = self._grow_tree(left_X, left_y, depth + 1, rng)
        right = self._grow_tree(right_X, right_y, depth + 1, rng)
        return _TreeNode(feature_index=int(best_feat), threshold=float(best_thresh), left=left, right=right)

    def _resolve_max_features(self, n_features: int) -> int:
        mf = self.max_features
        if mf is None:
            return n_features
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            if mf == "log2":
                return max(1, int(np.log2(n_features)))
            raise ValueError(f"Unsupported max_features option: {mf}")
        if isinstance(mf, float):
            if 0 < mf <= 1:
                return max(1, int(np.ceil(mf * n_features)))
            mf = int(mf)
        if isinstance(mf, int):
            return max(1, min(n_features, mf))
        return n_features

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray
    ) -> tuple[int | None, float | None]:
        best_feat: int | None = None
        best_thresh: float | None = None
        best_score = float("inf")
        for feat in feature_indices:
            column = X[:, feat]
            order = np.argsort(column)
            column_sorted = column[order]
            y_sorted = y[order]
            unique_mask = np.diff(column_sorted) > 1e-12
            if not np.any(unique_mask):
                continue
            split_candidates = np.where(unique_mask)[0]
            for idx in split_candidates:
                split_index = idx + 1
                if split_index < self.min_samples_leaf or len(y) - split_index < self.min_samples_leaf:
                    continue
                left_y = y_sorted[:split_index]
                right_y = y_sorted[split_index:]
                thresh = (column_sorted[split_index - 1] + column_sorted[split_index]) / 2.0
                left_var = np.var(left_y)
                right_var = np.var(right_y)
                score = left_var * len(left_y) + right_var * len(right_y)
                if score < best_score:
                    best_score = score
                    best_feat = int(feat)
                    best_thresh = float(thresh)
        return best_feat, best_thresh

    def _predict_row(self, row: np.ndarray, node: _TreeNode) -> float:
        while node.value is None:
            if node.feature_index is None or node.threshold is None or node.left is None or node.right is None:
                break
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return float(node.value if node.value is not None else 0.0)


class _RandomForestRegressor:
    """Small random forest ensemble built on top of :class:`_DecisionTreeRegressor`."""

    default_scaler = _IdentityScaler

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int | None = 4,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | float | None = "sqrt",
        bootstrap: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees_: list[_DecisionTreeRegressor] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RandomForestRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or len(X) == 0:
            self.trees_ = []
            return self
        rng = np.random.default_rng(self.random_state)
        self.trees_ = []
        for _ in range(self.n_estimators):
            if self.bootstrap:
                indices = rng.choice(len(X), size=len(X), replace=True)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample, y_sample = X, y
            tree_seed = None if self.random_state is None else int(rng.integers(0, 2**32 - 1))
            tree = _DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_seed,
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise ValueError("The random forest has not been fitted.")
        X = np.asarray(X, dtype=float)
        preds = np.column_stack([tree.predict(X) for tree in self.trees_])
        return np.mean(preds, axis=1)


@dataclass(frozen=True)
class _ModelSpec:
    name: str
    builder: Callable[..., Any]
    param_grid: Sequence[Mapping[str, Any]]
    scaler_factory: Callable[[], Any]
    default_params: Mapping[str, Any] | None = None


MODEL_REGISTRY: Dict[str, _ModelSpec] = {
    "linear_boost": _ModelSpec(
        name="linear_boost",
        builder=_GradientBoostingLinear,
        default_params={"n_estimators": 100, "learning_rate": 0.1, "ridge_lambda": 1e-4},
        param_grid=[
            {},
            {"n_estimators": 60, "learning_rate": 0.05},
            {"n_estimators": 150, "learning_rate": 0.08},
        ],
        scaler_factory=_StandardScaler,
    ),
    "random_forest": _ModelSpec(
        name="random_forest",
        builder=_RandomForestRegressor,
        default_params={
            "n_estimators": 60,
            "max_depth": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "bootstrap": True,
        },
        param_grid=[
            {},
            {"n_estimators": 100, "max_depth": 5},
            {"n_estimators": 120, "max_depth": None, "min_samples_leaf": 1},
        ],
        scaler_factory=_IdentityScaler,
    ),
}


def _resolve_model_spec(model: str | _ModelSpec | Callable[..., Any]) -> _ModelSpec:
    if isinstance(model, _ModelSpec):
        return model
    if isinstance(model, str):
        try:
            return MODEL_REGISTRY[model]
        except KeyError as exc:  # pragma: no cover - defensive
            available = ", ".join(sorted(MODEL_REGISTRY))
            raise ValueError(f"Unknown model '{model}'. Available: {available}") from exc
    if callable(model):
        scaler_factory = getattr(model, "default_scaler", _StandardScaler)
        return _ModelSpec(
            name=getattr(model, "__name__", "custom_model"),
            builder=model,
            default_params={},
            param_grid=[{}],
            scaler_factory=scaler_factory,
        )
    raise TypeError("model must be a string key, ModelSpec or callable")


def _normalise_feature_sets(feature_sets: Mapping[str, pd.DataFrame] | pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if isinstance(feature_sets, pd.DataFrame):
        if not isinstance(feature_sets.columns, pd.MultiIndex):
            raise ValueError("features must have MultiIndex columns (ticker, factor)")
        return {"core": feature_sets}
    if not isinstance(feature_sets, Mapping):
        raise TypeError("feature_sets must be a DataFrame or mapping of DataFrames")
    normalised: Dict[str, pd.DataFrame] = {}
    for name, frame in feature_sets.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("feature set values must be DataFrames")
        if not isinstance(frame.columns, pd.MultiIndex):
            raise ValueError("feature set DataFrames must use MultiIndex columns (ticker, factor)")
        normalised[str(name)] = frame
    return normalised


def _collect_ticker_features(
    feature_sets: Mapping[str, pd.DataFrame], ticker: str, index: pd.Index
) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    for set_name, frame in feature_sets.items():
        if ticker not in frame.columns.get_level_values(0):
            continue
        feat = frame[ticker].copy()
        feat = feat.reindex(index)
        feat.columns = [f"{set_name}__{str(col)}" for col in feat.columns]
        frames.append(feat)
    if not frames:
        return None
    combined = pd.concat(frames, axis=1)
    return combined


def _unique_dicts(dicts: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    unique: list[Dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for mapping in dicts:
        items = tuple(sorted(mapping.items()))
        if items in seen:
            continue
        seen.add(items)
        unique.append(dict(mapping))
    return unique


def _build_candidate_params(
    spec: _ModelSpec,
    model_params: Mapping[str, Any] | None,
    candidate_params: Sequence[Mapping[str, Any]] | None,
    random_state: int | None,
) -> list[Dict[str, Any]]:
    base = dict(spec.default_params or {})
    if model_params is not None:
        candidates = [{**base, **dict(model_params)}]
    else:
        params_iter = candidate_params if candidate_params is not None else spec.param_grid
        if not params_iter:
            params_iter = [{}]
        candidates = [{**base, **dict(params)} for params in params_iter]
        candidates.insert(0, base)
    if random_state is not None and _builder_accepts_param(spec.builder, "random_state"):
        for params in candidates:
            params.setdefault("random_state", random_state)
    return _unique_dicts(candidates)


def _builder_accepts_param(builder: Callable[..., Any], param: str) -> bool:
    try:
        import inspect

        signature = inspect.signature(builder)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False
    return param in signature.parameters


def _time_series_cv_score(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    model_builder: Callable[[], Any],
    scaler_factory: Callable[[], Any],
) -> float:
    n_samples = len(X)
    if n_splits <= 0 or n_samples < n_splits + 2:
        return float("nan")

    fold_size = n_samples // (n_splits + 1)
    if fold_size == 0:
        return float("nan")

    scores: list[float] = []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end = fold_size * (i + 2)
        if test_end > n_samples:
            break
        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[train_end:test_end], y[train_end:test_end]
        if len(X_test) == 0 or len(X_train) == 0:
            continue
        scaler = scaler_factory()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = model_builder()
        model.fit(X_train_s, y_train)
        pred = np.asarray(model.predict(X_test_s), dtype=float)
        scores.append(float(np.mean((pred - y_test) ** 2)))

    return float(np.mean(scores)) if scores else float("nan")


def _ticker_seed(base_seed: int | None, ticker: str) -> int | None:
    if base_seed is None:
        return None
    return abs(hash((ticker, base_seed))) % (2**32 - 1)


def predict_next_returns(
    prices: pd.DataFrame,
    feature_sets: Mapping[str, pd.DataFrame] | pd.DataFrame,
    n_splits: int = 3,
    *,
    model: str | _ModelSpec | Callable[..., Any] = "linear_boost",
    model_params: Mapping[str, Any] | None = None,
    candidate_params: Sequence[Mapping[str, Any]] | None = None,
    random_state: int | None = None,
    return_metrics: bool = False,
    logger: logging.Logger | None = None,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """Forecast next-period returns for each ticker.

    Parameters
    ----------
    prices : DataFrame
        Historical prices indexed by date with tickers as columns.
    feature_sets : DataFrame or mapping of DataFrames
        Either a single feature table with MultiIndex columns or a mapping of
        named feature tables.  Tables must use ``(ticker, factor)`` columns.
    n_splits : int, optional
        Number of cross-validation splits for the rolling-origin evaluation.
    model : str or callable, optional
        Model to use.  ``"linear_boost"`` (default) uses the gradient boosting
        model.  ``"random_forest"`` selects the ensemble.  A custom callable can
        also be supplied.
    model_params : mapping, optional
        Hyper-parameters applied to every ticker.  When provided this disables
        the internal parameter grid search.
    candidate_params : sequence of mappings, optional
        Custom parameter grid explored during model selection.
    random_state : int, optional
        Seed for stochastic models.  The seed is varied per ticker for
        reproducibility while maintaining diversity.
    return_metrics : bool, optional
        When ``True`` the function returns ``(predictions, metrics)`` where the
        metrics frame summarises model selection diagnostics per ticker.
    logger : logging.Logger, optional
        Logger used to record model selection information.  Falls back to the
        module logger when omitted.

    Returns
    -------
    Series or tuple
        Per-ticker forecast of the next-period return.  When
        ``return_metrics`` is ``True`` a tuple ``(predictions, metrics)`` is
        returned where ``metrics`` is a DataFrame.
    """

    logger = logger or LOGGER
    spec = _resolve_model_spec(model)
    feature_sets_norm = _normalise_feature_sets(feature_sets)

    future_rets = prices.pct_change().shift(-1)
    preds: Dict[str, float] = {}
    metrics_records: list[Dict[str, Any]] = []

    for ticker in prices.columns:
        ticker_features = _collect_ticker_features(feature_sets_norm, ticker, prices.index)
        if ticker_features is None:
            metrics_records.append({"ticker": ticker, "model": spec.name, "status": "missing_features"})
            continue

        ticker_features = ticker_features.sort_index().ffill()
        if ticker_features.iloc[-1].isna().any():
            metrics_records.append({"ticker": ticker, "model": spec.name, "status": "nan_latest_features"})
            continue

        target = future_rets[ticker].reindex(ticker_features.index)
        data = pd.concat([ticker_features, target.rename("target")], axis=1).dropna()
        if len(data) < max(3, n_splits + 2):
            metrics_records.append({"ticker": ticker, "model": spec.name, "status": "insufficient_history"})
            continue

        X = data.drop(columns=["target"]).to_numpy(dtype=float)
        y = data["target"].to_numpy(dtype=float)
        latest_feat = ticker_features.iloc[[-1]].to_numpy(dtype=float)

        seed = _ticker_seed(random_state, ticker)
        candidates = _build_candidate_params(spec, model_params, candidate_params, seed)

        best_score = float("inf")
        best_params: Dict[str, Any] | None = None
        candidate_scores: list[Dict[str, Any]] = []
        for params in candidates:
            model_builder = lambda params=params: spec.builder(**params)
            score = _time_series_cv_score(X, y, n_splits, model_builder, spec.scaler_factory)
            candidate_scores.append({"params": dict(params), "cv_mse": score})
            logger.debug(
                "Ticker %s | model=%s | candidate_params=%s | cv_mse=%s",
                ticker,
                spec.name,
                params,
                score,
            )
            if not np.isnan(score) and score < best_score:
                best_score = score
                best_params = dict(params)

        if best_params is None:
            best_params = dict(candidates[0])
            best_score = float("nan")

        scaler = spec.scaler_factory()
        X_s = scaler.fit_transform(X)
        latest_s = scaler.transform(latest_feat)
        model_instance = spec.builder(**best_params)
        model_instance.fit(X_s, y)
        prediction = float(np.asarray(model_instance.predict(latest_s), dtype=float)[0])
        preds[ticker] = float(np.clip(prediction, -0.30, 0.30))

        metrics_record = {
            "ticker": ticker,
            "model": spec.name,
            "status": "fitted",
            "params": best_params,
            "cv_mse": float(best_score) if np.isfinite(best_score) else np.nan,
            "n_observations": len(data),
            "candidate_scores": candidate_scores,
        }
        metrics_records.append(metrics_record)
        logger.info(
            "Ticker %s | model=%s | params=%s | cv_mse=%s | n=%d",
            ticker,
            spec.name,
            best_params,
            metrics_record["cv_mse"],
            len(data),
        )

    predictions = pd.Series(preds)
    if return_metrics:
        metrics_df = pd.DataFrame(metrics_records)
        return predictions, metrics_df
    return predictions
