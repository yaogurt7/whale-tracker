import os

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, QuantileTransformer, RobustScaler


class WhaleFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create row-local whale-flow features without changing the raw dataset."""

    def __init__(self, mode="compact"):
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        flow = X[:, [0]]
        hhi = X[:, [1]]
        top10 = X[:, [2]]
        gini = X[:, [3]]
        vol = X[:, [4]]
        eps = 1e-9

        log_vol = np.log1p(np.maximum(vol, 0.0))
        abs_flow = np.abs(flow)
        signed_log_flow = np.sign(flow) * np.log1p(abs_flow)
        concentration = 0.50 * hhi + 0.50 * top10
        inequality = 0.50 * gini + 0.50 * top10
        flow_per_vol = flow / (log_vol + 1.0)
        flow_per_conc = flow / (concentration + eps)
        whale_pressure = signed_log_flow * concentration
        crowding_pressure = signed_log_flow * inequality

        compact = np.hstack(
            [
                X,
                log_vol,
                signed_log_flow,
                concentration,
                inequality,
                flow_per_vol,
                flow_per_conc,
                whale_pressure,
                crowding_pressure,
                hhi * top10,
                gini * top10,
                hhi * gini,
            ]
        )
        if self.mode == "compact":
            return compact

        squared = np.hstack(
            [
                flow * flow,
                hhi * hhi,
                top10 * top10,
                gini * gini,
                log_vol * log_vol,
                whale_pressure * whale_pressure,
            ]
        )
        crosses = np.hstack(
            [
                flow * hhi,
                flow * top10,
                flow * gini,
                signed_log_flow * hhi,
                signed_log_flow * top10,
                signed_log_flow * gini,
                log_vol * hhi,
                log_vol * top10,
                log_vol * gini,
            ]
        )
        full = np.hstack([compact, squared, crosses])
        if self.mode == "full":
            return full

        dense = np.hstack(
            [
                full,
                np.sqrt(np.maximum(hhi, 0.0)),
                np.sqrt(np.maximum(top10, 0.0)),
                np.sqrt(np.maximum(gini, 0.0)),
                np.sqrt(np.maximum(log_vol, 0.0)),
                np.tanh(flow_per_vol),
                np.tanh(flow_per_conc),
                whale_pressure / (inequality + eps),
                crowding_pressure / (concentration + eps),
            ]
        )
        return dense


class CausalRollingFeatures(BaseEstimator, TransformerMixin):
    """Lag-only rolling context computed in the observed row order."""

    def __init__(self, windows=(5, 20), include_engineered=True):
        self.windows = windows
        self.include_engineered = include_engineered
        self.engineer_ = WhaleFeatureEngineer("compact")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        base = self.engineer_.transform(X) if self.include_engineered else X
        pieces = [base]
        for window in self.windows:
            for col in (0, 1, 2, 3, 4):
                values = X[:, col]
                row_ids = np.arange(len(values))
                starts = np.maximum(0, row_ids - window)
                prefix = np.concatenate(([0.0], np.cumsum(values)))
                counts = np.maximum(1, row_ids - starts)
                lag_mean = (prefix[row_ids] - prefix[starts]) / counts
                lag_mean[0] = 0.0
                pieces.append(lag_mean.reshape(-1, 1))
                pieces.append((values - lag_mean).reshape(-1, 1))
        return np.hstack(pieces)


def _rf(**kwargs):
    params = dict(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=100,
        max_features=None,
        n_jobs=-1,
        random_state=42,
    )
    params.update(kwargs)
    return RandomForestClassifier(**params)


def _experiments():
    return {
        "baseline_rf": _rf(),
        "rf_compact_fe": Pipeline([("features", WhaleFeatureEngineer("compact")), ("rf", _rf())]),
        "rf_full_fe": Pipeline([("features", WhaleFeatureEngineer("full")), ("rf", _rf())]),
        "rf_dense_fe": Pipeline([("features", WhaleFeatureEngineer("dense")), ("rf", _rf())]),
        "rf_full_depth4_leaf50": Pipeline(
            [("features", WhaleFeatureEngineer("full")), ("rf", _rf(max_depth=4, min_samples_leaf=50))]
        ),
        "rf_full_depth6_leaf150": Pipeline(
            [("features", WhaleFeatureEngineer("full")), ("rf", _rf(max_depth=6, min_samples_leaf=150))]
        ),
        "rf_temporal_fe": Pipeline([("features", CausalRollingFeatures()), ("rf", _rf())]),
        "rf_temporal_depth4": Pipeline(
            [("features", CausalRollingFeatures()), ("rf", _rf(max_depth=4, min_samples_leaf=75))]
        ),
        "hgb_full_fe": Pipeline(
            [
                ("features", WhaleFeatureEngineer("full")),
                (
                    "hgb",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        learning_rate=0.03,
                        max_leaf_nodes=31,
                        l2_regularization=0.1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hgb_dense_fe": Pipeline(
            [
                ("features", WhaleFeatureEngineer("dense")),
                (
                    "hgb",
                    HistGradientBoostingClassifier(
                        max_iter=150,
                        learning_rate=0.04,
                        max_leaf_nodes=15,
                        l2_regularization=0.2,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "extra_full_fe": Pipeline(
            [
                ("features", WhaleFeatureEngineer("full")),
                (
                    "extra",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_depth=5,
                        min_samples_leaf=100,
                        max_features=None,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "gb_full_fe": Pipeline(
            [
                ("features", WhaleFeatureEngineer("full")),
                (
                    "gb",
                    GradientBoostingClassifier(
                        n_estimators=120,
                        learning_rate=0.03,
                        max_depth=2,
                        subsample=0.8,
                        min_samples_leaf=80,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "logit_compact_quantile": Pipeline(
            [
                ("features", WhaleFeatureEngineer("compact")),
                ("quantile", QuantileTransformer(output_distribution="normal", random_state=42)),
                ("logit", LogisticRegression(C=0.5, max_iter=1000)),
            ]
        ),
        "logit_poly_compact": Pipeline(
            [
                ("features", WhaleFeatureEngineer("compact")),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scale", RobustScaler()),
                ("logit", LogisticRegression(C=0.05, max_iter=1000)),
            ]
        ),
    }


def build_model():
    """Returns the requested experiment, defaulting to the best current model."""
    experiment = os.environ.get("WHALE_EXPERIMENT", "rf_temporal_fe")
    return _experiments().get(experiment, _experiments()["baseline_rf"])
