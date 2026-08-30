"""LightGBM classifier wrapper for coalesced behavior-graph features."""
from typing import Any, Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

DEFAULT_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "feature_fraction": 0.6,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 25,
    "max_depth": 5,
    "num_leaves": 20,
    "lambda_l1": 1.5,
    "lambda_l2": 5.0,
    "min_gain_to_split": 0.02,
    "learning_rate": 0.03,
    "verbosity": -1,
}


class SigmoidCalibrator:
    def __init__(self, base_model: Any, calibrator: LogisticRegression):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X: Any) -> np.ndarray:
        frame = pd.DataFrame(X)
        raw_scores = self.base_model.predict_proba(frame)[:, 1]
        calibrated = self.calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
        return np.clip(calibrated, 1e-6, 1.0 - 1e-6)

    def predict(self, X: Any, thresh: float = 0.5) -> List[int]:
        probs = self.predict_proba(X)
        return [1 if p >= thresh else 0 for p in probs]


def _as_dataframe(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X)


def _fallback_sigmoid_calibration(base_model: Any, frame: pd.DataFrame, y: List[int]) -> SigmoidCalibrator:
    raw_scores = base_model.predict_proba(frame)[:, 1]
    clf = LogisticRegression(max_iter=1000)
    clf.fit(raw_scores.reshape(-1, 1), y)
    return SigmoidCalibrator(base_model, clf)


def train_lgbm(X: Any, y: List[int], params: Dict = None, num_boost_round: int = 200, calibrate: bool = True):
    frame = _as_dataframe(X)
    params = {**DEFAULT_LGBM_PARAMS, **(params or {})}

    base_model = lgb.LGBMClassifier(
        n_estimators=num_boost_round,
        **params,
    )
    base_model.fit(frame, y)

    if not calibrate:
        return base_model

    class_counts = np.bincount(np.asarray(y, dtype=int))
    if len(class_counts) < 2 or np.min(class_counts) < 2:
        return _fallback_sigmoid_calibration(base_model, frame, y)

    try:
        calibrated = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv=2)
        calibrated.fit(frame, y)
        return calibrated
    except ValueError:
        return _fallback_sigmoid_calibration(base_model, frame, y)


def predict_proba(model: Any, X: Any) -> List[float]:
    frame = _as_dataframe(X)
    if isinstance(model, SigmoidCalibrator):
        return model.predict_proba(frame).tolist()
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(frame)
        if probs.ndim == 2:
            return probs[:, 1].tolist()
        return probs.tolist()
    return model.predict(frame).tolist()


def predict(model: Any, X: Any, thresh: float = 0.5) -> List[int]:
    probs = predict_proba(model, X)
    return [1 if p >= thresh else 0 for p in probs]
