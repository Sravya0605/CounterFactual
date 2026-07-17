"""LightGBM classifier wrapper for coalesced behavior-graph features."""
import lightgbm as lgb
import pandas as pd
from typing import Any, Dict, List


def train_lgbm(X: pd.DataFrame, y: List[int], params: Dict = None, num_boost_round: int = 100):
    params = params or {"objective": "binary", "metric": "binary_logloss"}
    ds = lgb.Dataset(X, label=y)
    model = lgb.train(params, ds, num_boost_round=num_boost_round)
    return model


def predict_proba(model: Any, X: pd.DataFrame) -> List[float]:
    return model.predict(X).tolist()


def predict(model: Any, X: pd.DataFrame, thresh: float = 0.5) -> List[int]:
    probs = predict_proba(model, X)
    return [1 if p >= thresh else 0 for p in probs]
