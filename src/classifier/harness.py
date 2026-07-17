"""Classifier harness: unified train/predict interface for experiments.

Currently supports LightGBM via `lgbm_model` and provides stubs for GNN training.
"""
from typing import List, Any, Tuple
import pandas as pd
from src.classifier.lgbm_model import train_lgbm, predict_proba
from src.utils.graph_features import graph_list_to_bow


class ClassifierHarness:
    def __init__(self, backend: str = "lgbm"):
        self.backend = backend
        self.model = None

    def train(self, graphs: List[Any], labels: List[int], **kwargs):
        if self.backend == "lgbm":
            X = graph_list_to_bow(graphs)
            self.model = train_lgbm(X, labels, params=kwargs.get("params"), num_boost_round=kwargs.get("rounds", 100))
            return self.model
        elif self.backend == "gnn":
            raise NotImplementedError("GNN training harness not implemented yet")
        else:
            raise ValueError("unknown backend")

    def predict_proba(self, graphs: List[Any]) -> List[float]:
        if self.backend == "lgbm":
            X = graph_list_to_bow(graphs)
            return predict_proba(self.model, X)
        elif self.backend == "gnn":
            raise NotImplementedError("GNN predict not implemented yet")
        else:
            raise ValueError("unknown backend")

    def predict(self, graphs: List[Any], thresh: float = 0.5) -> List[int]:
        probs = self.predict_proba(graphs)
        return [1 if p >= thresh else 0 for p in probs]
