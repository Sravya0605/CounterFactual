"""Classifier harness with a lightweight fallback for the prototype."""
from typing import List, Any

from src.utils.graph_features import graph_list_to_bow


class ClassifierHarness:
    def __init__(self, backend: str = "heuristic"):
        self.backend = (backend or "heuristic").lower()
        self.model = None
        self._initialize_backend()

    def _initialize_backend(self):
        if self.backend in {"heuristic", "sklearn"}:
            from src.classifier.heuristic_model import HeuristicClassifier

            self.model = HeuristicClassifier()
            return

        if self.backend == "lgbm":
            try:
                from src.classifier.lgbm_model import train_lgbm, predict_proba as lgbm_predict_proba
            except Exception:
                from src.classifier.heuristic_model import HeuristicClassifier

                self.model = HeuristicClassifier()
                self.backend = "heuristic"
                return

            self._train_lgbm = train_lgbm
            self._predict_lgbm = lgbm_predict_proba
            self.model = None
            return

        if self.backend == "gnn":
            try:
                from src.classifier.gnn_model import SimpleGCN
            except Exception:
                from src.classifier.heuristic_model import HeuristicClassifier

                self.model = HeuristicClassifier()
                self.backend = "heuristic"
                return

            self.model = SimpleGCN(in_channels=1)

    def train(self, graphs: List[Any], labels: List[int], **kwargs):
        if self.backend == "lgbm":
            X = graph_list_to_bow(graphs)
            self.model = self._train_lgbm(X, labels, params=kwargs.get("params"), num_boost_round=kwargs.get("rounds", 100))
            return self.model
        if self.backend == "heuristic":
            self.model = self.model or self._initialize_backend()
            return self.model
        raise NotImplementedError("GNN training harness not implemented yet")

    def predict_proba(self, graphs: List[Any]) -> List[float]:
        if self.backend == "lgbm" and self.model is not None:
            X = graph_list_to_bow(graphs)
            return self._predict_lgbm(self.model, X)
        if self.backend == "heuristic":
            return self.model.predict_proba(graphs)
        if self.backend == "gnn":
            raise NotImplementedError("GNN predict not implemented yet")
        return self.model.predict_proba(graphs)

    def predict(self, graphs: List[Any], thresh: float = 0.5) -> List[int]:
        probs = self.predict_proba(graphs)
        return [1 if p >= thresh else 0 for p in probs]
