"""Classifier harness with a lightweight fallback for the prototype."""
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.graph_features import build_feature_vocab, graph_list_to_bow, graph_to_numpy_row


class ClassifierHarness:
    def __init__(self, backend: str = "heuristic", model_path: Optional[str] = None):
        self.backend = (backend or "heuristic").lower()
        self.model = None
        self.feature_vocab = None
        self.api_vocab = None
        self.model_path = Path(model_path) if model_path else self._default_model_path()
        # Cached token→column-index map for fast single-graph numpy scoring.
        self._token_to_idx: Optional[Dict[str, int]] = None
        self._initialize_backend()
        self._load_model_if_available()

    def _default_model_path(self) -> Path:
        root = Path(__file__).resolve().parents[2]
        return root / "models" / f"{self.backend}.pkl"

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
            from src.classifier.gnn_harness import train_gnn, predict_gnn_proba
            self._train_gnn = train_gnn
            self._predict_gnn = predict_gnn_proba
            self.model = None
            return


    def _load_model_if_available(self):
        if not self.model_path or not self.model_path.exists():
            return
        with open(self.model_path, "rb") as handle:
            state = pickle.load(handle)
        self.model = state.get("model")
        self.feature_vocab = state.get("feature_vocab")
        self.api_vocab = state.get("api_vocab")
        # Rebuild the index whenever a vocab is loaded.
        self._token_to_idx = None
        self._ensure_token_index()

    def _ensure_token_index(self):
        """Build (once) the token→column-index dict used by the fast numpy path."""
        if self._token_to_idx is None and self.feature_vocab is not None:
            self._token_to_idx = {tok: i for i, tok in enumerate(self.feature_vocab)}

    def save_model(self, path: Optional[str] = None):
        target = Path(path) if path else self.model_path
        target.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "backend": self.backend,
            "model": self.model,
            "feature_vocab": self.feature_vocab,
            "api_vocab": self.api_vocab,
        }
        with open(target, "wb") as handle:
            pickle.dump(state, handle)
        return target

    def load_model(self, path: Optional[str] = None):
        target = Path(path) if path else self.model_path
        if not target.exists():
            return None
        with open(target, "rb") as handle:
            state = pickle.load(handle)
        self.model = state.get("model")
        self.feature_vocab = state.get("feature_vocab")
        self.api_vocab = state.get("api_vocab")
        self._token_to_idx = None
        self._ensure_token_index()
        return self.model

    def ensure_trained(self, graphs: List[Any], labels: Optional[List[int]] = None):
        if self.backend == "heuristic":
            if self.model is None:
                self._initialize_backend()
            return self.model
        if self.model is not None:
            return self.model
        if self.model_path.exists():
            self.load_model(self.model_path)
            return self.model
        labels = labels or [1] * len(graphs)
        return self.train(graphs, labels)

    def train(self, graphs: List[Any], labels: List[int], **kwargs):
        if self.backend == "lgbm":
            self.feature_vocab = build_feature_vocab(graphs)
            X = graph_list_to_bow(graphs, vocab=self.feature_vocab)
            self.model = self._train_lgbm(
                X,
                labels,
                params=kwargs.get("params"),
                num_boost_round=kwargs.get("rounds", 100),
            )
            self._token_to_idx = None
            self._ensure_token_index()
            self.save_model(self.model_path)
            return self.model
        if self.backend == "heuristic":
            self.model = self.model or self._initialize_backend()
            return self.model
        if self.backend == "gnn":
            from src.utils.pyg_adapter import build_api_vocab
            self.api_vocab = build_api_vocab(graphs)
            self.model = self._train_gnn(graphs, labels, epochs=kwargs.get("epochs", 10), batch_size=kwargs.get("batch_size", 16))
            self.save_model(self.model_path)
            return self.model
        raise NotImplementedError("GNN training harness not implemented yet")

    def predict_proba(self, graphs: List[Any]) -> List[float]:
        if self.backend == "lgbm":
            if self.model is None:
                self.ensure_trained(graphs)
            if self.feature_vocab is None:
                self.feature_vocab = build_feature_vocab(graphs)
                self._token_to_idx = None
                self._ensure_token_index()
            # Fast path: single-graph scoring avoids pandas DataFrame overhead.
            if len(graphs) == 1:
                self._ensure_token_index()
                if self._token_to_idx is not None:
                    X = graph_to_numpy_row(graphs[0], self._token_to_idx)
                    return self._predict_lgbm(self.model, X)
            # Batch path: keep pandas for multi-graph calls (train / eval sets).
            X = graph_list_to_bow(graphs, vocab=self.feature_vocab)
            return self._predict_lgbm(self.model, X)
        if self.backend == "heuristic":
            return self.model.predict_proba(graphs)
        if self.backend == "gnn":
            if self.model is None:
                self.ensure_trained(graphs)
            return self._predict_gnn(self.model, graphs, self.api_vocab)
        return self.model.predict_proba(graphs)

    def predict(self, graphs: List[Any], thresh: float = 0.5) -> List[int]:
        probs = self.predict_proba(graphs)
        return [1 if p >= thresh else 0 for p in probs]
