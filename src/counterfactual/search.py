"""Counterfactual search engine skeleton."""
import logging
from typing import Any, Dict, List, Optional

import networkx as nx

from src.counterfactual import feasibility, substitutions

try:
    from src.utils.pyg_adapter import build_api_vocab, graph_to_pyg_data
    import torch
except Exception:  # pragma: no cover - optional dependency
    build_api_vocab = None
    graph_to_pyg_data = None
    torch = None

logger = logging.getLogger(__name__)


class CounterfactualSearch:
    def __init__(self, classifier: Optional[Any], graph: nx.DiGraph):
        self.classifier = classifier
        self.graph = graph
        self.max_edits = 10
        self.max_candidates = 200
        self.threshold = 0.5
        self.api_vocab = None

    def _apply_candidate(self, candidate: Dict) -> nx.DiGraph:
        return feasibility.apply_candidate(self.graph, candidate)

    def _candidate_cost(self, candidate: Dict) -> int:
        return feasibility.candidate_cost(candidate)

    def _within_edit_budget(self, candidate: Dict) -> bool:
        return self._candidate_cost(candidate) <= self.max_edits

    def _downstream_cascade(self, node: str) -> List[str]:
        """Return the recursive resource/temporal dependency closure for a node."""
        seen = set()
        ordered = []
        stack = [node]
        while stack:
            current = stack.pop()
            for succ in self.graph.successors(current):
                edge_data = self.graph.get_edge_data(current, succ) or {}
                if edge_data.get("type") not in {"resource", "temporal"}:
                    continue
                if succ in seen:
                    continue
                seen.add(succ)
                ordered.append(succ)
                stack.append(succ)
        return ordered

    def propose(self) -> List[Dict]:
        """Generate candidate edits with a small edit budget and simple cascades."""
        try:
            import torch as torch_module
        except Exception:
            torch_module = None

        if torch_module is not None and hasattr(self.classifier, "parameters"):
            try:
                from src.counterfactual.gradient_proposer import propose_from_gradients
                from src.counterfactual.edge_mask_proposer import propose_edge_deletions

                if self.api_vocab is None and build_api_vocab is not None:
                    self.api_vocab = build_api_vocab([self.graph])

                cands = propose_from_gradients(self.classifier, self.graph, top_k=20, api_vocab=self.api_vocab)
                edge_cands = propose_edge_deletions(self.classifier, self.graph, top_k=10)
                cands.extend(edge_cands)
                extended = []
                for candidate in cands:
                    if self._within_edit_budget(candidate):
                        extended.append(candidate)
                    for n in candidate.get("delete_nodes", []):
                        api = self.graph.nodes[n].get("api", "")
                        for sub in substitutions.get_substitutes(api):
                            sub_candidate = {"delete_nodes": [], "substitute": {n: sub}}
                            if self._within_edit_budget(sub_candidate):
                                extended.append(sub_candidate)
                return extended[: self.max_candidates]
            except Exception as exc:
                logger.warning("Gradient proposer failed, falling back to enumerative proposals: %s", exc)

        cands: List[Dict] = []
        nodes = list(self.graph.nodes())
        for n in nodes:
            if len(cands) >= self.max_candidates:
                break
            delete_candidate = {"delete_nodes": [n], "substitute": {}}
            if self._within_edit_budget(delete_candidate):
                cands.append(delete_candidate)
            api = self.graph.nodes[n].get("api", "")
            for sub in substitutions.get_substitutes(api):
                if len(cands) >= self.max_candidates:
                    break
                sub_candidate = {"delete_nodes": [], "substitute": {n: sub}}
                if self._within_edit_budget(sub_candidate):
                    cands.append(sub_candidate)
            for downstream in self._downstream_cascade(n):
                if len(cands) >= self.max_candidates:
                    break
                cascade_candidate = {"delete_nodes": [n, downstream], "substitute": {}}
                if self._within_edit_budget(cascade_candidate):
                    cands.append(cascade_candidate)
        return cands

    def validate(self, candidate: Dict) -> bool:
        return feasibility.validate_candidate(self.graph, candidate)

    def run(self) -> Optional[Dict]:
        """Run propose->validate->query loop and return a structured outcome."""
        orig_prob = None
        unscoreable = False
        if self.classifier is not None:
            try:
                if hasattr(self.classifier, "predict_proba"):
                    orig_prob = float(self.classifier.predict_proba([self.graph])[0])
            except Exception as exc:
                logger.warning("Unable to score original graph: %s", exc)
                orig_prob = None
                unscoreable = True

        if self.classifier is not None and orig_prob is None and unscoreable:
            return {"status": "unscoreable", "candidate": None, "edited_graph": None, "orig_prob": None, "new_prob": None}

        if self.classifier is not None and orig_prob is not None and orig_prob < self.threshold:
            return {"status": "not_malicious", "candidate": None, "edited_graph": None, "orig_prob": orig_prob, "new_prob": None}

        if self.classifier is None:
            best_result = None
            for cand in self.propose():
                if not self.validate(cand):
                    continue
                edited = self._apply_candidate(cand)
                result = {"status": "completed", "candidate": cand, "edited_graph": edited}
                if best_result is None or self._candidate_cost(cand) < self._candidate_cost(best_result["candidate"]):
                    best_result = result
            return best_result or {"status": "no_flip_found", "candidate": None, "edited_graph": None}

        best_flip_result = None
        best_flip_cost = None
        for cand in self.propose():
            if not self.validate(cand):
                continue
            edited = self._apply_candidate(cand)

            try:
                if hasattr(self.classifier, "predict_proba"):
                    prob = float(self.classifier.predict_proba([edited])[0])
                else:
                    if self.api_vocab is None:
                        self.api_vocab = build_api_vocab([self.graph, edited])
                    data = graph_to_pyg_data(edited, self.api_vocab)
                    model = self.classifier
                    model.eval()
                    with torch.no_grad():
                        out = model(data.x, data.edge_index)
                        prob = float(torch.sigmoid(out).item())
            except Exception as exc:
                logger.warning("Unable to score edited graph: %s", exc)
                prob = None
                unscoreable = True

            if orig_prob is not None and prob is not None and orig_prob >= self.threshold and prob < self.threshold:
                flip_result = {"status": "completed", "candidate": cand, "edited_graph": edited, "orig_prob": orig_prob, "new_prob": prob}
                flip_cost = self._candidate_cost(cand)
                if best_flip_result is None or flip_cost < best_flip_cost:
                    best_flip_result = flip_result
                    best_flip_cost = flip_cost

        if best_flip_result is not None:
            return best_flip_result

        if unscoreable:
            return {"status": "unscoreable", "candidate": None, "edited_graph": None, "orig_prob": orig_prob, "new_prob": None}

        return {"status": "no_flip_found", "candidate": None, "edited_graph": None, "orig_prob": orig_prob, "new_prob": None}
