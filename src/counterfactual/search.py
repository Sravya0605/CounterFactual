"""Counterfactual search engine skeleton.

This file contains a minimal skeleton for the counterfactual search engine described
in the design doc. The full implementation (CF-GNNExplainer adaptation, validity
checker, substitution library) will be added iteratively.
"""
from typing import Any, Dict, List, Optional
import networkx as nx
from src.counterfactual import feasibility, substitutions
from src.utils.pyg_adapter import build_api_vocab, graph_to_pyg_data
from src.classifier.gnn_model import SimpleGCN
import torch


class CounterfactualSearch:
    def __init__(self, classifier: Optional[Any], graph: nx.DiGraph):
        self.classifier = classifier
        self.graph = graph
        # search parameters
        self.max_edits = 10
        self.max_candidates = 200

    def _apply_candidate(self, candidate: Dict) -> nx.DiGraph:
        return feasibility.apply_candidate(self.graph, candidate)

    def propose(self) -> List[Dict]:
        """Generate candidate edits. Prefer gradient-guided proposals for GNNs.

        Falls back to enumerative single-node deletions/substitutions when no
        gradient proposer is available or when the classifier isn't a PyTorch
        model.
        """
        # If we have a PyTorch GNN model, use gradient proposer
        try:
            import torch
        except Exception:
            torch = None

        if torch is not None and hasattr(self.classifier, "parameters"):
            try:
                from src.counterfactual.gradient_proposer import propose_from_gradients
                from src.counterfactual.edge_mask_proposer import propose_edge_deletions
                cands = propose_from_gradients(self.classifier, self.graph, top_k=20)
                # edge deletion candidates
                edge_cands = propose_edge_deletions(self.classifier, self.graph, top_k=10)
                cands.extend(edge_cands)
                # attach substitutions for each candidate's nodes as well
                extended = []
                for c in cands:
                    extended.append(c)
                    for n in c.get("delete_nodes", []):
                        api = self.graph.nodes[n].get("api", "")
                        for sub in substitutions.get_substitutes(api):
                            extended.append({"delete_nodes": [], "substitute": {n: sub}})
                return extended[: self.max_candidates]
            except Exception:
                # fall through to enumerative proposer on failure
                pass

        # enumerative fallback
        cands = []
        nodes = list(self.graph.nodes())
        for i, n in enumerate(nodes):
            if len(cands) >= self.max_candidates:
                break
            cands.append({"delete_nodes": [n], "substitute": {}})
            api = self.graph.nodes[n].get("api", "")
            for sub in substitutions.get_substitutes(api):
                cands.append({"delete_nodes": [], "substitute": {n: sub}})
        return cands

    def validate(self, candidate: Dict) -> bool:
        return feasibility.validate_candidate(self.graph, candidate)

    def run(self) -> Optional[Dict]:
        """Run propose->validate->query loop and return the first flipping edit.

        If `self.classifier` is None the method returns the first valid candidate
        (useful for structural testing). If a classifier is provided, it must
        implement `predict_proba(graph_list)` and accept a list of graphs.
        """
        # original prediction (if classifier provided)
        orig_prob = None
        if self.classifier is not None:
            try:
                orig_prob = float(self.classifier.predict_proba([self.graph])[0])
            except Exception:
                orig_prob = None

        for cand in self.propose():
            if not self.validate(cand):
                continue
            edited = self._apply_candidate(cand)
            if self.classifier is None:
                return {"candidate": cand, "edited_graph": edited}
            try:
                # handle GNN classifier objects (SimpleGCN) or harnesses
                if hasattr(self.classifier, "predict_proba"):
                    prob = float(self.classifier.predict_proba([edited])[0])
                else:
                    # assume a PyG model: convert graph and run forward
                    vocab = build_api_vocab([self.graph, edited])
                    data = graph_to_pyg_data(edited, vocab)
                    model = self.classifier
                    model.eval()
                    with torch.no_grad():
                        out = model(data.x, data.edge_index)
                        prob = float(torch.sigmoid(out).item())
            except Exception:
                prob = None
            # consider flip from malicious->benign as prob drop below 0.5
            if orig_prob is not None and prob is not None:
                if orig_prob >= 0.5 and prob < 0.5:
                    return {"candidate": cand, "edited_graph": edited, "orig_prob": orig_prob, "new_prob": prob}

        return None
