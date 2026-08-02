"""A lightweight heuristic classifier for behavior-graph explanations.

This provides a deterministic scoring function for smoke tests and fast demos.
It is intentionally simple and should not be treated as a production evaluator.
"""
from typing import Any, List


class HeuristicClassifier:
    """Score a behavior graph by counting suspicious operations."""

    SUSPICIOUS_APIS = {
        "regsetvalue": 0.35,
        "createscheduledtask": 0.30,
        "createservice": 0.30,
        "createremotethread": 0.25,
        "apcinject": 0.25,
        "processhollow": 0.25,
        "writefile": 0.10,
        "createfile": 0.10,
    }

    def predict_proba(self, graphs: List[Any]) -> List[float]:
        return [self._score_graph(graph) for graph in graphs]

    def _score_graph(self, graph: Any) -> float:
        total = 0.0
        for _, data in graph.nodes(data=True):
            api = str(data.get("api", "unknown") or "unknown").lower()
            for token, weight in self.SUSPICIOUS_APIS.items():
                if token in api:
                    total += weight
                    break
        if total <= 0.0:
            return 0.15
        node_scale = max(1.0, graph.number_of_nodes() * 0.6)
        normalized = min(0.99, 0.15 + total / node_scale)
        return round(normalized, 4)
