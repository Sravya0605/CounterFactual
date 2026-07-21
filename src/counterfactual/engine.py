"""Counterfactual search engine wrapper.

Provides a simple `CounterfactualEngine` that ties together ingestion, graph
building, classifier harness (optional), search, and result export.
"""
from typing import Optional, Any, Dict
import json
import networkx as nx

from src.classifier.harness import ClassifierHarness
from src.counterfactual.search import CounterfactualSearch
from src.graph.graph_builder import build_behavior_graph
from src.ingestion.parser import parse_cape_json


class CounterfactualEngine:
    def __init__(self, classifier_backend: Optional[str] = None, classifier_model: Optional[Any] = None):
        if classifier_model is not None:
            self.classifier = classifier_model
        else:
            backend = (classifier_backend or "heuristic").lower()
            if backend in {"heuristic", "lgbm", "gnn", "sklearn"}:
                self.classifier = ClassifierHarness(backend=backend)
            else:
                self.classifier = None

    def explain_from_cape(self, cape_json_path: str) -> Dict:
        events = parse_cape_json(cape_json_path)
        graph = build_behavior_graph(events)
        search = CounterfactualSearch(classifier=self.classifier, graph=graph)
        result = search.run()

        output = {
            "input": cape_json_path,
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "result": None,
        }
        if result is None:
            output["result"] = {"status": "no_flip_found"}
        else:
            edited = result.get("edited_graph")
            output["result"] = {k: v for k, v in result.items() if k != "edited_graph"}
            if isinstance(edited, nx.DiGraph):
                output["result"]["edited_nodes"] = edited.number_of_nodes()
                output["result"]["edited_edges"] = edited.number_of_edges()
        return output

    def explain_and_write(self, cape_json_path: str, out_path: str) -> Dict:
        result = self.explain_from_cape(cape_json_path)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        return result
