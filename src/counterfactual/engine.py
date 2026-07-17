"""Counterfactual search engine wrapper.

Provides a simple `CounterfactualEngine` that ties together ingestion, graph
building, classifier harness (optional), search, and result export.
This is a skeleton useful for wiring the pieces during prototyping.
"""
from typing import Optional, Any, Dict
import json
import networkx as nx

from src.counterfactual.search import CounterfactualSearch
from src.classifier.harness import ClassifierHarness
from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph


class CounterfactualEngine:
    def __init__(self, classifier_backend: Optional[str] = None, classifier_model: Optional[Any] = None):
        # If classifier_model is provided, use it directly; otherwise, a harness
        # may be created later when training is implemented.
        if classifier_model is not None:
            self.classifier = classifier_model
        elif classifier_backend is not None:
            self.classifier = ClassifierHarness(backend=classifier_backend)
        else:
            self.classifier = None

    def explain_from_cape(self, cape_json_path: str) -> Dict:
        events = parse_cape_json(cape_json_path)
        G = build_behavior_graph(events)
        search = CounterfactualSearch(classifier=self.classifier, graph=G)
        result = search.run()
        # serialize basic metadata
        out = {"input": cape_json_path, "graph_nodes": G.number_of_nodes(), "graph_edges": G.number_of_edges(), "result": None}
        if result is None:
            out["result"] = {"status": "no_flip_found"}
        else:
            # simplify edited graph information to counts
            edited = result.get("edited_graph")
            out["result"] = {k: v for k, v in result.items() if k != "edited_graph"}
            if isinstance(edited, nx.DiGraph):
                out["result"]["edited_nodes"] = edited.number_of_nodes()
                out["result"]["edited_edges"] = edited.number_of_edges()
        return out

    def explain_and_write(self, cape_json_path: str, out_path: str) -> None:
        res = self.explain_from_cape(cape_json_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
