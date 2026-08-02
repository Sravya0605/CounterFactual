"""Counterfactual search engine wrapper."""
import json
import logging
from pathlib import Path
from typing import Optional, Any, Dict

import networkx as nx

from src.classifier.harness import ClassifierHarness
from src.counterfactual.capa import infer_capa_techniques
from src.counterfactual.metrics import compute_metrics, detect_decoy_flips
from src.counterfactual.search import CounterfactualSearch
from src.graph.graph_builder import build_behavior_graph
from src.ingestion.parser import parse_cape_json

logger = logging.getLogger(__name__)


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
        if result is None or result.get("status") != "completed":
            output["result"] = result or {"status": "no_flip_found", "warning": "classifier unavailable or no flip found"}
        else:
            edited = result.get("edited_graph")
            output["result"] = {k: v for k, v in result.items() if k != "edited_graph"}
            if isinstance(edited, nx.DiGraph):
                output["result"]["edited_nodes"] = edited.number_of_nodes()
                output["result"]["edited_edges"] = edited.number_of_edges()
                output["result"]["attack_ids"] = sorted({
                    str(data.get("attack_id")) for _, data in edited.nodes(data=True) if data.get("attack_id")
                })
                metrics = compute_metrics(graph, edited, result.get("candidate", {}))
                output["result"]["metrics"] = metrics
                output["result"]["decoy_flips"] = detect_decoy_flips(
                    graph,
                    edited,
                    result.get("candidate"),
                    threshold=0.5,
                    orig_prob=result.get("orig_prob"),
                    new_prob=result.get("new_prob"),
                )
                output["result"]["capabilities"] = infer_capa_techniques(edited)
                candidate = result.get("candidate", {}) or {}
                output["result"]["substitution_provenance"] = [
                    {"node": node, "from": data.get("api"), "to": candidate.get("substitute", {}).get(node)}
                    for node, data in edited.nodes(data=True)
                    if candidate.get("substitute", {}).get(node)
                ]
                output["result"]["baselines"] = {
                    "shap": {"status": "not_run"},
                    "lime": {"status": "not_run"},
                    "gnn_explainer": {"status": "not_run"},
                    "cf_gnn_explainer": {"status": "not_run"},
                }
        return output

    def explain_and_write(self, cape_json_path: str, out_path: str) -> Dict:
        result = self.explain_from_cape(cape_json_path)
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        return result
