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
from src.behavior.resource_lifetimes import match_resource_lifetimes
from src.graph.graph_builder import build_behavior_graph
from src.ingestion.parser import parse_cape_json

logger = logging.getLogger(__name__)


class CounterfactualEngine:
    def __init__(self, classifier_backend: Optional[str] = None, classifier_model: Optional[Any] = None):
        self.classifier_backend = classifier_backend
        if classifier_model is not None:
            self.classifier = classifier_model
        elif classifier_backend:
            self.classifier = ClassifierHarness(backend=classifier_backend)
        else:
            self.classifier = None

    def explain_from_cape(self, cape_json_path: str) -> Dict:
        events = parse_cape_json(cape_json_path)
        lifetime_result = match_resource_lifetimes(events)
        graph = build_behavior_graph(
            events,
            lifetimes=lifetime_result["lifetimes"],
            active_resources=lifetime_result["still_active"],
        )
        search = CounterfactualSearch(graph=graph)
        result = search.find_flip(self.classifier) if self.classifier is not None else search.generate_candidates()

        output = {
            "input": cape_json_path,
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "resource_lifetimes": {
                "matched": len(lifetime_result["lifetimes"]),
                "orphan_releases": len(lifetime_result["orphan_releases"]),
                "still_active": len(lifetime_result["still_active"]),
            },
            "result": None,
        }
        if result is None or result.get("status") != "completed":
            result = result or {"status": "no_feasible_candidate"}
            trace = result.pop("candidate_trace", [])
            feasible = result.pop("feasible_candidates", [])
            result["search_summary"] = {
                **result.get("search", {}),
                "feasible_candidates": len(feasible),
                "valid_candidates_evaluated": sum(1 for item in trace if item.get("valid")),
                "invalid_candidates_evaluated": sum(1 for item in trace if not item.get("valid")),
            }
            output["result"] = result
            output["result"]["classifier_evaluation"] = (
                {"status": "not_run"} if self.classifier is None else {"status": result.get("status")}
            )
            output["result"]["feasibility"] = {
                "tier": "tier1_structural",
                "execution_confirmed": False,
            }
        else:
            edited = result.get("edited_graph")
            output["result"] = {k: v for k, v in result.items() if k != "edited_graph"}
            output["result"]["classifier_evaluation"] = {
                "status": "flipped",
                "original_probability": result.get("orig_prob"),
                "edited_probability": result.get("new_prob"),
            }
            output["result"]["feasibility"] = {
                "tier": "tier1_structural",
                "execution_confirmed": False,
            }
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
                    {
                        "node": node,
                        "from": graph.nodes[node].get("api"),
                        "to": replacement,
                    }
                    for node, replacement in candidate.get("substitute", {}).items()
                    if node in graph
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
