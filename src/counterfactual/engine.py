"""Counterfactual search engine wrapper."""
import json
import logging
from pathlib import Path
from typing import Optional, Any, Dict, List

import networkx as nx

from src.classifier.harness import ClassifierHarness
from src.counterfactual.capa import infer_capa_techniques
from src.counterfactual.metrics import compute_metrics, detect_decoy_flips
from src.counterfactual.search import CounterfactualSearch
from src.counterfactual.tier2 import build_synthetic_cape_report
from src.behavior.resource_lifetimes import match_resource_lifetimes
from src.graph.graph_builder import build_behavior_graph
from src.ingestion.parser import parse_cape_json, parse_cape_report

logger = logging.getLogger(__name__)


class CounterfactualEngine:
    def __init__(
        self,
        classifier_backend: Optional[str] = None,
        classifier_model: Optional[Any] = None,
        excluded_apis: Optional[List[str]] = None,
    ):
        self.classifier_backend = classifier_backend
        self.excluded_apis = excluded_apis or []
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
        search = CounterfactualSearch(
            graph=graph,
            excluded_apis=self.excluded_apis,
        )
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
            candidate = result.get("candidate", {}) or {}
            edit_details = []
            for node in candidate.get("delete_nodes", []) or []:
                if node not in graph:
                    continue
                data = graph.nodes[node]
                edit_details.append({
                    "action": "delete_node",
                    "node": node,
                    "api": data.get("api"),
                    "entity_type": data.get("entity_type"),
                    "process_id": data.get("process_id"),
                    "attack_id": data.get("attack_id"),
                    "resources": data.get("resources", []),
                })
            for edge in candidate.get("delete_edges", []) or []:
                if len(edge) != 2:
                    continue
                source, target = edge
                edit_details.append({
                    "action": "delete_edge",
                    "source": source,
                    "target": target,
                    "edge_type": (graph.get_edge_data(source, target) or {}).get("type"),
                })
            for node, replacement in (candidate.get("substitute", {}) or {}).items():
                edit_details.append({
                    "action": "substitute_api",
                    "node": node,
                    "from": graph.nodes[node].get("api") if node in graph else None,
                    "to": replacement,
                })
            for insertion in candidate.get("insert_nodes", []) or []:
                edit_details.append({
                    "action": "insert_node",
                    **insertion,
                })

            output["result"]["classifier_evaluation"] = {
                "status": "flipped",
                "original_probability": result.get("orig_prob"),
                "edited_probability": result.get("new_prob"),
            }
            edit_labels = [
                detail.get("api") or detail.get("from") or detail.get("action")
                for detail in edit_details
            ]
            output["result"]["explanation"] = {
                "text": (
                    f"Applying {len(edit_details)} edit(s)"
                    f"{': ' + ', '.join(map(str, edit_labels)) if edit_labels else ''} "
                    f"changed the classifier's malware probability from "
                    f"{result.get('orig_prob'):.4f} to {result.get('new_prob'):.4f}, "
                    f"crossing the {search.threshold:.2f} decision threshold."
                ),
                "threshold": search.threshold,
                "edits": edit_details,
            }
            output["result"]["feasibility"] = {
                "tier": "tier1_structural",
                "execution_confirmed": False,
            }
            if isinstance(edited, nx.DiGraph):
                synthetic_report = build_synthetic_cape_report(edited)
                round_trip_events = parse_cape_report(synthetic_report)
                round_trip_lifetimes = match_resource_lifetimes(round_trip_events)
                round_trip_graph = build_behavior_graph(
                    round_trip_events,
                    lifetimes=round_trip_lifetimes["lifetimes"],
                    active_resources=round_trip_lifetimes["still_active"],
                )
                output["result"]["tier2_round_trip"] = {
                    "status": "passed",
                    "parsed_events": len(round_trip_events),
                    "graph_nodes": round_trip_graph.number_of_nodes(),
                    "graph_edges": round_trip_graph.number_of_edges(),
                }
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
