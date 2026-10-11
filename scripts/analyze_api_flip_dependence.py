"""Test target-API necessity and matched-position deletion controls."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_full_pipeline import build_graph
from src.classifier.harness import ClassifierHarness
from src.counterfactual.search import CounterfactualSearch

DEFAULT_APIS = ("NtTestAlert", "GetSystemTimeAsFileTime")


def _candidate_apis(candidate, graph):
    apis = set()
    edited_nodes = set(candidate.get("delete_nodes", []) or [])
    edited_nodes.update((candidate.get("substitute", {}) or {}).keys())
    for edge in candidate.get("delete_edges", []) or []:
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            edited_nodes.update(edge)
    for node in edited_nodes:
        if node in graph:
            api = graph.nodes[node].get("api")
            if api:
                apis.add(str(api).casefold())
    apis.update(
        str(item.get("api")).casefold()
        for item in candidate.get("insert_nodes", []) or []
        if item.get("api")
    )
    apis.update(
        str(api).casefold()
        for api in (candidate.get("substitute", {}) or {}).values()
        if api
    )
    return apis


def _node_positions(graph):
    process_lengths = {}
    for node, data in graph.nodes(data=True):
        if data.get("entity_type") == "resource":
            continue
        process_id = data.get("process_id")
        sequences = [
            sequence
            for sequence in (data.get("sequences") or [])
            if sequence is not None
        ]
        if sequences:
            process_lengths[process_id] = (
                process_lengths.get(process_id, 0) + len(sequences)
            )
    positions = {}
    for node, data in graph.nodes(data=True):
        if data.get("entity_type") == "resource":
            continue
        sequences = [
            sequence
            for sequence in (data.get("sequences") or [])
            if sequence is not None
        ]
        process_id = data.get("process_id")
        process_length = process_lengths.get(process_id, 0)
        if sequences and process_length:
            positions[node] = min(sequences) / process_length
    return positions


def _nearest_nodes(target_node, graph, positions, target_apis, limit=20):
    target_position = positions.get(target_node)
    if target_position is None:
        return []
    target_degree = graph.degree(target_node)
    neighbors = []
    for node, data in graph.nodes(data=True):
        if node == target_node or node not in positions:
            continue
        api = str(data.get("api") or "").casefold()
        if api in target_apis or api == "process":
            continue
        position_delta = abs(positions[node] - target_position)
        degree_delta = abs(graph.degree(node) - target_degree)
        neighbors.append(
            (
                (position_delta, degree_delta, str(node)),
                {
                    "node": node,
                    "api": data.get("api"),
                    "position_delta": position_delta,
                    "degree_delta": degree_delta,
                },
            )
        )
    neighbors.sort(key=lambda item: item[0])
    return [neighbor for _, neighbor in neighbors[:limit]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="pipeline_results.json")
    parser.add_argument("--model", default="models/pipeline_gnn.pkl")
    parser.add_argument("--out", default="api_flip_dependence.jsonl")
    parser.add_argument("--apis", default=",".join(DEFAULT_APIS))
    parser.add_argument("--boilerplate-json", default="api_prevalence_oct10.json")
    parser.add_argument("--control-samples", type=int, default=20)
    args = parser.parse_args()
    if args.control_samples < 1:
        parser.error("--control-samples must be at least 1")

    targets = {
        api.strip().casefold()
        for api in args.apis.split(",")
        if api.strip()
    }
    if not targets:
        parser.error("--apis must contain at least one API")

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    results_path = resolve(args.results)
    model_path = resolve(args.model)
    output_path = resolve(args.out)
    boilerplate_path = resolve(args.boilerplate_json)
    if output_path.exists():
        parser.error(f"Output already exists; choose a new --out: {output_path}")

    saved = json.loads(results_path.read_text(encoding="utf-8"))
    prevalence = json.loads(boilerplate_path.read_text(encoding="utf-8"))
    boilerplate = {
        str(api).casefold()
        for api in prevalence["boilerplate_candidates"]
    }
    successful_rows = [
        row for row in saved["results"] if row.get("status") == "completed"
    ]
    classifier = ClassifierHarness(backend="gnn", model_path=str(model_path))
    if classifier.model is None or classifier.api_vocab is None:
        parser.error(f"Could not load baseline GNN model: {model_path}")
    api_vocab = {str(api).casefold() for api in classifier.api_vocab}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        safe_flip_count = 0
        output.write(
            json.dumps(
                {
                    "type": "metadata",
                    "baseline_results": str(results_path),
                    "baseline_model": str(model_path),
                    "target_apis": sorted(targets),
                    "excluded_boilerplate_apis": sorted(boilerplate),
                    "substitution_vocab": str(model_path),
                    "successful_reports": len(successful_rows),
                    "nearest_control_count_per_target_node": args.control_samples,
                    "control_match": "same process and entity type; sorted by absolute relative-position delta, then absolute total-degree delta; no hard tolerance",
                }
            )
            + "\n"
        )
        output.flush()

        for report_index, row in enumerate(successful_rows, 1):
            explanation_path = resolve(row["counterfactual_explanation"])
            explanation = json.loads(
                explanation_path.read_text(encoding="utf-8")
            )
            saved_result = explanation["result"]
            candidates = saved_result.get("feasible_candidates")
            if not isinstance(candidates, list):
                raise ValueError(
                    f"Missing feasible_candidates in {explanation_path}"
                )
            input_path = resolve(explanation["input"])
            graph = build_graph(str(input_path))
            original_probability = float(
                classifier.predict_proba([graph])[0]
            )

            eligible_candidates = [
                candidate
                for candidate in candidates
                if not (_candidate_apis(candidate, graph) & boilerplate)
                and all(
                    str(api).casefold() in api_vocab
                    for api in (candidate.get("substitute", {}) or {}).values()
                )
            ]
            safe_scores = [
                float(classifier.predict_candidate_proba(graph, candidate))
                for candidate in eligible_candidates
            ]
            best_safe_probability = min(safe_scores, default=None)
            safe_flip = (
                best_safe_probability is not None
                and best_safe_probability
                < float(saved["run_config"]["threshold"])
            )
            safe_flip_count += int(safe_flip)

            winning_candidate = saved_result.get("candidate") or {}
            winning_deleted_targets = [
                node
                for node in (winning_candidate.get("delete_nodes", []) or [])
                if node in graph
                and str(graph.nodes[node].get("api") or "").casefold() in targets
            ]

            positions = _node_positions(graph)
            nearest_by_target = {
                target_node: _nearest_nodes(
                    target_node,
                    graph,
                    positions,
                    targets,
                    args.control_samples,
                )
                for target_node in winning_deleted_targets
            }
            control_records = []
            winning_deletions = set(
                winning_candidate.get("delete_nodes", []) or []
            )
            control_search = CounterfactualSearch(graph)
            seen_controls = set()
            for target_node, neighbors in nearest_by_target.items():
                for replacement in neighbors:
                    if replacement["node"] in winning_deletions:
                        continue
                    deleted = (winning_deletions - {target_node}) | {
                        replacement["node"]
                    }
                    control_candidate = {
                        "delete_nodes": sorted(deleted, key=str),
                        "delete_edges": winning_candidate.get(
                            "delete_edges", []
                        ),
                        "substitute": dict(
                            winning_candidate.get("substitute", {}) or {}
                        ),
                        "insert_nodes": winning_candidate.get(
                            "insert_nodes", []
                        ),
                    }
                    key = control_search._candidate_key(control_candidate)
                    if key in seen_controls:
                        continue
                    seen_controls.add(key)
                    if not control_search.validate(control_candidate):
                        continue
                    probability = float(
                        classifier.predict_candidate_proba(
                            graph, control_candidate
                        )
                    )
                    control_records.append(
                        {
                            "candidate": control_candidate,
                            "probability": probability,
                            "flipped": probability
                            < float(saved["run_config"]["threshold"]),
                            "match": {
                                "target_node": target_node,
                                "target_api": graph.nodes[target_node].get("api"),
                                "control_node": replacement["node"],
                                "control_api": replacement.get("api"),
                                "position_delta": replacement["position_delta"],
                                "degree_delta": replacement["degree_delta"],
                            },
                        }
                    )

            record = {
                "type": "report",
                "index": report_index,
                "file": row["file"],
                "original_probability": original_probability,
                "saved_winning_probability": row.get("new_prob"),
                "winning_candidate": winning_candidate,
                "winning_deleted_target_apis": [
                    graph.nodes[node].get("api")
                    for node in winning_deleted_targets
                ],
                "feasible_candidates": len(candidates),
                "candidates_after_boilerplate_and_vocab_filters": len(
                    eligible_candidates
                ),
                "best_probability_after_filters": best_safe_probability,
                "boilerplate_excluded_flip": safe_flip,
                "nearest_control_neighbors_by_target": nearest_by_target,
                "position_control_candidates_available": sum(
                    len(neighbors) for neighbors in nearest_by_target.values()
                ),
                "position_control_candidates_scored": len(control_records),
                "position_control_flip_count": sum(
                    item["flipped"] for item in control_records
                ),
                "position_control_best_probability": min(
                    (
                        item["probability"]
                        for item in control_records
                    ),
                    default=None,
                ),
                "position_control_records": control_records,
            }
            output.write(json.dumps(record, default=str) + "\n")
            output.flush()
            print(
                f"[{report_index}/{len(successful_rows)}] {row['file']}: "
                f"original={original_probability:.6f} "
                f"eligible_candidates={len(eligible_candidates)}/{len(candidates)} "
                f"best_after_filters={best_safe_probability} "
                f"controls_available={record['position_control_candidates_available']} "
                f"controls_scored={len(control_records)} "
                f"controls_flipped={record['position_control_flip_count']}",
                flush=True,
            )

        output.write(
            json.dumps(
                {
                    "type": "summary",
                    "reports": len(successful_rows),
                    "reports_flipping_after_boilerplate_and_vocab_filters": safe_flip_count,
                }
            )
            + "\n"
        )
        output.flush()
        print(
            "Restricted candidate summary: "
            f"{safe_flip_count}/{len(successful_rows)} reports retain "
            "at least one flip after boilerplate and OOV-substitution filters.",
            flush=True,
        )


if __name__ == "__main__":
    main()
