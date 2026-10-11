"""Compare APIs in saved feasible candidates with APIs used in successful flips."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_full_pipeline import build_graph


def _candidate_apis(candidate, graph):
    apis = set()
    for node in candidate.get("delete_nodes", []) or []:
        if node in graph:
            api = graph.nodes[node].get("api")
            if api:
                apis.add(str(api))
    for node in (candidate.get("substitute", {}) or {}):
        if node in graph:
            api = graph.nodes[node].get("api")
            if api:
                apis.add(str(api))
    for source, target in candidate.get("delete_edges", []) or []:
        for node in (source, target):
            if node in graph:
                api = graph.nodes[node].get("api")
                if api:
                    apis.add(str(api))
    for insertion in candidate.get("insert_nodes", []) or []:
        api = insertion.get("api")
        if api:
            apis.add(str(api))
    return apis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="pipeline_results.json")
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    results_path = Path(args.results)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    rows = [row for row in results["results"] if row.get("status") == "completed"]

    flip_reports = Counter()
    feasible_candidate_reports = Counter()
    feasible_candidate_counts = Counter()
    total_candidates = 0
    reports_with_candidates = 0

    for row in rows:
        explanation_path = Path(row["counterfactual_explanation"])
        if not explanation_path.is_absolute():
            explanation_path = ROOT / explanation_path
        explanation = json.loads(explanation_path.read_text(encoding="utf-8"))
        saved_result = explanation["result"]
        candidates = saved_result.get("feasible_candidates")
        if not isinstance(candidates, list):
            raise ValueError(
                f"Missing feasible_candidates list in {explanation_path}"
            )

        graph_path = Path(explanation["input"])
        if not graph_path.is_absolute():
            graph_path = ROOT / graph_path
        graph = build_graph(str(graph_path))
        reports_with_candidates += 1
        total_candidates += len(candidates)

        apis_in_report = set()
        for candidate in candidates:
            candidate_apis = _candidate_apis(candidate, graph)
            apis_in_report.update(candidate_apis)
            feasible_candidate_counts.update(candidate_apis)
        feasible_candidate_reports.update(apis_in_report)

        selected = _candidate_apis(saved_result.get("candidate") or {}, graph)
        flip_reports.update(selected)

    print(
        f"Successful reports analyzed: {reports_with_candidates}/{len(rows)}"
    )
    print(f"Feasible candidates retained: {total_candidates}")
    print(
        "Scope: successful-report candidate lists only; no-flip reports do not "
        "retain their candidates."
    )
    print(
        f"{'API':40} {'flip reports':>12} {'candidate reports':>18} "
        f"{'candidate count':>16} {'flip share':>11} {'candidate share':>16}"
    )

    apis = set(flip_reports) | set(feasible_candidate_counts)
    ranked = sorted(
        apis,
        key=lambda api: (
            feasible_candidate_counts[api],
            flip_reports[api],
            api.casefold(),
        ),
        reverse=True,
    )
    for api in ranked[:args.top]:
        print(
            f"{api:40} {flip_reports[api]:12d} "
            f"{feasible_candidate_reports[api]:18d} "
            f"{feasible_candidate_counts[api]:16d} "
            f"{flip_reports[api] / reports_with_candidates:10.1%} "
            f"{feasible_candidate_counts[api] / total_candidates:15.1%}"
        )

    for target in ("NtTestAlert", "GetSystemTimeAsFileTime"):
        matches = [api for api in apis if api.casefold() == target.casefold()]
        if not matches:
            print(f"TARGET NOT FOUND: {target}")
            continue
        api = matches[0]
        print(
            f"TARGET {api}: selected in {flip_reports[api]} flips; "
            f"appeared in {feasible_candidate_counts[api]} feasible candidates "
            f"across {feasible_candidate_reports[api]} reports."
        )


if __name__ == "__main__":
    main()
