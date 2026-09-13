import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from src.counterfactual.search import CounterfactualSearch
from src.counterfactual.feasibility import validate_candidate, candidate_cost
from src.graph.graph_builder import build_behavior_graph
from src.ingestion.parser import parse_cape_json


TRAINING_BATCH_PATH = Path("data/training_batch.csv")
REPORTS_DIR = Path("data/training_reports")
OUT_PATH = Path("data/evaluation_results.json")


def _bool_to_float(value: bool) -> float:
    return 1.0 if value else 0.0


def evaluate_dataset() -> Dict[str, Any]:
    if not TRAINING_BATCH_PATH.exists():
        raise FileNotFoundError(f"Missing training batch: {TRAINING_BATCH_PATH}")

    with TRAINING_BATCH_PATH.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    per_sample: List[Dict[str, Any]] = []
    accepted = 0
    minimality_total = 0.0
    decoy_flip_count = 0
    unconstrained_impossible = 0
    total_runtime = 0.0

    for idx, row in enumerate(rows, 1):
        md5 = row["md5"]
        report_path = REPORTS_DIR / f"{md5}.json"
        if not report_path.exists():
            continue

        start = time.perf_counter()
        events = parse_cape_json(str(report_path))
        graph = build_behavior_graph(events)
        search = CounterfactualSearch(graph=graph)
        candidates = search.propose()

        feasible = []
        for candidate in candidates:
            if validate_candidate(graph, candidate):
                feasible.append(candidate)

        if feasible:
            accepted += 1
            best = min(feasible, key=candidate_cost)
            minimality_total += candidate_cost(best)
            if candidate_cost(best) == 1:
                decoy_flip_count += 1

        # Structure check against a simple unconstrained baseline: any candidate
        # generated without feasibility checks is counted as structurally invalid.
        for candidate in candidates:
            if not validate_candidate(graph, candidate):
                unconstrained_impossible += 1
                break

        total_runtime += time.perf_counter() - start
        per_sample.append({
            "md5": md5,
            "family": row.get("avclass_family", "unknown"),
            "search_time_seconds": round(time.perf_counter() - start, 6),
            "feasible_candidates": len(feasible),
            "best_cost": min((candidate_cost(c) for c in feasible), default=0),
            "candidate_count": len(candidates),
        })

    total_samples = max(len(per_sample), 1)
    results: Dict[str, Any] = {
        "summary": {
            "samples_processed": total_samples,
            "tier1_feasibility_pass_rate": round((accepted / total_samples) * 100.0, 4) if total_samples else 0.0,
            "average_edit_distance": round(minimality_total / total_samples, 4) if total_samples else 0.0,
            "decoy_flip_rate": round((decoy_flip_count / total_samples) * 100.0, 4) if total_samples else 0.0,
            "unconstrained_baseline_impossible_rate": round((unconstrained_impossible / total_samples) * 100.0, 4) if total_samples else 0.0,
            "average_runtime_seconds": round(total_runtime / total_samples, 6) if total_samples else 0.0,
        },
        "samples": per_sample,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    results = evaluate_dataset()
    print(json.dumps(results["summary"], indent=2))
