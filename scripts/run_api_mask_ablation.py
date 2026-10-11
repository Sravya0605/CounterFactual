"""Retrain the saved GNN run with selected API labels masked."""
import argparse
import gc
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_full_pipeline import build_graph
from src.classifier.harness import ClassifierHarness
from src.counterfactual.search import CounterfactualSearch

MASK_TOKEN = "<MASKED_API>"
DEFAULT_APIS = ("NtTestAlert", "GetSystemTimeAsFileTime")


def _probabilities(classifier, graphs):
    return [
        float(classifier.predict_proba([graph])[0])
        for graph in graphs
    ]


def _accuracy(probabilities, labels, threshold):
    if not labels:
        raise ValueError("Cannot compute accuracy for an empty test split")
    predictions = [int(probability >= threshold) for probability in probabilities]
    return {
        "accuracy": sum(
            prediction == label
            for prediction, label in zip(predictions, labels)
        ) / len(labels),
        "correct": sum(
            prediction == label
            for prediction, label in zip(predictions, labels)
        ),
        "total": len(labels),
        "threshold": threshold,
        "true_positive": sum(
            prediction == 1 and label == 1
            for prediction, label in zip(predictions, labels)
        ),
        "true_negative": sum(
            prediction == 0 and label == 0
            for prediction, label in zip(predictions, labels)
        ),
        "false_positive": sum(
            prediction == 1 and label == 0
            for prediction, label in zip(predictions, labels)
        ),
        "false_negative": sum(
            prediction == 0 and label == 1
            for prediction, label in zip(predictions, labels)
        ),
    }


def _mask_graph(graph, api_names):
    masked = 0
    for _, data in graph.nodes(data=True):
        api = str(data.get("api") or "")
        if api.casefold() in api_names:
            data["api"] = MASK_TOKEN
            masked += 1
    return masked


def _append_jsonl(handle, record):
    handle.write(json.dumps(record, default=str) + "\n")
    handle.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="pipeline_results.json")
    parser.add_argument("--model", default="models/pipeline_gnn.pkl")
    parser.add_argument("--out", default="api_mask_ablation.jsonl")
    parser.add_argument(
        "--apis",
        default=",".join(DEFAULT_APIS),
        help="Comma-separated API labels to replace with the shared mask token",
    )
    parser.add_argument(
        "--counterfactual-count",
        type=int,
        default=5,
        help="Number of the saved run's held-out malware reports to search (0 disables searches)",
    )
    parser.add_argument(
        "--successful-only",
        action="store_true",
        help="Select counterfactual reports marked completed in the saved results",
    )
    args = parser.parse_args()

    if args.counterfactual_count < 0:
        parser.error("--counterfactual-count must be non-negative")
    api_names = {
        api.strip().casefold()
        for api in args.apis.split(",")
        if api.strip()
    }
    if not api_names:
        parser.error("--apis must contain at least one API name")

    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = ROOT / results_path
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    if output_path.exists():
        parser.error(f"Output already exists; choose another --out path: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved = json.loads(results_path.read_text(encoding="utf-8"))
    config = saved["run_config"]
    if config.get("backend") != "gnn":
        parser.error("The results file must describe a GNN run")
    training_files = config["training_files"]
    counterfactual_files = config.get("counterfactual_files", [])
    seed = int(config["seed"])
    test_fraction = float(config["test_fraction"])
    threshold = float(config["threshold"])
    benign_dir = ROOT / Path(saved["counterfactual_source"]).parent / "benign_reports"
    malware_dir = ROOT / saved["counterfactual_source"]
    baseline_rows = {row["file"]: row for row in saved["results"]}
    if args.successful_only:
        successful_files = {
            name
            for name, row in baseline_rows.items()
            if row.get("status") == "completed"
        }
        counterfactual_files = [
            name for name in counterfactual_files if name in successful_files
        ]
    counterfactual_files = counterfactual_files[:args.counterfactual_count]

    paths_and_labels = []
    for name in training_files:
        benign_path = benign_dir / name
        malware_path = malware_dir / name
        if benign_path.is_file() == malware_path.is_file():
            raise FileNotFoundError(
                f"Expected {name} in exactly one training class directory"
            )
        path = benign_path if benign_path.is_file() else malware_path
        paths_and_labels.append((path, int(path.parent == malware_dir)))

    if not paths_and_labels:
        parser.error("No training reports listed in the saved run configuration")
    if not model_path.is_file():
        parser.error(f"Baseline GNN model not found: {model_path}")

    header = {
        "type": "metadata",
        "baseline_results": str(results_path),
        "baseline_model": str(model_path),
        "mask_token": MASK_TOKEN,
        "masked_apis": sorted(api_names),
        "seed": seed,
        "test_fraction": test_fraction,
        "threshold": threshold,
        "training_reports": len(paths_and_labels),
        "pipeline_recorded_held_out_accuracy": saved.get("held_out_accuracy"),
        "counterfactual_reports_requested": min(
            args.counterfactual_count, len(counterfactual_files)
        ),
        "successful_only": args.successful_only,
    }

    with output_path.open("x", encoding="utf-8") as output:
        _append_jsonl(output, header)
        started = time.perf_counter()
        graphs = []
        labels = []
        for index, (path, label) in enumerate(paths_and_labels, 1):
            graphs.append(build_graph(str(path)))
            labels.append(label)
            if index % 50 == 0:
                print(
                    f"Loaded {index}/{len(paths_and_labels)} training graphs",
                    flush=True,
                )

        random.seed(seed)
        indices = list(range(len(graphs)))
        random.shuffle(indices)
        split_at = int(len(indices) * (1 - test_fraction))
        train_indices, test_indices = indices[:split_at], indices[split_at:]
        test_graphs = [graphs[index] for index in test_indices]
        test_labels = [labels[index] for index in test_indices]

        baseline = ClassifierHarness(
            backend="gnn",
            model_path=str(model_path),
        )
        if baseline.model is None or baseline.api_vocab is None:
            raise RuntimeError("Could not load the saved baseline GNN and API vocabulary")
        baseline_probs = _probabilities(baseline, test_graphs)
        baseline_accuracy = _accuracy(baseline_probs, test_labels, threshold)
        del baseline

        mask_counts = Counter()
        train_index_set = set(train_indices)
        test_index_set = set(test_indices)
        for index, graph in enumerate(graphs):
            count = _mask_graph(graph, api_names)
            if index in train_index_set:
                mask_counts["train"] += count
            if index in test_index_set:
                mask_counts["test"] += count

        import torch

        random.seed(seed)
        torch.manual_seed(seed)
        with TemporaryDirectory(prefix="counterfactual_api_mask_") as temp_dir:
            masked_model_path = Path(temp_dir) / "masked_gnn.pkl"
            masked = ClassifierHarness(
                backend="gnn",
                model_path=str(masked_model_path),
            )
            masked.train(
                [graphs[index] for index in train_indices],
                [labels[index] for index in train_indices],
                epochs=int(config["epochs"]),
                batch_size=int(config["batch_size"]),
                pooling=config["gnn_pooling"],
            )
            masked_probs = _probabilities(masked, test_graphs)
            masked_accuracy = _accuracy(masked_probs, test_labels, threshold)

            accuracy_record = {
                "type": "accuracy",
                "baseline": baseline_accuracy,
                "masked": masked_accuracy,
                "delta": masked_accuracy["accuracy"] - baseline_accuracy["accuracy"],
                "masked_api_node_counts": dict(mask_counts),
                "training_seconds": round(time.perf_counter() - started, 2),
            }
            _append_jsonl(output, accuracy_record)
            print(
                f"Baseline accuracy: {baseline_accuracy['accuracy']:.3%} "
                f"({baseline_accuracy['correct']}/{baseline_accuracy['total']})"
            )
            print(
                f"Masked accuracy:   {masked_accuracy['accuracy']:.3%} "
                f"({masked_accuracy['correct']}/{masked_accuracy['total']})"
            )
            print(f"Accuracy delta:     {accuracy_record['delta']:+.3%}")

            del graphs, labels, test_graphs, test_labels, baseline_probs, masked_probs
            gc.collect()

            cf_files = counterfactual_files
            cf_counts = Counter()
            for index, name in enumerate(cf_files, 1):
                path = malware_dir / name
                graph = build_graph(str(path))
                masked_nodes = _mask_graph(graph, api_names)
                baseline_candidate = baseline_rows.get(name, {}).get("candidate")
                masked_original_probability = float(
                    masked.predict_proba([graph])[0]
                )
                masked_same_edit_probability = (
                    float(masked.predict_candidate_proba(graph, baseline_candidate))
                    if baseline_candidate
                    else None
                )
                search = CounterfactualSearch(graph)
                cf_started = time.perf_counter()
                result = search.find_flip(masked)
                record = {
                    "type": "counterfactual",
                    "file": name,
                    "baseline_status": baseline_rows.get(name, {}).get("status"),
                    "baseline_original_probability": baseline_rows.get(name, {}).get("orig_prob"),
                    "baseline_best_probability": baseline_rows.get(name, {}).get("best_prob"),
                    "baseline_new_probability": baseline_rows.get(name, {}).get("new_prob"),
                    "masked_nodes": masked_nodes,
                    "baseline_winning_candidate": baseline_candidate,
                    "masked_original_probability": masked_original_probability,
                    "masked_baseline_winning_edit_probability": masked_same_edit_probability,
                    "masked_baseline_winning_edit_flips": (
                        masked_same_edit_probability < threshold
                        if masked_same_edit_probability is not None
                        else None
                    ),
                    "status": result.get("status"),
                    "original_probability": result.get("orig_prob"),
                    "best_probability": result.get("best_prob"),
                    "new_probability": result.get("new_prob"),
                    "scored_candidates": result.get("scored_candidates"),
                    "candidate": result.get("candidate"),
                    "seconds": round(time.perf_counter() - cf_started, 2),
                }
                _append_jsonl(output, record)
                cf_counts[record["status"]] += 1
                print(
                    f"[{index}/{len(cf_files)}] {name}: "
                    f"baseline={record['baseline_status']} "
                    f"masked={record['status']} "
                    f"orig={record['original_probability']} "
                    f"best={record['best_probability']} "
                    f"({record['seconds']}s)",
                    flush=True,
                )

            summary = {
                "type": "counterfactual_summary",
                "reports": len(cf_files),
                "baseline_flips": sum(
                    baseline_rows.get(name, {}).get("status") == "completed"
                    for name in cf_files
                ),
                "masked_status_counts": dict(cf_counts),
                "elapsed_seconds": round(time.perf_counter() - started, 2),
            }
            _append_jsonl(output, summary)
            print(f"Results written incrementally to: {output_path}")


if __name__ == "__main__":
    main()
