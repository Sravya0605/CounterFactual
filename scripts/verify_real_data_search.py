import json
import os
import random
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.behavior.resource_lifetimes import match_resource_lifetimes
from src.classifier.harness import ClassifierHarness
from src.counterfactual.feasibility import apply_candidate
from src.counterfactual.search import CounterfactualSearch
from src.graph.graph_builder import build_behavior_graph
from src.ingestion.parser import parse_cape_json


def main():
    report_files = [
        *((path, 0) for path in sorted((ROOT / "data" / "benign_reports").glob("*.json"))),
        *((path, 1) for path in sorted((ROOT / "data" / "training_reports").glob("*.json"))),
    ]
    random.Random(0).shuffle(report_files)
    split_index = int(len(report_files) * 0.8)
    train_files = report_files[:split_index]
    held_out_files = report_files[split_index:]

    print(f"REPORT_FILES total={len(report_files)} benign={sum(label == 0 for _, label in report_files)} malware={sum(label == 1 for _, label in report_files)}")
    print(f"SPLIT train={len(train_files)} held_out={len(held_out_files)} train_malware={sum(label == 1 for _, label in train_files)} held_out_malware={sum(label == 1 for _, label in held_out_files)}")

    graphs = []
    labels = []
    process = psutil.Process()
    for report_index, (path, label) in enumerate(report_files, start=1):
        try:
            events = parse_cape_json(str(path))
            lifetime_result = match_resource_lifetimes(events)
            print(f"building graph {report_index}: {os.path.basename(path)}", flush=True)
            graph = build_behavior_graph(
                events,
                lifetimes=lifetime_result["lifetimes"],
                active_resources=lifetime_result["still_active"],
            )
        except Exception as exc:
            print(f"GRAPH_BUILD_ERROR index={report_index} file={os.path.basename(path)!r} error={exc!r}", flush=True)
            traceback.print_exc()
            raise
        graphs.append(graph)
        labels.append(label)
        if report_index % 100 == 0 or report_index == len(report_files):
            rss_mb = process.memory_info().rss / (1024 * 1024)
            print(f"GRAPH_BUILD progress={report_index}/{len(report_files)} rss_mb={rss_mb:.2f}", flush=True)

    train_graphs = graphs[:split_index]
    train_labels = labels[:split_index]
    held_out_graphs = graphs[split_index:]
    held_out_labels = labels[split_index:]
    print(f"GRAPH_COUNTS total={len(graphs)} empty={sum(graph.number_of_nodes() == 0 for graph in graphs)} nodes={sum(graph.number_of_nodes() for graph in graphs)} edges={sum(graph.number_of_edges() for graph in graphs)}")
    print("TRAIN backend=gnn epochs=10 batch_size=16")

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory(prefix="counterfactual-real-data-") as temp_dir:
        harness = ClassifierHarness(backend="gnn", model_path=str(Path(temp_dir) / "gnn.pkl"))
        harness.train(train_graphs, train_labels, epochs=10, batch_size=16)

        held_out_probabilities = harness.predict_proba(held_out_graphs)
        held_out_predictions = [1 if probability >= 0.5 else 0 for probability in held_out_probabilities]
        correct = sum(prediction == label for prediction, label in zip(held_out_predictions, held_out_labels))
        accuracy = correct / len(held_out_labels)
        true_positive = sum(label == 1 and prediction == 1 for label, prediction in zip(held_out_labels, held_out_predictions))
        false_positive = sum(label == 0 and prediction == 1 for label, prediction in zip(held_out_labels, held_out_predictions))
        true_negative = sum(label == 0 and prediction == 0 for label, prediction in zip(held_out_labels, held_out_predictions))
        false_negative = sum(label == 1 and prediction == 0 for label, prediction in zip(held_out_labels, held_out_predictions))
        saturated_zero = sum(probability == 0.0 for probability in held_out_probabilities)
        saturated_one = sum(probability == 1.0 for probability in held_out_probabilities)

        print(f"HELD_OUT accuracy={accuracy!r} correct={correct} total={len(held_out_labels)}")
        print(f"CONFUSION_MATRIX TP={true_positive} FP={false_positive} TN={true_negative} FN={false_negative}")
        print(f"SATURATION exactly_0={saturated_zero} exactly_1={saturated_one} total={len(held_out_probabilities)}")

        malware_indices = [index for index, label in enumerate(held_out_labels) if label == 1]
        if len(malware_indices) < 5:
            raise RuntimeError(f"Need at least 5 held-out malware files; found {len(malware_indices)}")

        for malware_index in malware_indices[:5]:
            path = held_out_files[malware_index][0]
            graph = held_out_graphs[malware_index]
            started = time.perf_counter()
            orig_prob = harness.predict_proba([graph])[0]
            search = CounterfactualSearch(graph=graph)
            result = search.generate_candidates()
            feasible_candidates = result["feasible_candidates"]
            edit_size_distribution = Counter(
                len(candidate.get("delete_nodes", []))
                for candidate in feasible_candidates
            )
            multi_node_count = sum(size >= 2 for size in (
                len(candidate.get("delete_nodes", []))
                for candidate in feasible_candidates
            ))

            lowest_probability = None
            lowest_candidate = None
            for candidate in feasible_candidates:
                edited_graph = apply_candidate(graph, candidate)
                probability = harness.predict_proba([edited_graph])[0]
                if lowest_probability is None or probability < lowest_probability:
                    lowest_probability = probability
                    lowest_candidate = candidate

            genuine_flip = orig_prob >= 0.5 and lowest_probability is not None and lowest_probability < 0.5
            elapsed_seconds = time.perf_counter() - started
            print(f"MALWARE_FILE name={path.name!r} nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")
            print(f"ORIG_PROB {orig_prob!r}")
            print(f"FEASIBLE_CANDIDATES {len(feasible_candidates)}")
            print(f"EDIT_SIZE_DISTRIBUTION {edit_size_distribution!r}")
            print(f"MULTI_NODE_CANDIDATES {multi_node_count}")
            print(f"LOWEST_PROBABILITY {lowest_probability!r}")
            print(f"GENUINE_FLIP {genuine_flip}")
            if genuine_flip:
                print(f"FLIP_CANDIDATE {json.dumps(lowest_candidate, sort_keys=True, default=str)}")
            print(f"ELAPSED_SECONDS {elapsed_seconds!r}")


if __name__ == "__main__":
    main()