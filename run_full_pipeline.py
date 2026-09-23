"""End-to-end pipeline runner for the CounterFactual project.

Trains the malware/benign classifier using data/benign_reports and
data/training_reports. Counterfactual generation is then performed ONLY
on malware reports from data/training_reports.

When a feasible counterfactual that flips the classifier from malicious
to benign is found, the generated CAPE-like counterfactual report is
written to data/counterfactual_reports.

Run this from the repo root (it needs `src/` importable):

    python run_full_pipeline.py
    python run_full_pipeline.py --max-per-class 300
    python run_full_pipeline.py --max-counterfactual 20
    python run_full_pipeline.py --backend gnn

Expected folders:
    data/benign_reports/*.json       -- label 0, classifier training
    data/training_reports/*.json     -- label 1, classifier training AND
                                        counterfactual source samples
    data/counterfactual_reports/     -- generated successful flips
"""
import argparse
import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.behavior.resource_lifetimes import match_resource_lifetimes
from src.classifier.harness import ClassifierHarness
from src.counterfactual.engine import CounterfactualEngine


def build_graph(path):
    events = parse_cape_json(path)
    lt = match_resource_lifetimes(events)
    return build_behavior_graph(events, lifetimes=lt["lifetimes"], active_resources=lt["still_active"])


def list_paths(folder, max_count=None, rng=None):
    """List json files in folder, optionally shuffled before capping.

    Shuffling before slicing (rather than always taking the alphabetically
    first N) means --max-per-class samples a different, genuinely random
    subset each run when --seed changes, instead of the same fixed files
    every time.
    """
    paths = sorted(glob.glob(os.path.join(folder, "*.json")))
    if rng is not None:
        rng.shuffle(paths)
    if max_count:
        paths = paths[:max_count]
    return paths


def load_class(paths, label):
    graphs, labels, names, skipped = [], [], [], []
    for i, path in enumerate(paths, 1):
        try:
            graphs.append(build_graph(path))
            labels.append(label)
            names.append(os.path.basename(path))
        except Exception as exc:
            skipped.append((path, str(exc)))
        if i % 50 == 0:
            print(f"  ...{i}/{len(paths)} parsed")
    if skipped:
        print(f"  Skipped {len(skipped)} unreadable file(s):")
        for p, err in skipped[:5]:
            print(f"    {p}: {err}")
    return graphs, labels, names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--backend", default="lgbm", choices=["lgbm", "gnn", "heuristic"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--test-fraction", type=float, default=0.2,
                     help="Fraction of benign+training_reports held out to report classifier accuracy")
    ap.add_argument("--max-per-class", type=int, default=None,
                     help="Cap samples loaded per class -- useful for a quick smoke run before a full one")
    ap.add_argument("--max-counterfactual", type=int, default=None,
                     help="Cap how many malware reports from training_reports are tested for counterfactual flips")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pipeline_results.json")
    ap.add_argument("--epochs", type=int, default=10, help="GNN backend only: training epochs")
    ap.add_argument("--batch-size", type=int, default=16, help="GNN backend only: batch size")
    ap.add_argument("--rounds", type=int, default=100, help="lgbm backend only: boosting rounds")
    args = ap.parse_args()

    benign_dir = os.path.join(args.data_dir, "benign_reports")
    malware_dir = os.path.join(args.data_dir, "training_reports")
    cf_dir = os.path.join(args.data_dir, "counterfactual_reports")

    for d in (benign_dir, malware_dir, cf_dir):
        if not os.path.isdir(d):
            print(f"WARNING: expected folder not found: {d}")

    rng = random.Random(args.seed)

    benign_paths = list_paths(benign_dir, args.max_per_class, rng)
    malware_all_paths = list_paths(malware_dir, None, rng)  # shuffle full pool once
    malware_train_paths = malware_all_paths[: args.max_per_class] if args.max_per_class else malware_all_paths
    malware_train_set = set(malware_train_paths)

    # Counterfactual test files must be malware reports the classifier has
    # NEVER seen during training -- otherwise you're measuring how hard it
    # is to flip a memorized training example, which tends to look even
    # more confidently "malicious" than a genuine held-out case and makes
    # every result here misleadingly pessimistic.
    malware_holdout_paths = [p for p in malware_all_paths if p not in malware_train_set]
    if not malware_holdout_paths:
        print("WARNING: --max-per-class used the entire malware_dir for training, leaving "
              "nothing held out to test counterfactuals on. Lower --max-per-class or add more files.")

    print(f"=== Loading benign reports from {benign_dir} ===")
    benign_graphs, benign_labels, benign_names = load_class(benign_paths, 0)
    print(f"  {len(benign_graphs)} loaded")

    print(f"=== Loading malware reports from {malware_dir} ===")
    malware_graphs, malware_labels, malware_names = load_class(malware_train_paths, 1)
    print(f"  {len(malware_graphs)} loaded ({len(malware_holdout_paths)} more held out, untouched by training)")

    graphs = benign_graphs + malware_graphs
    labels = benign_labels + malware_labels

    if not graphs:
        print("No training data found -- check --data-dir and folder names.")
        return

    random.seed(args.seed)
    idx = list(range(len(graphs)))
    random.shuffle(idx)
    split = int(len(idx) * (1 - args.test_fraction))
    train_idx, test_idx = idx[:split], idx[split:]

    print(f"\n=== Training {args.backend} classifier: {len(train_idx)} train / {len(test_idx)} held-out ===")
    harness = ClassifierHarness(backend=args.backend, model_path=f"models/pipeline_{args.backend}.pkl")
    if args.backend == "gnn":
        harness.train([graphs[i] for i in train_idx], [labels[i] for i in train_idx],
                       epochs=args.epochs, batch_size=args.batch_size)
    else:
        harness.train([graphs[i] for i in train_idx], [labels[i] for i in train_idx],
                       rounds=args.rounds)

    accuracy = None
    if test_idx:
        probs = harness.predict_proba([graphs[i] for i in test_idx])
        preds = [1 if p >= args.threshold else 0 for p in probs]
        correct = sum(1 for i, p in zip(test_idx, preds) if p == labels[i])
        accuracy = correct / len(test_idx)
        tp = sum(1 for i, p in zip(test_idx, preds) if labels[i] == 1 and p == 1)
        fn = sum(1 for i, p in zip(test_idx, preds) if labels[i] == 1 and p == 0)
        tn = sum(1 for i, p in zip(test_idx, preds) if labels[i] == 0 and p == 0)
        fp = sum(1 for i, p in zip(test_idx, preds) if labels[i] == 0 and p == 1)
        print(f"Held-out accuracy: {correct}/{len(test_idx)} = {accuracy:.2%}   (TP={tp} FN={fn} TN={tn} FP={fp})")
        if len({round(p, 3) for p in probs}) <= 3:
            print("  NOTE: predicted probabilities cluster into <=3 distinct values -- "
                  "this project has previously found that pattern to indicate the classifier "
                  "may be relying on a shortcut rather than genuine behavioral signal. Worth checking "
                  "feature importances before trusting this accuracy number.")

    # Counterfactual generation is evaluated on HELD-OUT malware reports --
    # files the classifier never saw during training. See malware_holdout_paths
    # above.
    print(f"\n=== Running counterfactual pipeline on held-out malware reports from {malware_dir} ===")
    cf_paths = malware_holdout_paths
    rng.shuffle(cf_paths)
    if args.max_counterfactual:
        cf_paths = cf_paths[:args.max_counterfactual]
    print(f"  {len(cf_paths)} malware report(s) to test (drawn from {len(malware_holdout_paths)} available held-out files)")

    os.makedirs(cf_dir, exist_ok=True)

    engine = CounterfactualEngine(classifier_model=harness)

    results = []
    for i, path in enumerate(cf_paths, 1):
        name = os.path.basename(path)
        start = time.perf_counter()
        try:
            out = engine.explain_from_cape(path)
            res = out["result"]
            elapsed = time.perf_counter() - start

            # Only completed results represent a successful feasible flip.
            # The engine's generated CAPE-like report is expected in
            # res["counterfactual_report"] when a flip is found.
            generated_report = res.get("counterfactual_report")

            saved_path = None
            if res.get("status") == "completed" and generated_report is not None:
                stem = os.path.splitext(name)[0]
                saved_path = os.path.join(
                    cf_dir,
                    f"{stem}_counterfactual.json"
                )
                with open(saved_path, "w", encoding="utf-8") as f:
                    json.dump(generated_report, f, indent=2, default=str)

            row = {
                "file": name,
                "nodes": out.get("graph_nodes"),
                "edges": out.get("graph_edges"),
                "status": res.get("status"),
                "orig_prob": res.get("orig_prob"),
                "new_prob": res.get("new_prob"),
                "candidate": res.get("candidate"),
                "counterfactual_report": saved_path,
                "seconds": round(elapsed, 2),
            }
        except Exception as exc:
            row = {
                "file": name,
                "status": "error",
                "error": str(exc),
                "seconds": round(time.perf_counter() - start, 2),
            }

        results.append(row)
        print(
            f"  [{i}/{len(cf_paths)}] {name}: {row.get('status')}  "
            f"orig={row.get('orig_prob')}  new={row.get('new_prob')}  "
            f"saved={row.get('counterfactual_report')} ({row['seconds']}s)"
        )

    flipped = sum(1 for r in results if r.get("status") == "completed")
    not_malicious = sum(1 for r in results if r.get("status") == "not_malicious")
    no_flip = sum(1 for r in results if r.get("status") == "no_flip_found")
    no_feasible = sum(1 for r in results if r.get("status") in ("no_feasible_candidate", "candidates_available"))
    errors = sum(1 for r in results if r.get("status") == "error")

    print("\n=== Summary ===")
    print(f"  flipped (counterfactual found): {flipped}/{len(results)}")
    print(f"  already not malicious:          {not_malicious}/{len(results)}")
    print(f"  malicious, no flip found:       {no_flip}/{len(results)}")
    print(f"  no feasible candidate at all:   {no_feasible}/{len(results)}")
    print(f"  errors:                         {errors}/{len(results)}")
    print(f"  successful reports saved to:    {cf_dir}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "held_out_accuracy": accuracy,
                "counterfactual_source": malware_dir,
                "counterfactual_output": cf_dir,
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()