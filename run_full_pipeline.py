"""End-to-end pipeline runner for the CounterFactual project.

Trains the malware/benign classifier using data/benign_reports and
data/training_reports. Counterfactual generation is then performed ONLY
on malware reports from data/training_reports.

When a feasible counterfactual that flips the classifier from malicious
to benign is found, its explanation JSON is written to
data/counterfactual_reports. The synthetic Tier-2 round-trip report is
validated in memory and is not written to disk.

Run this from the repo root (it needs `src/` importable):

    python run_full_pipeline.py
    python run_full_pipeline.py --max-per-class 300
    python run_full_pipeline.py --max-counterfactual 20
    python run_full_pipeline.py --backend gnn
    python run_full_pipeline.py --workers 2

Expected folders:
    data/benign_reports/*.json       -- label 0, classifier training
    data/training_reports/*.json     -- label 1, classifier training AND
                                        counterfactual source samples
    data/counterfactual_reports/     -- successful counterfactual explanations
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import glob
import json
import multiprocessing
import os
import random
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.behavior.resource_lifetimes import match_resource_lifetimes
from src.classifier.harness import ClassifierHarness
from src.counterfactual.engine import CounterfactualEngine

_PARALLEL_ENGINE = None


def _counterfactual_worker(
    connection,
    cape_json_path,
    backend,
    model_path,
    excluded_apis=None,
):
    try:
        classifier = ClassifierHarness(backend=backend, model_path=model_path)
        output = CounterfactualEngine(
            classifier_model=classifier,
            excluded_apis=excluded_apis,
        ).explain_from_cape(cape_json_path)
        connection.send({"output": output})
    except Exception as exc:
        connection.send({"error": str(exc), "traceback": traceback.format_exc()})
    finally:
        connection.close()


def _initialize_parallel_worker(backend, model_path, excluded_apis=None):
    global _PARALLEL_ENGINE
    classifier = ClassifierHarness(backend=backend, model_path=model_path)
    _PARALLEL_ENGINE = CounterfactualEngine(
        classifier_model=classifier,
        excluded_apis=excluded_apis,
    )


def _run_parallel_counterfactual_task(path):
    if _PARALLEL_ENGINE is None:
        raise RuntimeError("Counterfactual process worker was not initialized")
    started = time.perf_counter()
    result = {"output": _PARALLEL_ENGINE.explain_from_cape(path)}
    return result, time.perf_counter() - started


def run_counterfactual_with_timeout(
    engine,
    cape_json_path,
    backend,
    model_path,
    timeout_seconds=None,
    worker_target=None,
    force_process=False,
    excluded_apis=None,
):
    if timeout_seconds is None and not force_process:
        return {"output": engine.explain_from_cape(cape_json_path)}

    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process_target = worker_target or _counterfactual_worker
    process_args = (
        child_connection,
        cape_json_path,
        backend,
        model_path,
    )
    if worker_target is None:
        process_args += (excluded_apis,)
    process = context.Process(
        target=process_target,
        args=process_args,
    )
    started = time.perf_counter()
    try:
        try:
            process.start()
        finally:
            child_connection.close()
        if parent_connection.poll(timeout_seconds):
            try:
                message = parent_connection.recv()
            except EOFError:
                process.join()
                return {
                    "error": f"Worker exited without a result (exit code {process.exitcode})"
                }
            process.join()
            return message

        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        return {
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        parent_connection.close()
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join()


def _run_counterfactual_task(
    engine,
    path,
    backend,
    model_path,
    timeout_seconds,
    force_process,
    excluded_apis=None,
):
    started = time.perf_counter()
    result = run_counterfactual_with_timeout(
        engine,
        path,
        backend,
        model_path,
        timeout_seconds,
        force_process=force_process,
        excluded_apis=excluded_apis,
    )
    return result, time.perf_counter() - started


def _checkpoint_path(output_path):
    return f"{output_path}.checkpoint.jsonl"


def _load_checkpoint(checkpoint_path, run_config):
    with open(checkpoint_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    if not lines:
        raise ValueError(f"Checkpoint is empty: {checkpoint_path}")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid checkpoint header: {checkpoint_path}") from exc
    if header.get("type") != "metadata":
        raise ValueError(f"Invalid checkpoint header type: {checkpoint_path}")
    if header.get("run_config") != run_config:
        raise ValueError(
            "Checkpoint configuration or selected files differ from this run; "
            "choose a new --out path or remove --resume."
        )

    rows = {}
    for index, line in enumerate(lines[1:], 2):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) and not line.endswith("\n"):
                break
            raise ValueError(
                f"Invalid checkpoint record at line {index}: {checkpoint_path}"
            ) from exc
        if record.get("type") != "result" or not isinstance(record.get("row"), dict):
            raise ValueError(f"Invalid checkpoint record at line {index}")
        rows[record["row"]["file"]] = record["row"]
    return header, rows


def _write_checkpoint_header(checkpoint_path, header):
    parent = os.path.dirname(os.path.abspath(checkpoint_path))
    os.makedirs(parent, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_checkpoint_row(checkpoint_path, row):
    with open(checkpoint_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "result", "row": row}, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(output_path, payload):
    parent = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(parent, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=parent,
            prefix=f".{os.path.basename(output_path)}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(payload, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _save_counterfactual_explanation(output, input_name, output_dir):
    stem = os.path.splitext(os.path.basename(input_name))[0]
    output_path = os.path.join(
        output_dir,
        f"{stem}_counterfactual_explanation.json",
    )
    explanation = dict(output)
    result = dict(explanation.get("result") or {})
    result.pop("counterfactual_report", None)
    explanation["result"] = result
    explanation["artifact_type"] = "counterfactual_explanation"
    _atomic_write_json(output_path, explanation)
    return output_path


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


def load_class(paths, label, max_graph_nodes=None):
    graphs, labels, names, skipped = [], [], [], []
    for i, path in enumerate(paths, 1):
        try:
            graph = build_graph(path)
            if max_graph_nodes is not None and graph.number_of_nodes() > max_graph_nodes:
                skipped.append((path, f"skipped: node count {graph.number_of_nodes()} exceeds --max-graph-nodes={max_graph_nodes}"))
                print(f"  Skipping {os.path.basename(path)}: {graph.number_of_nodes()} nodes exceeds --max-graph-nodes={max_graph_nodes} (graph object is too large to keep in the training list)")
                continue
            graphs.append(graph)
            labels.append(label)
            names.append(os.path.basename(path))
        except Exception as exc:
            skipped.append((path, str(exc)))
        if i % 50 == 0:
            print(f"  ...{i}/{len(paths)} parsed")
    if skipped:
        print(f"  Skipped {len(skipped)} file(s):")
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
    ap.add_argument("--max-graph-nodes", type=int, default=None,
                     help="Skip any report whose behavior graph exceeds this node count; useful for pruning pathological giant reports that blow up memory before training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pipeline_results.json")
    ap.add_argument("--counterfactual-output-dir", default=None,
                    help="Directory for successful counterfactual explanations (default: <data-dir>/counterfactual_reports)")
    ap.add_argument("--epochs", type=int, default=10, help="GNN backend only: training epochs")
    ap.add_argument("--batch-size", type=int, default=16, help="GNN backend only: batch size")
    ap.add_argument("--gnn-pooling", choices=["mean_max", "attention"], default="mean_max",
                     help="GNN graph pooling method (default preserves mean+max pooling)")
    ap.add_argument(
        "--exclude-boilerplate",
        action="store_true",
        help="Exclude boilerplate API nodes from every counterfactual candidate pool",
    )
    ap.add_argument(
        "--boilerplate-apis-file",
        default="api_prevalence_oct10.json",
        help="JSON output from scripts/analyze_training_api_prevalence.py",
    )
    ap.add_argument(
        "--reuse-model",
        action="store_true",
        help="Load the existing pipeline model instead of training and overwriting it",
    )
    ap.add_argument("--per-file-timeout", type=float, default=None,
                    help="Stop an individual counterfactual search after this many seconds; disabled by default")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of reports to search concurrently (higher values use more memory; default: 1)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume completed reports from --out.checkpoint.jsonl when its run configuration matches")
    ap.add_argument("--rounds", type=int, default=100, help="lgbm backend only: boosting rounds")
    args = ap.parse_args()
    if args.per_file_timeout is not None and args.per_file_timeout <= 0:
        ap.error("--per-file-timeout must be greater than zero")
    if args.workers < 1:
        ap.error("--workers must be at least 1")
    excluded_apis = []
    if args.exclude_boilerplate:
        boilerplate_path = args.boilerplate_apis_file
        if not os.path.isabs(boilerplate_path):
            boilerplate_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                boilerplate_path,
            )
        try:
            with open(boilerplate_path, "r", encoding="utf-8") as handle:
                prevalence = json.load(handle)
            excluded_apis = list(prevalence["boilerplate_candidates"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            ap.error(
                f"Unable to load boilerplate API list from {boilerplate_path}: {exc}"
            )
        if not excluded_apis:
            ap.error(
                f"No boilerplate candidates found in {boilerplate_path}"
            )

    benign_dir = os.path.join(args.data_dir, "benign_reports")
    malware_dir = os.path.join(args.data_dir, "training_reports")
    cf_dir = args.counterfactual_output_dir or os.path.join(args.data_dir, "counterfactual_reports")

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
    benign_graphs, benign_labels, benign_names = load_class(benign_paths, 0, args.max_graph_nodes)
    print(f"  {len(benign_graphs)} loaded")

    print(f"=== Loading malware reports from {malware_dir} ===")
    malware_graphs, malware_labels, malware_names = load_class(malware_train_paths, 1, args.max_graph_nodes)
    print(f"  {len(malware_graphs)} loaded ({len(malware_holdout_paths)} more held out, untouched by training)")

    graphs = benign_graphs + malware_graphs
    labels = benign_labels + malware_labels

    if not graphs:
        print("No training data found -- check --data-dir and folder names.")
        return

    random.seed(args.seed)
    if args.backend == "gnn":
        import torch

        torch.manual_seed(args.seed)
    idx = list(range(len(graphs)))
    random.shuffle(idx)
    split = int(len(idx) * (1 - args.test_fraction))
    train_idx, test_idx = idx[:split], idx[split:]

    print(f"\n=== Training {args.backend} classifier: {len(train_idx)} train / {len(test_idx)} held-out ===")
    model_path = f"models/pipeline_{args.backend}.pkl"
    harness = ClassifierHarness(backend=args.backend, model_path=model_path)
    if args.reuse_model:
        if harness.model is None:
            ap.error(f"--reuse-model requested but no saved model exists: {model_path}")
        print(f"Reusing existing model: {model_path}")
    elif args.backend == "gnn":
        harness.train([graphs[i] for i in train_idx], [labels[i] for i in train_idx],
                      epochs=args.epochs, batch_size=args.batch_size,
                      pooling=args.gnn_pooling)
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

    engine = CounterfactualEngine(
        classifier_model=harness,
        excluded_apis=excluded_apis,
    )
    checkpoint_path = _checkpoint_path(args.out)
    run_config = {
        "backend": args.backend,
        "threshold": args.threshold,
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "max_per_class": args.max_per_class,
        "max_graph_nodes": args.max_graph_nodes,
        "max_counterfactual": args.max_counterfactual,
        "counterfactual_output": cf_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gnn_pooling": args.gnn_pooling,
        "rounds": args.rounds,
        "exclude_boilerplate": args.exclude_boilerplate,
        "excluded_apis": excluded_apis,
        "reuse_model": args.reuse_model,
        "training_files": benign_names + malware_names,
        "counterfactual_files": [os.path.basename(path) for path in cf_paths],
    }
    checkpoint_header = {
        "type": "metadata",
        "held_out_accuracy": accuracy,
        "counterfactual_source": malware_dir,
        "counterfactual_output": cf_dir,
        "run_config": run_config,
    }
    if args.resume:
        if not os.path.exists(checkpoint_path):
            ap.error(f"--resume requested, but checkpoint does not exist: {checkpoint_path}")
        _, result_by_file = _load_checkpoint(checkpoint_path, run_config)
        print(f"  Resuming from {len(result_by_file)} checkpointed result(s)")
    else:
        result_by_file = {}
        _write_checkpoint_header(checkpoint_path, checkpoint_header)

    total_files = len(cf_paths)
    use_persistent_workers = args.workers > 1 and args.per_file_timeout is None
    total_started = time.perf_counter()
    if use_persistent_workers:
        # ProcessPoolExecutor keeps one classifier/model loaded per worker
        # instead of spawning and reloading it for each report.
        worker_executor = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_parallel_worker,
            initargs=(args.backend, model_path, excluded_apis),
        )
    else:
        worker_executor = ThreadPoolExecutor(max_workers=args.workers)
    with worker_executor as executor:
        futures = {}
        for i, path in enumerate(cf_paths, 1):
            name = os.path.basename(path)
            existing = result_by_file.get(name)
            if existing:
                saved_explanation = existing.get("counterfactual_explanation")
                completed_artifacts_exist = (
                    existing.get("status") == "completed"
                    and saved_explanation
                    and os.path.isfile(saved_explanation)
                )
                if (
                    existing.get("status") != "completed"
                    and existing.get("status") not in {"error", "timed_out"}
                ) or completed_artifacts_exist:
                    print(f"  [{i}/{total_files}] {name}: already checkpointed; skipping")
                    continue

            submitted_at = time.perf_counter()
            if use_persistent_workers:
                future = executor.submit(_run_parallel_counterfactual_task, path)
            else:
                future = executor.submit(
                    _run_counterfactual_task,
                    engine,
                    path,
                    args.backend,
                    model_path,
                    args.per_file_timeout,
                    False,
                    excluded_apis,
                )
            futures[future] = (i, path, name, submitted_at)

        for future in as_completed(futures):
            i, path, name, submitted_at = futures[future]
            try:
                worker_result, elapsed = future.result()
                if worker_result.get("timed_out"):
                    row = {
                        "file": name,
                        "status": "timed_out",
                        "timeout_seconds": args.per_file_timeout,
                        "seconds": round(worker_result["elapsed_seconds"], 2),
                    }
                    result_by_file[name] = row
                    _append_checkpoint_row(checkpoint_path, row)
                    print(
                        f"  [{i}/{total_files}] {name}: timed_out after "
                        f"{row['seconds']}s (limit={args.per_file_timeout}s)"
                    )
                    continue
                if "error" in worker_result:
                    raise RuntimeError(
                        f"{worker_result['error']}\n{worker_result.get('traceback', '')}"
                    )
                out = worker_result["output"]
                res = out["result"]

                explanation_path = None
                if res.get("status") == "completed":
                    if not res.get("explanation") or not res.get("explanation", {}).get("edits"):
                        raise ValueError(
                            "Counterfactual search completed without explanation edit details"
                        )
                    explanation_path = _save_counterfactual_explanation(
                        out,
                        name,
                        cf_dir,
                    )

                row = {
                    "file": name,
                    "nodes": out.get("graph_nodes"),
                    "edges": out.get("graph_edges"),
                    "status": res.get("status"),
                    "orig_prob": res.get("orig_prob"),
                    "best_prob": res.get("best_prob"),
                    "new_prob": res.get("new_prob"),
                    "candidate": res.get("candidate"),
                    "counterfactual_explanation": explanation_path,
                    "seconds": round(elapsed, 2),
                }
            except Exception as exc:
                row = {
                    "file": name,
                    "status": "error",
                    "error": str(exc),
                    "seconds": round(time.perf_counter() - submitted_at, 2),
                }

            result_by_file[name] = row
            _append_checkpoint_row(checkpoint_path, row)
            print(
                f"  [{i}/{total_files}] {name}: {row.get('status')}  "
                f"orig={row.get('orig_prob')}  best={row.get('best_prob')}  "
                f"new={row.get('new_prob')}  "
                f"saved={row.get('counterfactual_explanation')} "
                f"({row['seconds']}s)"
            )
    total = time.perf_counter() - total_started
    print(f"Total wall-clock time: {total:.1f}s")

    results = [result_by_file[os.path.basename(path)] for path in cf_paths if os.path.basename(path) in result_by_file]
    flipped = sum(1 for r in results if r.get("status") == "completed")
    not_malicious = sum(1 for r in results if r.get("status") == "not_malicious")
    no_flip = sum(1 for r in results if r.get("status") == "no_flip_found")
    no_feasible = sum(1 for r in results if r.get("status") in ("no_feasible_candidate", "candidates_available"))
    timed_out = sum(1 for r in results if r.get("status") == "timed_out")
    errors = sum(1 for r in results if r.get("status") == "error")

    print("\n=== Summary ===")
    print(f"  flipped (counterfactual found): {flipped}/{len(results)}")
    print(f"  already not malicious:          {not_malicious}/{len(results)}")
    print(f"  malicious, no flip found:       {no_flip}/{len(results)}")
    print(f"  no feasible candidate at all:   {no_feasible}/{len(results)}")
    print(f"  timed out:                      {timed_out}/{len(results)}")
    print(f"  errors:                         {errors}/{len(results)}")
    print(f"  successful explanations saved: {cf_dir}")
    print(f"  per-flip explanation files:     *_counterfactual_explanation.json")
    print(f"  incremental checkpoint:         {checkpoint_path}")

    _atomic_write_json(
        args.out,
        {
            "held_out_accuracy": accuracy,
            "counterfactual_source": malware_dir,
            "counterfactual_output": cf_dir,
            "run_config": run_config,
            "checkpoint_file": checkpoint_path,
            "results": results,
        },
    )
    print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()