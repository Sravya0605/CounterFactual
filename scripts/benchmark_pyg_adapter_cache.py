"""Benchmark repeated PyG conversion of one real CAPE report graph."""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_full_pipeline import build_graph
from src.utils.pyg_adapter import (
    _node_feature_vector_cached,
    build_api_vocab,
    graph_to_pyg_data,
    graph_to_pyg_data_uncached,
)


DEFAULT_REPORT = (
    ROOT
    / "data"
    / "training_reports"
    / "cc6b71e9ad9098c329427da3006bc00626bf79b1b4c0523373bda7ad20eb71bf_task_863_report.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    graph = build_graph(str(args.report))
    api_vocab = build_api_vocab([graph])

    # Keep report parsing and graph construction outside both timed regions.
    # Warm each conversion path once to avoid charging one-time setup to either.
    uncached_sample = graph_to_pyg_data_uncached(graph, api_vocab)
    _node_feature_vector_cached.cache_clear()
    cached_sample = graph_to_pyg_data(graph, api_vocab)
    if not (
        uncached_sample.x.equal(cached_sample.x)
        and uncached_sample.edge_index.equal(cached_sample.edge_index)
        and uncached_sample.edge_attr.equal(cached_sample.edge_attr)
    ):
        raise AssertionError("Cached and uncached conversions produced different tensors")
    del uncached_sample, cached_sample

    started = time.perf_counter()
    for _ in range(args.iterations):
        data = graph_to_pyg_data_uncached(graph, api_vocab)
        del data
    uncached_elapsed = time.perf_counter() - started

    _node_feature_vector_cached.cache_clear()
    graph_to_pyg_data(graph, api_vocab)
    started = time.perf_counter()
    for _ in range(args.iterations):
        data = graph_to_pyg_data(graph, api_vocab)
        del data
    cached_elapsed = time.perf_counter() - started

    print(f"report={args.report.name}")
    print(f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")
    print(f"iterations={args.iterations}")
    print("timed_scope=graph_to_pyg_data conversion only (graph load/vocab setup excluded)")
    print(f"uncached_total_seconds={uncached_elapsed:.6f}")
    print(f"cached_total_seconds={cached_elapsed:.6f}")
    print(f"speedup_ratio={uncached_elapsed / cached_elapsed:.2f}x")


if __name__ == "__main__":
    main()
