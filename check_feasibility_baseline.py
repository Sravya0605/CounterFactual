"""Hunt for feasibility checks that reject their OWN unedited ground truth.

The litmus test: run every report's graph through validate_candidate() with
a true no-op edit (nothing deleted, nothing substituted, nothing inserted).
That graph IS the real trace, verbatim -- if the checker rejects it, the
checker is wrong, full stop, regardless of what candidate happened to
trigger it. This is exactly the test that found the NtReleaseMutant bug;
this script just runs it across your whole corpus instead of one sample,
and drills into *why* for anything that fails.

Usage (from repo root):
    python check_feasibility_baseline.py
    python check_feasibility_baseline.py --data-dir data --max-per-folder 500
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.behavior.resource_lifetimes import match_resource_lifetimes
from src.counterfactual.feasibility import (
    validate_candidate, apply_candidate, _check_resource_lifetime,
    _check_temporal_order, _check_argument_data_flow, _check_lifetime_entities,
    _matches_any, _node_timestamp, OPENING_API_TOKENS, CLOSING_API_TOKENS,
)

NOOP = {"delete_nodes": [], "delete_edges": [], "substitute": {}, "insert_nodes": []}


def build_graph(path):
    events = parse_cape_json(path)
    lt = match_resource_lifetimes(events)
    return build_behavior_graph(events, lifetimes=lt["lifetimes"], active_resources=lt["still_active"])


def diagnose_resource_lifetime(G):
    """Print the exact (resource, event-sequence) that's failing, same
    technique that found the NtReleaseMutant bug."""
    events_by_resource = {}
    for node, data in G.nodes(data=True):
        api = data.get("api")
        ts = _node_timestamp(data)
        for resource in data.get("resources", []) or []:
            events_by_resource.setdefault(resource, []).append((ts, api, node))

    for resource, evs in events_by_resource.items():
        evs.sort(key=lambda p: p[0])
        is_open = True
        for ts, api, node in evs:
            if _matches_any(api, CLOSING_API_TOKENS):
                is_open = False
            elif _matches_any(api, OPENING_API_TOKENS):
                is_open = True
            else:
                if not is_open:
                    print(f"    resource_lifetime violation on resource={resource!r}")
                    for ts2, api2, node2 in evs:
                        tag = "CLOSE" if _matches_any(api2, CLOSING_API_TOKENS) else ("OPEN" if _matches_any(api2, OPENING_API_TOKENS) else "  -  ")
                        print(f"      ts={ts2}  [{tag}]  api={api2}  node={node2}")
                    return


def diagnose(path, G):
    print(f"\nFAILS: {path}")
    G2 = apply_candidate(G, NOOP)
    if not _check_resource_lifetime(G2):
        diagnose_resource_lifetime(G)
    if not _check_temporal_order(G2):
        print("    temporal_order check fails on the unedited graph -- a temporal edge points backward in time already")
    if not _check_argument_data_flow(G, set(), G2):
        print("    argument_data_flow check fails on the unedited graph (with delete_nodes=set(), this should be structurally impossible -- worth a closer look)")
    if not _check_lifetime_entities(G2):
        for node, data in G2.nodes(data=True):
            if data.get("entity_type") != "resource":
                continue
            if not list(G2.predecessors(node)):
                print(f"    lifetime_entities: resource entity node {node!r} has no predecessor")
            if data.get("state") == "released" and not list(G2.successors(node)):
                print(f"    lifetime_entities: released resource entity node {node!r} has no successor")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--max-per-folder", type=int, default=None)
    args = ap.parse_args()

    paths = []
    for folder in ("benign_reports", "training_reports", "counterfactual_reports"):
        found = sorted(glob.glob(os.path.join(args.data_dir, folder, "*.json")))
        if args.max_per_folder:
            found = found[: args.max_per_folder]
        paths.extend(found)

    print(f"Testing {len(paths)} reports: does each one's own unedited graph pass validate_candidate()?")

    failures = 0
    errors = 0
    for i, path in enumerate(paths, 1):
        try:
            G = build_graph(path)
        except Exception as exc:
            errors += 1
            print(f"\nPARSE/BUILD ERROR: {path}: {exc}")
            continue
        if not validate_candidate(G, NOOP):
            failures += 1
            diagnose(path, G)
        if i % 200 == 0:
            print(f"  ...{i}/{len(paths)} checked, {failures} failures so far")

    print(f"\n=== Done: {len(paths)} tested, {failures} failed the no-op baseline check, {errors} parse/build errors ===")
    if failures == 0:
        print("No hidden global-poisoning bugs found in this corpus -- every graph accepts its own ground truth.")
    else:
        print("Each failure above needs its own fix, the same way the NtReleaseMutant one did:")
        print("find the exact rule doing the misclassifying, confirm it's wrong, fix it, and rerun this script.")


if __name__ == "__main__":
    main()