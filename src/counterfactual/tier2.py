"""Tier-2 synthetic re-instantiation harness and checklist generator.

This module provides utilities to generate a synthetic CAPE-like report from
an edited behavior graph (useful for automated spot-check replays), and to
emit a short manual checklist that guides a researcher through manual
re-instantiation steps.
"""
from typing import Any
import json
import networkx as nx
from datetime import datetime, timezone


def generate_synthetic_cape_report(G: nx.DiGraph, out_path: str) -> None:
    """Write a minimal CAPE-like JSON where each node becomes a single call.

    The output is intentionally simple: it allows replaying the parser->graph
    pipeline to confirm that the edited graph can be represented as a trace.
    """
    processes = []
    calls_by_process = {}
    ts = 1
    for n, d in sorted(G.nodes(data=True), key=lambda item: item[0]):
        if d.get("entity_type") == "resource":
            continue
        api = d.get("api", "unknown")
        if not api or (str(api).lower() == "process" and not d.get("event_ids")):
            continue
        process_id = d.get("process_id") or "generated"
        calls = calls_by_process.setdefault(str(process_id), [])
        resources = [str(resource) for resource in (d.get("resources", []) or [])]
        args = {}
        for resource_index, resource in enumerate(resources):
            key = "path" if resource_index == 0 else f"path_{resource_index}"
            args[key] = resource
        repeat_count = max(1, int(d.get("count", 1) or 1))
        for _ in range(repeat_count):
            calls.append({"api": api, "timestamp": ts, "arguments": args})
            ts += 1

    for process_id, calls in calls_by_process.items():
        try:
            pid = int(process_id)
        except (TypeError, ValueError):
            pid = process_id
        processes.append({"pid": pid, "process_id": process_id, "calls": calls})
    report = {
        "behavior": {"processes": processes},
        "counterfactual_metadata": {
            "synthetic": True,
            "execution_status": "not_executed",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def generate_manual_checklist(result: dict, out_path: str) -> None:
    """Emit a short human checklist for manual Tier-2 re-instantiation.

    The checklist includes the candidate edit summary and suggested synthetic
    tests to run before attempting to patch or re-run real malware.
    """
    lines = []
    lines.append("Tier-2 Re-instantiation Checklist")
    lines.append("Generated: " + datetime.now(timezone.utc).isoformat())
    lines.append("")
    res = result.get("result", {})
    lines.append("Candidate summary:")
    lines.append(str(res))
    lines.append("")
    lines.append("Suggested steps:")
    lines.append("1. Review the candidate edit and confirm substitutions are semantically valid.")
    lines.append("2. Run the provided synthetic CAPE report through the parser and graph builder to confirm the edited graph is representable.")
    lines.append("3. If synthetic test passes, create a small synthetic test program that performs the edited sequence (prefer non-malicious equivalents), then run in CAPE and compare traces.")
    lines.append("4. For a single real-sample confirmation, patch the sample only if you have legal/ethical clearance and an isolated lab; otherwise, skip.")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Synthetic tests are not full proof but help catch simple infeasibilities.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
