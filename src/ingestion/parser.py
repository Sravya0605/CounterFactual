"""CAPE JSON ingestion -> event list

This module provides a lightweight parser that converts a CAPE sandbox JSON
report into a list of event dicts with normalized fields used by the graph
builder.

The parser is intentionally tolerant: it tries to find common CAPE fields but
falls back gracefully when keys are missing. For production use, refine this
based on the exact CAPE JSON version you run.
"""
import json
import hashlib
from typing import List, Dict

RESOURCE_KEYS = ("file", "filename", "path", "regkey", "key", "ip", "domain", "url")


def _hash_event(evt: Dict) -> str:
    s = json.dumps(evt, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def parse_cape_json(path: str) -> List[Dict]:
    """Parse a CAPE JSON report and return a list of normalized events.

    Each event is a dict with keys: `id`, `api`, `timestamp`, `args`, `resources`.
    """
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)

    events = []

    # CAPE reports typically have report['behavior']['processes'] -> list
    processes = []
    if isinstance(report.get("behavior"), dict) and isinstance(report["behavior"].get("processes"), list):
        processes = report["behavior"]["processes"]
    elif isinstance(report.get("processes"), list):
        processes = report["processes"]

    # Flatten process->calls if present
    for proc in processes:
        calls = proc.get("calls") or proc.get("calls") or []
        for call in calls:
            api = call.get("api") or call.get("name") or "unknown"
            ts = call.get("timestamp") or call.get("time") or proc.get("timestamp")
            args = call.get("arguments") or call.get("args") or {}
            # Extract resources heuristically from args
            resources = []
            if isinstance(args, dict):
                for k, v in args.items():
                    key_low = str(k).lower()
                    if any(rk in key_low for rk in RESOURCE_KEYS):
                        resources.append(str(v))

            evt = {"api": api, "timestamp": ts, "args": args, "resources": resources}
            evt["id"] = _hash_event(evt)
            events.append(evt)

    # Fallback: some CAPE variants put calls under report['calls']
    if not events and isinstance(report.get("calls"), list):
        for call in report["calls"]:
            api = call.get("api") or call.get("name") or "unknown"
            ts = call.get("timestamp") or call.get("time")
            args = call.get("arguments") or call.get("args") or {}
            resources = []
            if isinstance(args, dict):
                for k, v in args.items():
                    key_low = str(k).lower()
                    if any(rk in key_low for rk in RESOURCE_KEYS):
                        resources.append(str(v))

            evt = {"api": api, "timestamp": ts, "args": args, "resources": resources}
            evt["id"] = _hash_event(evt)
            events.append(evt)

    return events


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: parser.py /path/to/cape_report.json")
        sys.exit(1)
    evts = parse_cape_json(sys.argv[1])
    print(f"Parsed {len(evts)} events")
