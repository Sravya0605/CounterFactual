"""CAPE JSON ingestion -> normalized event list.

The parser keeps process context, event categories, and resource hints so the
subsequent graph builder can create process-aware entity graphs rather than a
flat event stream.
"""
import hashlib
import json
from typing import Any, Dict, List, Optional

RESOURCE_KEYS = ("file", "filename", "path", "regkey", "key", "ip", "domain", "url")
ATTACK_HINTS = {
    "createfile": "T1105",
    "writefile": "T1027",
    "regsetvalue": "T1547",
    "createscheduledtask": "T1053.005",
    "createservice": "T1543.003",
    "createremotethread": "T1055.003",
}


def _hash_event(evt: Dict[str, Any]) -> str:
    payload = json.dumps(evt, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _normalize_resource(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        values = [str(v) for v in value if v is not None]
        return ", ".join(values) if values else None
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _classify_event(api: str, args: Dict[str, Any]) -> Dict[str, Any]:
    normalized_api = (api or "unknown").lower()
    event_type = "system"
    if "file" in normalized_api or "create" in normalized_api:
        event_type = "file"
    if "reg" in normalized_api or "registry" in normalized_api:
        event_type = "registry"
    if "socket" in normalized_api or "connect" in normalized_api or "dns" in normalized_api:
        event_type = "network"
    if "process" in normalized_api or "thread" in normalized_api:
        event_type = "process"
    if "task" in normalized_api or "service" in normalized_api:
        event_type = "persistence"

    resources = []
    if isinstance(args, dict):
        for key, value in args.items():
            key_low = str(key).lower()
            if any(key_low == candidate or candidate in key_low for candidate in RESOURCE_KEYS):
                resource = _normalize_resource(value)
                if resource is not None:
                    resources.append(resource)
    attack_id = None
    for token, hint in ATTACK_HINTS.items():
        if token in normalized_api:
            attack_id = hint
            break
    return {"event_type": event_type, "resources": resources, "attack_id": attack_id}


def parse_cape_json(path: str) -> List[Dict[str, Any]]:
    """Parse a CAPE JSON report and return normalized events.

    Each event carries process_id, timestamp, event_type, attack_id, resources,
    and a stable id.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CAPE report not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in CAPE report: {path}") from exc

    events: List[Dict[str, Any]] = []
    processes: List[Dict[str, Any]] = []
    if isinstance(report.get("behavior"), dict) and isinstance(report["behavior"].get("processes"), list):
        processes = report["behavior"]["processes"]
    elif isinstance(report.get("processes"), list):
        processes = report["processes"]

    for proc in processes:
        proc_id = proc.get("pid") or proc.get("process_id") or proc.get("name") or "unknown"
        calls = proc.get("calls") or []
        for call in calls:
            api = call.get("api") or call.get("name") or "unknown"
            timestamp = call.get("timestamp") or call.get("time") or proc.get("timestamp")
            args = call.get("arguments") or call.get("args") or {}
            metadata = _classify_event(api, args)
            evt = {
                "id": None,
                "api": api,
                "process_id": proc_id,
                "timestamp": timestamp,
                "args": args,
                "event_type": metadata["event_type"],
                "resources": metadata["resources"],
                "attack_id": metadata["attack_id"],
            }
            evt["id"] = _hash_event(evt)
            events.append(evt)

    if not events and isinstance(report.get("calls"), list):
        for call in report["calls"]:
            api = call.get("api") or call.get("name") or "unknown"
            timestamp = call.get("timestamp") or call.get("time")
            args = call.get("arguments") or call.get("args") or {}
            metadata = _classify_event(api, args)
            evt = {
                "id": None,
                "api": api,
                "process_id": "root",
                "timestamp": timestamp,
                "args": args,
                "event_type": metadata["event_type"],
                "resources": metadata["resources"],
                "attack_id": metadata["attack_id"],
            }
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
