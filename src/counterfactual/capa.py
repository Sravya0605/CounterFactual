"""Lightweight ATT&CK/CAPA-style technique inference for the prototype."""
from typing import Any, Dict, List

import networkx as nx


def infer_capa_techniques(graph: nx.DiGraph) -> List[Dict[str, Any]]:
    """Return a compact list of inferred techniques from the edited graph."""
    names = {
        "T1027": "Obfuscated/Compressed Files or Information",
        "T1055.003": "Process Injection: Thread Execution Hijacking",
        "T1053.005": "Scheduled Task/Job: Scheduled Task",
        "T1105": "Ingress Tool Transfer",
        "T1543.003": "Create or Modify System Process: Windows Service",
        "T1547": "Boot or Logon Autostart Execution",
    }
    grouped: Dict[str, List[str]] = {}
    for _, data in graph.nodes(data=True):
        attack_id = data.get("attack_id")
        if attack_id:
            grouped.setdefault(str(attack_id), []).append(str(data.get("api", "unknown")))
    return [
        {"technique": technique, "name": names.get(technique, "ATT&CK technique"), "evidence": sorted(set(evidence))}
        for technique, evidence in sorted(grouped.items())
    ]
