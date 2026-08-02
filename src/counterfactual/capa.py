"""Lightweight ATT&CK/CAPA-style technique inference for the prototype."""
from typing import Any, Dict, List

import networkx as nx


def infer_capa_techniques(graph: nx.DiGraph) -> List[Dict[str, Any]]:
    """Return a compact list of inferred techniques from the edited graph."""
    techniques = []
    apis = {str(data.get("api", "")).lower() for _, data in graph.nodes(data=True)}

    if any(api in apis for api in {"createfile", "writefile"}):
        techniques.append({"technique": "T1565", "name": "Data Manipulation", "evidence": ["CreateFile", "WriteFile"]})
    if any(api in apis for api in {"regsetvalue", "createservice", "createscheduledtask"}):
        techniques.append({"technique": "T1543", "name": "Create or Modify System Process", "evidence": ["RegSetValue", "CreateService", "CreateScheduledTask"]})
    if not techniques:
        techniques.append({"technique": "T1203", "name": "Exploitation for Client Execution", "evidence": []})
    return techniques
