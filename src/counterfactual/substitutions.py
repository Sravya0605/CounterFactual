"""Behavioral substitution library for the prototype.

The library is intentionally small but explicit: each pair encodes a plausible
behavioral-equivalence claim that can be used in the counterfactual search.
"""
from typing import List


def update_resources_for_substitution(api_name: str, original_resources: List[str] | None = None) -> List[str]:
    """Return a resource list that reflects the substituted API."""
    resources = list(original_resources or [])
    if not resources:
        return [api_name.lower()] if api_name else []
    prefix = (api_name or "").lower()
    updated = []
    for resource in resources:
        resource_text = str(resource)
        # If the resource already appears to be namespaced (prefix:suffix),
        # replace the prefix with the new API prefix rather than naively
        # stacking prefixes. This produces more realistic resource names for
        # substitutions that change persistence mechanism.
        if ":" in resource_text:
            parts = resource_text.split(":", 1)
            updated.append(f"{prefix}:{parts[1]}")
        else:
            updated.append(f"{prefix}:{resource_text}")
    return updated
SUBSTITUTION_LIBRARY = {
    "RegSetValue": ["CreateScheduledTask", "CreateService"],
    "CreateService": ["RegSetValue", "CreateScheduledTask"],
    "CreateScheduledTask": ["RegSetValue", "CreateService"],
    "CreateRemoteThread": ["APCInject", "ProcessHollow"],
    "APCInject": ["CreateRemoteThread", "ProcessHollow"],
}

# Provenance mapping for substitution pairs. Each key is (original, substitute)
# and the value is a short comment indicating why the substitution is plausible
# (e.g., observed behavior, literature, CAPE heuristics). These strings are
# intended for traceability in results and reporting.
SUBSTITUTION_PROVENANCE = {
    ("RegSetValue", "CreateScheduledTask"): "observed registry->scheduledtask usage in replayed samples",
    ("RegSetValue", "CreateService"): "registry used to persist configuration for a service",
    ("CreateService", "RegSetValue"): "service creation can be replaced by registry persistence in some campaigns",
    ("CreateScheduledTask", "RegSetValue"): "scheduled tasks often set registry keys for persistence",
    ("CreateRemoteThread", "APCInject"): "APC vs CreateRemoteThread are both known code-injection patterns",
    ("CreateRemoteThread", "ProcessHollow"): "Process hollowing is alternative injection technique",
    ("APCInject", "CreateRemoteThread"): "APC injection and CreateRemoteThread are interchangeable in many cases",
}


def get_substitutes(api_name: str) -> List[str]:
    """Return a list of substitute API names for a given API."""
    if api_name is None:
        return []
    if api_name in SUBSTITUTION_LIBRARY:
        return SUBSTITUTION_LIBRARY[api_name]
    normalized = api_name.lower()
    for key, values in SUBSTITUTION_LIBRARY.items():
        if normalized == key.lower():
            return values
        if key.lower() in normalized:
            return values
    return []


def get_substitution_provenance(original: str, substitute: str) -> str | None:
    """Return provenance comment for a substitution pair if available."""
    return SUBSTITUTION_PROVENANCE.get((original, substitute))
