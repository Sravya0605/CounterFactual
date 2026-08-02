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
        if resource_text.startswith(f"{prefix}:"):
            updated.append(resource_text)
        else:
            updated.append(f"{prefix}:{resource_text}")
    return updated

SUBSTITUTION_LIBRARY = {
    "RegSetValue": ["CreateScheduledTask", "CreateService"],
    "CreateService": ["RegSetValue", "CreateScheduledTask"],
    "CreateScheduledTask": ["RegSetValue", "CreateService"],
    "CreateRemoteThread": ["APCInject", "ProcessHollow"],
    "APCInject": ["CreateRemoteThread", "ProcessHollow"],
    "WriteFile": ["CreateFile"],
    "CreateFile": ["WriteFile"],
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
