"""Behavioral substitution library for the prototype.

The library is intentionally small but explicit: each pair encodes a plausible
behavioral-equivalence claim that can be used in the counterfactual search.
"""
from typing import List

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
