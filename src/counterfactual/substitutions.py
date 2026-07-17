"""Behavioral substitution library (curated pairs).

This small library provides a hand-curated set of behaviorally-equivalent
substitutions useful for prototyping node-substitution edits. Real research
should expand and validate this list; here we provide a simple programmatic
interface used by the counterfactual search skeleton.
"""
from typing import List

SUBSTITUTION_LIBRARY = {
    # common persistence patterns (heuristic keys)
    "RegSetValue": ["CreateScheduledTask", "CreateService"],
    "CreateService": ["RegSetValue", "CreateScheduledTask"],
    "CreateScheduledTask": ["RegSetValue", "CreateService"],
    # process injection variants
    "CreateRemoteThread": ["APCInject", "ProcessHollow"],
    "APCInject": ["CreateRemoteThread", "ProcessHollow"],
}


def get_substitutes(api_name: str) -> List[str]:
    """Return a list of substitute api names for a given api_name.

    Matches exact keys first; then falls back to substring matching.
    """
    if api_name in SUBSTITUTION_LIBRARY:
        return SUBSTITUTION_LIBRARY[api_name]
    # substring fallback
    for key, vals in SUBSTITUTION_LIBRARY.items():
        if key.lower() in api_name.lower():
            return vals
    return []
