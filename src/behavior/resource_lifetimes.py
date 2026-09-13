"""Process- and handle-aware resource lifetime reconstruction."""
from collections import defaultdict
from typing import Dict, List

from src.behavior.resources import extract_acquisition, extract_release


def match_resource_lifetimes(events: List[Dict]) -> Dict[str, list]:
    acquisitions = [item for event in events if (item := extract_acquisition(event)) is not None]
    releases = [item for event in events if (item := extract_release(event)) is not None]
    active = defaultdict(list)
    timeline = [
        (item.get("sequence", 0), 0, "acquisition", item) for item in acquisitions
    ] + [
        (item.get("sequence", 0), 1, "release", item) for item in releases
    ]
    timeline.sort(key=lambda item: (item[0], item[1]))

    lifetimes = []
    orphan_releases = []
    for _, _, event_type, item in timeline:
        handle = item.get("handle")
        if handle is None:
            continue
        key = (item.get("process_id"), handle)
        if event_type == "acquisition":
            active[key].append(item)
            continue
        if not active.get(key):
            orphan_releases.append(item)
            continue
        acquisition = active[key].pop()
        if not active[key]:
            del active[key]
        lifetimes.append({
            "process_id": acquisition.get("process_id"),
            "handle": handle,
            "resource_type": acquisition.get("resource_type"),
            "acquisition_api": acquisition.get("api"),
            "release_api": item.get("api"),
            "acquisition_sequence": acquisition.get("sequence"),
            "release_sequence": item.get("sequence"),
            "acquisition_timestamp": acquisition.get("timestamp"),
            "release_timestamp": item.get("timestamp"),
            "acquisition_event_id": acquisition.get("event_id"),
            "release_event_id": item.get("event_id"),
        })

    still_active = [acquisition for resources in active.values() for acquisition in resources]
    return {
        "acquisitions": acquisitions,
        "releases": releases,
        "lifetimes": lifetimes,
        "orphan_releases": orphan_releases,
        "still_active": still_active,
    }
