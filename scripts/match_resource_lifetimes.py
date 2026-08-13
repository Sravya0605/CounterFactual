import sys
from collections import defaultdict

from src.ingestion.parser import parse_cape_json
from src.behavior.resources import extract_acquisition, extract_release


def match_resource_lifetimes(events):
    acquisitions = []
    releases = []

    for event in events:
        acquisition = extract_acquisition(event)
        if acquisition is not None:
            acquisitions.append(acquisition)

        release = extract_release(event)
        if release is not None:
            releases.append(release)

    # Active resource instances.
    #
    # Key:
    #     (process_id, handle)
    #
    # Value:
    #     list of acquisitions currently active for that handle.
    #
    # A list is required because the same handle can be reused.
    active = defaultdict(list)

    lifetimes = []
    orphan_releases = []

    # Process everything in original CAPE sequence order.
    timeline = []

    for acquisition in acquisitions:
        timeline.append(
            (
                acquisition.get("sequence", 0),
                0,  # acquisition before release at same sequence
                "acquisition",
                acquisition,
            )
        )

    for release in releases:
        timeline.append(
            (
                release.get("sequence", 0),
                1,
                "release",
                release,
            )
        )

    timeline.sort(key=lambda x: (x[0], x[1]))

    for _, _, event_type, item in timeline:

        if event_type == "acquisition":
            handle = item.get("handle")

            # Handle-less resources cannot participate in handle lifetime
            # matching.
            if handle is None:
                continue

            key = (
                item.get("process_id"),
                handle,
            )

            active[key].append(item)

        else:
            handle = item.get("handle")

            if handle is None:
                continue

            key = (
                item.get("process_id"),
                handle,
            )

            candidates = active.get(key)

            if not candidates:
                orphan_releases.append(item)
                continue

            # The most recent active acquisition is the correct candidate
            # when a handle value is reused.
            acquisition = candidates.pop()

            if not candidates:
                del active[key]

            lifetimes.append(
                {
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
                }
            )

    still_active = []

    for key, resources in active.items():
        for acquisition in resources:
            still_active.append(acquisition)

    return {
        "acquisitions": acquisitions,
        "releases": releases,
        "lifetimes": lifetimes,
        "orphan_releases": orphan_releases,
        "still_active": still_active,
    }


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python scripts/match_resource_lifetimes.py "
            "<cape_report.json>"
        )
        sys.exit(1)

    events = parse_cape_json(sys.argv[1])
    result = match_resource_lifetimes(events)

    print("=" * 70)
    print("SEQUENCE-AWARE RESOURCE LIFETIME MATCHING")
    print("=" * 70)

    print(f"Raw events:             {len(events)}")
    print(f"Acquisitions:           {len(result['acquisitions'])}")
    print(f"Releases:               {len(result['releases'])}")
    print(f"Matched lifetimes:      {len(result['lifetimes'])}")
    print(f"Orphan releases:        {len(result['orphan_releases'])}")
    print(f"Still-active resources: {len(result['still_active'])}")

    print()
    print("First 20 matched lifetimes:")
    print("-" * 100)

    for item in result["lifetimes"][:20]:
        print(
            f"{item['handle']} | "
            f"{item['resource_type']} | "
            f"{item['acquisition_api']} #{item['acquisition_sequence']} "
            f"-> {item['release_api']} #{item['release_sequence']} | "
            f"{item['acquisition_timestamp']} -> "
            f"{item['release_timestamp']}"
        )

    print()
    print("First 20 orphan releases:")
    print("-" * 100)

    for item in result["orphan_releases"][:20]:
        print(
            f"{item['handle']} | "
            f"{item['api']} | "
            f"sequence={item['sequence']} | "
            f"timestamp={item['timestamp']}"
        )

    print()
    print("Still-active resources:")
    print("-" * 100)

    for item in result["still_active"][:20]:
        print(
            f"{item['handle']} | "
            f"{item['resource_type']} | "
            f"{item['api']} #{item['sequence']} | "
            f"{item['timestamp']}"
        )


if __name__ == "__main__":
    main()