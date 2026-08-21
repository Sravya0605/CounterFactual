import json
from collections import defaultdict


REPORT = r"data\real_cape_report.json"


# APIs that acquire a resource and the argument containing its handle.
ACQUIRE_APIS = {
    "NtOpenFile": ("FileHandle", "file"),
    "NtCreateFile": ("FileHandle", "file"),
    "NtCreateMutant": ("Handle", "mutant"),
    "RegOpenKeyExA": ("Handle", "registry"),
    "RegOpenKeyExW": ("Handle", "registry"),
}

# APIs that release resources.
RELEASE_APIS = {
    "NtClose": ("Handle", None),
    "NtReleaseMutant": ("Handle", "mutant"),
    "RegCloseKey": ("Handle", "registry"),
}


def get_argument(call, name):
    for arg in call.get("arguments", []):
        if arg.get("name") == name:
            return arg.get("value")
    return None


def main():
    with open(REPORT, "r", encoding="utf-8") as f:
        report = json.load(f)

    processes = report.get("behavior", {}).get("processes", [])

    acquisitions = []
    releases = []

    for process in processes:
        process_id = process.get("process_id")

        for sequence, call in enumerate(process.get("calls", [])):
            api = call.get("api")

            if api in ACQUIRE_APIS:
                arg_name, resource_type = ACQUIRE_APIS[api]
                handle = get_argument(call, arg_name)

                if handle:
                    acquisitions.append({
                        "sequence": sequence,
                        "call_id": call.get("id"),
                        "timestamp": call.get("timestamp"),
                        "process_id": process_id,
                        "handle": handle,
                        "api": api,
                        "type": resource_type,
                    })

            elif api in RELEASE_APIS:
                arg_name, expected_type = RELEASE_APIS[api]
                handle = get_argument(call, arg_name)

                if handle:
                    releases.append({
                        "sequence": sequence,
                        "call_id": call.get("id"),
                        "timestamp": call.get("timestamp"),
                        "process_id": process_id,
                        "handle": handle,
                        "api": api,
                        "expected_type": expected_type,
                    })

    # Active resource instances indexed by (process, handle).
    active = defaultdict(list)

    matched = []
    orphan_releases = []

    # Process calls in their actual CAPE sequence.
    all_calls = []

    for process in processes:
        process_id = process.get("process_id")

        for sequence, call in enumerate(process.get("calls", [])):
            all_calls.append((process_id, sequence, call))

    all_calls.sort(key=lambda x: (x[0], x[1]))

    for process_id, sequence, call in all_calls:
        api = call.get("api")

        # Acquisition
        if api in ACQUIRE_APIS:
            arg_name, resource_type = ACQUIRE_APIS[api]
            handle = get_argument(call, arg_name)

            if handle:
                active[(process_id, handle)].append({
                    "acquire_sequence": sequence,
                    "acquire_call_id": call.get("id"),
                    "acquire_timestamp": call.get("timestamp"),
                    "acquire_api": api,
                    "type": resource_type,
                })

        # Release
        elif api in RELEASE_APIS:
            arg_name, expected_type = RELEASE_APIS[api]
            handle = get_argument(call, arg_name)

            if not handle:
                continue

            key = (process_id, handle)

            if not active[key]:
                orphan_releases.append({
                    "sequence": sequence,
                    "call_id": call.get("id"),
                    "timestamp": call.get("timestamp"),
                    "process_id": process_id,
                    "handle": handle,
                    "api": api,
                })
                continue

            # Handle reuse means we must release the most recent
            # currently-active instance of this handle.
            resource = active[key].pop()

            type_mismatch = (
                expected_type is not None
                and resource["type"] != expected_type
            )

            matched.append({
                **resource,
                "handle": handle,
                "release_sequence": sequence,
                "release_call_id": call.get("id"),
                "release_timestamp": call.get("timestamp"),
                "release_api": api,
                "type_mismatch": type_mismatch,
            })

    still_active = []

    for (process_id, handle), resources in active.items():
        for resource in resources:
            still_active.append({
                **resource,
                "process_id": process_id,
                "handle": handle,
            })

    print("=" * 70)
    print("SEQUENCE-AWARE CAPE RESOURCE LIFETIME ANALYSIS")
    print("=" * 70)

    print()
    print(f"Processes:              {len(processes)}")
    print(f"Acquisition instances:  {len(acquisitions)}")
    print(f"Release events:          {len(releases)}")
    print(f"Matched lifetimes:       {len(matched)}")
    print(f"Orphan releases:         {len(orphan_releases)}")
    print(f"Still-active resources:  {len(still_active)}")

    mismatches = [
        x for x in matched
        if x["type_mismatch"]
    ]

    print(f"Type mismatches:         {len(mismatches)}")

    if matched:
        print()
        print("First 20 reconstructed lifetimes:")
        print("-" * 100)

        for item in matched[:20]:
            print(
                f"{item['handle']} | "
                f"{item['type']} | "
                f"{item['acquire_api']} "
                f"#{item['acquire_sequence']} -> "
                f"{item['release_api']} "
                f"#{item['release_sequence']} | "
                f"{item['acquire_timestamp']} -> "
                f"{item['release_timestamp']}"
            )

    if orphan_releases:
        print()
        print("First 10 orphan releases:")
        print("-" * 100)

        for item in orphan_releases[:10]:
            print(
                f"{item['handle']} | "
                f"{item['api']} | "
                f"sequence={item['sequence']} | "
                f"timestamp={item['timestamp']}"
            )

    if still_active:
        print()
        print("First 20 still-active resources:")
        print("-" * 100)

        for item in still_active[:20]:
            print(
                f"{item['handle']} | "
                f"{item['type']} | "
                f"{item['acquire_api']} | "
                f"sequence={item['acquire_sequence']} | "
                f"timestamp={item['acquire_timestamp']}"
            )


if __name__ == "__main__":
    main()