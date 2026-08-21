from collections import Counter

from src.ingestion.parser import parse_cape_json
from scripts.match_resource_lifetimes import match_resource_lifetimes


def main():
    events = parse_cape_json(r"data\real_cape_report.json")
    result = match_resource_lifetimes(events)

    lifetimes = result["lifetimes"]

    print("=" * 70)
    print("RESOURCE LIFETIME STATISTICS")
    print("=" * 70)

    print(f"Raw events:             {len(events)}")
    print(f"Acquisitions:           {len(result['acquisitions'])}")
    print(f"Releases:               {len(result['releases'])}")
    print(f"Matched lifetimes:      {len(lifetimes)}")
    print(f"Orphan releases:        {len(result['orphan_releases'])}")
    print(f"Still-active resources: {len(result['still_active'])}")

    print()
    print("Lifetimes by resource type:")
    print("-" * 50)

    type_counts = Counter(
        item["resource_type"]
        for item in lifetimes
    )

    for resource_type, count in type_counts.most_common():
        print(f"{resource_type:12} {count}")

    print()
    print("Lifetimes by acquisition API:")
    print("-" * 50)

    api_counts = Counter(
        item["acquisition_api"]
        for item in lifetimes
    )

    for api, count in api_counts.most_common():
        print(f"{api:24} {count}")

    print()
    print("Lifetimes by release API:")
    print("-" * 50)

    release_counts = Counter(
        item["release_api"]
        for item in lifetimes
    )

    for api, count in release_counts.most_common():
        print(f"{api:24} {count}")

    print()
    print("Lifetime duration statistics:")
    print("-" * 50)

    print(
        "Note: CAPE timestamps have millisecond precision, "
        "but many events share the same timestamp."
    )

    durations = []

    from datetime import datetime

    for item in lifetimes:
        start = datetime.strptime(
            item["acquisition_timestamp"],
            "%Y-%m-%d %H:%M:%S,%f",
        )

        end = datetime.strptime(
            item["release_timestamp"],
            "%Y-%m-%d %H:%M:%S,%f",
        )

        duration_ms = (end - start).total_seconds() * 1000

        if duration_ms >= 0:
            durations.append(duration_ms)

    if durations:
        print(f"Count:   {len(durations)}")
        print(f"Min ms:  {min(durations):.3f}")
        print(f"Max ms:  {max(durations):.3f}")
        print(f"Mean ms: {sum(durations) / len(durations):.3f}")

        zero = sum(1 for x in durations if x == 0)
        print(f"Zero-duration: {zero}")

    print()
    print("Longest 20 lifetimes:")
    print("-" * 100)

    enriched = []

    for item in lifetimes:
        start = datetime.strptime(
            item["acquisition_timestamp"],
            "%Y-%m-%d %H:%M:%S,%f",
        )

        end = datetime.strptime(
            item["release_timestamp"],
            "%Y-%m-%d %H:%M:%S,%f",
        )

        duration_ms = (end - start).total_seconds() * 1000

        enriched.append((duration_ms, item))

    enriched.sort(key=lambda x: x[0], reverse=True)

    for duration_ms, item in enriched[:20]:
        print(
            f"{duration_ms:10.3f} ms | "
            f"{item['resource_type']:10} | "
            f"{item['handle']} | "
            f"{item['acquisition_api']} "
            f"#{item['acquisition_sequence']} -> "
            f"{item['release_api']} "
            f"#{item['release_sequence']}"
        )


if __name__ == "__main__":
    main()