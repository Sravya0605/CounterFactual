"""Report API prevalence and first-occurrence positions for a saved run."""
import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.parser import parse_cape_json

TARGET_APIS = ("NtTestAlert", "GetSystemTimeAsFileTime")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="pipeline_results.json")
    parser.add_argument("--out", default="api_prevalence_oct10.json")
    args = parser.parse_args()

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    results_path = resolve(args.results)
    output_path = resolve(args.out)
    if output_path.exists():
        parser.error(f"Output already exists; choose a new --out: {output_path}")

    run = json.loads(results_path.read_text(encoding="utf-8"))
    training_files = run["run_config"]["training_files"]
    benign_dir = ROOT / Path(run["counterfactual_source"]).parent / "benign_reports"
    malware_dir = ROOT / run["counterfactual_source"]
    groups = {
        "benign": [],
        "malware": [],
    }
    for name in training_files:
        benign_path = benign_dir / name
        malware_path = malware_dir / name
        if benign_path.is_file() == malware_path.is_file():
            raise FileNotFoundError(
                f"Expected {name} in exactly one training class directory"
            )
        label = "benign" if benign_path.is_file() else "malware"
        groups[label].append(benign_path if label == "benign" else malware_path)

    report_presence = {
        label: Counter()
        for label in groups
    }
    first_positions = defaultdict(lambda: {"benign": [], "malware": []})

    for label, paths in groups.items():
        for report_index, path in enumerate(paths, 1):
            events = parse_cape_json(str(path))
            process_lengths = Counter(
                event.get("process_id") for event in events
            )
            first_event_position = {}
            for event in events:
                api = str(event.get("api") or "")
                normalized_api = api.casefold()
                report_presence[label][normalized_api] += 0
                process_id = event.get("process_id")
                length = process_lengths[process_id]
                sequence = event.get("sequence")
                if sequence is None:
                    raise ValueError(
                        f"Event has no sequence index in {path}"
                    )
                relative_position = int(sequence) / length
                first_event_position[normalized_api] = min(
                    relative_position,
                    first_event_position.get(normalized_api, 1.0),
                )
            for normalized_api, position in first_event_position.items():
                report_presence[label][normalized_api] += 1
                first_positions[normalized_api][label].append(position)
            if report_index % 50 == 0:
                print(
                    f"{label}: parsed {report_index}/{len(paths)} reports",
                    flush=True,
                )

    total_by_class = {label: len(paths) for label, paths in groups.items()}
    apis = set(report_presence["benign"]) | set(report_presence["malware"])
    rows = []
    for api in apis:
        benign_share = (
            report_presence["benign"][api] / total_by_class["benign"]
        )
        malware_share = (
            report_presence["malware"][api] / total_by_class["malware"]
        )
        positions = (
            first_positions[api]["benign"]
            + first_positions[api]["malware"]
        )
        rows.append(
            {
                "api": api,
                "benign_reports": report_presence["benign"][api],
                "benign_share": benign_share,
                "malware_reports": report_presence["malware"][api],
                "malware_share": malware_share,
                "median_first_relative_position": (
                    statistics.median(positions) if positions else None
                ),
            }
        )

    boilerplate = sorted(
        (
            row["api"]
            for row in rows
            if row["benign_share"] >= 0.5
            and row["malware_share"] >= 0.5
            and row["median_first_relative_position"] is not None
            and row["median_first_relative_position"] < 0.10
        ),
        key=str.casefold,
    )
    by_api = {row["api"].casefold(): row for row in rows}

    print("\nTARGET API METRICS")
    for api in TARGET_APIS:
        row = by_api.get(api.casefold())
        if row is None:
            print(f"{api}: absent from the training set")
            continue
        print(
            f"{row['api']}\t"
            f"benign={row['benign_reports']}/{total_by_class['benign']} "
            f"({row['benign_share']:.2%})\t"
            f"malware={row['malware_reports']}/{total_by_class['malware']} "
            f"({row['malware_share']:.2%})\t"
            f"median_first_relative_position="
            f"{row['median_first_relative_position']:.6f}"
        )

    print("\nALL API METRICS (report-level shares; position median across reports containing API)")
    print(
        "api\tbenign_reports\tbenign_share\tmalware_reports\t"
        "malware_share\tmedian_first_relative_position"
    )
    for row in sorted(rows, key=lambda item: item["api"].casefold()):
        print(
            f"{row['api']}\t{row['benign_reports']}\t"
            f"{row['benign_share']:.6f}\t{row['malware_reports']}\t"
            f"{row['malware_share']:.6f}\t"
            f"{row['median_first_relative_position']:.6f}"
        )

    print("\nBOILERPLATE CANDIDATES")
    print("\n".join(boilerplate) if boilerplate else "(none)")

    payload = {
        "results_file": str(results_path),
        "position_definition": (
            "zero-based event sequence / number of parsed events "
            "in that event's process; per report, use the earliest relative "
            "position across processes"
        ),
        "report_counts": total_by_class,
        "boilerplate_rule": (
            "present in at least 50% of both classes and median first "
            "relative position < 0.10"
        ),
        "boilerplate_candidates": boilerplate,
        "apis": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON written to {output_path}")


if __name__ == "__main__":
    main()
