"""Filter WinMET reports by target families and build training batch.

Works seamlessly with both:
1. Compressed .7z archives (e.g. WinMET_volume_1.7z) via selective on-the-fly extraction (no 120GB disk space needed)
2. Pre-extracted volume directories (data/raw_winmet/volume_*)
3. Reports already consolidated in data/training_reports/

Usage:
    # 1. Automatic top 2 families from available data:
    python scripts/build_winmet_batch.py --top-families 2 --max-per-family 60

    # 2. Specific family pair (e.g. Redline vs AgentTesla):
    python scripts/build_winmet_batch.py --family-a redline --family-b agenttesla --max-per-family 50
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw_winmet"
LABEL_FILE = RAW_DIR / "reports_consensus_label.json"
OUT_REPORTS_DIR = DATA_DIR / "training_reports"
OUT_CSV = DATA_DIR / "training_batch.csv"
OUT_WINMET_CSV = DATA_DIR / "training_batch_winmet.csv"
SEVENZIP_PASSWORD = "infected"

METADATA_KEYS = {
    "reports_avclass_no_consensus",
    "reports_cape_no_consensus",
    "reports_both_no_consensus",
}


def find_7z_binary() -> str:
    candidates = [
        Path("7zip/7z.exe"),
        Path("7z/7z.exe"),
        Path("C:/Program Files/7-Zip/7z.exe"),
        Path("C:/Program Files (x86)/7-Zip/7z.exe"),
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    which_7z = shutil.which("7z") or shutil.which("7z.exe")
    if which_7z:
        return which_7z
    return "7z"


def load_report_labels():
    if not LABEL_FILE.exists():
        raise FileNotFoundError(f"Label file not found at {LABEL_FILE}")
    with open(LABEL_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k not in METADATA_KEYS}


def get_available_sources():
    report_files = sorted(OUT_REPORTS_DIR.glob("*.json"))
    extracted_dirs = sorted(RAW_DIR.glob("volume_*"))
    archives = sorted(RAW_DIR.glob("*.7z"))
    return report_files, extracted_dirs, archives


def list_archive_json_names(archive_path: Path, seven_zip: str):
    cmd = [seven_zip, "l", "-slt", f"-p{SEVENZIP_PASSWORD}", str(archive_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    names = []
    for line in result.stdout.splitlines():
        if line.startswith("Path = "):
            val = line[len("Path = "):].strip()
            if val.endswith(".json") and not val.endswith(archive_path.name):
                names.append(val)
    return names


def main():
    parser = argparse.ArgumentParser(description="Build training batch from WinMET dataset.")
    parser.add_argument("--family-a", help="First family name (e.g. redline)")
    parser.add_argument("--family-b", help="Second family name (e.g. agenttesla)")
    parser.add_argument("--families", help="Comma-separated family list")
    parser.add_argument("--top-families", type=int, default=2, help="Auto-pick top N families (default: 2)")
    parser.add_argument("--max-per-family", type=int, default=None, help="Maximum samples per family (default: unlimited)")
    args = parser.parse_args()

    report_labels = load_report_labels()
    report_files, extracted_dirs, archives = get_available_sources()

    if not report_files and not extracted_dirs and not archives:
        print(f"ERROR: No reports, volume directories, or .7z archives found in {RAW_DIR} or {OUT_REPORTS_DIR}")
        return

    seven_zip = find_7z_binary()
    available_hashes = {}  # hash -> (source_type, source_path, arcname)

    # 1. Index reports already consolidated in the training directory
    for jpath in report_files:
        h = jpath.stem
        available_hashes[h] = ("file", jpath, None)

    # 2. Index extracted directories
    for vdir in extracted_dirs:
        for jpath in vdir.glob("*.json"):
            h = jpath.stem
            if h not in available_hashes:
                available_hashes[h] = ("dir", jpath, None)

    # 3. Index .7z archives
    for arc in archives:
        print(f"Indexing archive: {arc.name} ...")
        arc_names = list_archive_json_names(arc, seven_zip)
        for aname in arc_names:
            h = Path(aname).name.replace(".json", "")
            if h not in available_hashes:
                available_hashes[h] = ("7z", arc, aname)

    print(f"Total reports available locally across sources: {len(available_hashes)}")

    # Compute family counts among locally available reports
    fam_counts = Counter()
    for h in available_hashes:
        info = report_labels.get(h)
        if info and info.get("avclass"):
            fam_counts[info["avclass"].strip().lower()] += 1

    print("\nTop families present in your local WinMET data:")
    for fam, c in fam_counts.most_common(15):
        print(f"  {fam:25s}: {c:5d} samples")

    # Determine target families
    target_families = set()
    if args.family_a or args.family_b or args.families:
        if args.families:
            target_families = {f.strip().lower() for f in args.families.split(",") if f.strip()}
        else:
            if args.family_a:
                target_families.add(args.family_a.strip().lower())
            if args.family_b:
                target_families.add(args.family_b.strip().lower())
    else:
        top_fams = [f for f, _ in fam_counts.most_common(args.top_families)]
        target_families = set(top_fams)

    print(f"\nTarget families for batch: {target_families}")

    # Select samples up to max_per_family
    selected_by_fam = {fam: [] for fam in target_families}
    for h, src_info in available_hashes.items():
        info = report_labels.get(h)
        if not info:
            continue
        fam = str(info.get("avclass", "")).strip().lower()
        if fam in target_families and (args.max_per_family is None or len(selected_by_fam[fam]) < args.max_per_family):
            selected_by_fam[fam].append((h, src_info, info))

    OUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    archive_extract_lists = {}  # arc_path -> [arcnames]

    for fam, items in selected_by_fam.items():
        limit = args.max_per_family if args.max_per_family is not None else "unlimited"
        print(f"  {fam:25s}: selecting {len(items)} samples (max {limit})")
        for h, (stype, spath, aname), info in items:
            dest_file = OUT_REPORTS_DIR / f"{h}.json"
            if stype == "file":
                pass
            elif stype == "dir":
                if not dest_file.exists():
                    shutil.copy2(spath, dest_file)
            elif stype == "7z":
                if not dest_file.exists():
                    archive_extract_lists.setdefault(spath, []).append(aname)

            csv_rows.append({
                "filename": f"{h}.json",
                "md5": h,
                "avclass_family": info.get("avclass", fam),
                "cape_family": info.get("cape", ""),
            })

    # Perform selective 7z extractions if needed
    for arc_path, anames in archive_extract_lists.items():
        print(f"\nSelectively extracting {len(anames)} reports from {arc_path.name} into {OUT_REPORTS_DIR} ...")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
            for name in anames:
                tf.write(name + "\n")
            flist = tf.name

        cmd = [seven_zip, "e", f"-p{SEVENZIP_PASSWORD}", str(arc_path), f"@{flist}", f"-o{OUT_REPORTS_DIR}", "-y"]
        subprocess.run(cmd, check=True)
        try:
            os.remove(flist)
        except OSError:
            pass

    for out_csv_path in (OUT_CSV, OUT_WINMET_CSV):
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "md5", "avclass_family", "cape_family"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Batch metadata written to: {out_csv_path}")

    print(f"\nBatch preparation complete: {len(csv_rows)} total samples ready in {OUT_REPORTS_DIR}")
    print("\nRun training and evaluation with:")
    print("  python -m scripts.train_full_dataset_classifier")
    print("  python -m scripts.evaluate_dataset")


if __name__ == "__main__":
    main()