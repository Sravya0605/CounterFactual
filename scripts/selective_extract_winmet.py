"""Selectively extract only the WinMET reports belonging to target families
from a NOT-YET-EXTRACTED .7z volume, without decompressing the full archive.

Features:
- Automatically discovers 7z.exe in 7zip/, 7z/, or PATH
- Inspects and lists all families present in the specific volume (--list-families)
- Supports extracting top N families or specific named families
- Generates the matching batch CSV ready for training and evaluation

Usage:
    # 1. List what families are inside your downloaded volume:
    python scripts/selective_extract_winmet.py --volume data/raw_winmet/WinMET_volume_1.7z --list-families

    # 2. Extract top 2 most populous families in the volume:
    python scripts/selective_extract_winmet.py --volume data/raw_winmet/WinMET_volume_1.7z --top-families 2

    # 3. Extract specific families (e.g. dacic & padodor, or agenttesla & qbot):
    python scripts/selective_extract_winmet.py --volume data/raw_winmet/WinMET_volume_1.7z --family-a dacic --family-b padodor --max-per-family 50
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

LABEL_FILE = Path("data/raw_winmet/reports_consensus_label.json")
OUT_REPORTS_DIR = Path("data/training_reports")
OUT_CSV = Path("data/training_batch.csv")
OUT_WINMET_CSV = Path("data/training_batch_winmet.csv")
SEVENZIP_PASSWORD = "infected"

METADATA_KEYS = {
    "reports_avclass_no_consensus",
    "reports_cape_no_consensus",
    "reports_both_no_consensus",
}


def find_7z_binary() -> str:
    """Find 7z executable in local repository directories or system PATH."""
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


def list_archive_contents(volume_path: Path, seven_zip: str):
    """List filenames inside the .7z archive without decompressing."""
    cmd = [seven_zip, "l", "-slt", f"-p{SEVENZIP_PASSWORD}", str(volume_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    names = []
    for line in result.stdout.splitlines():
        if line.startswith("Path = "):
            val = line[len("Path = "):].strip()
            if val.endswith(".json") and not val.endswith(volume_path.name):
                names.append(val)
    return names


def main():
    parser = argparse.ArgumentParser(description="Selective WinMET report extraction without full decompression.")
    parser.add_argument("--volume", default="data/raw_winmet/WinMET_volume_1.7z", help="Path to .7z volume")
    parser.add_argument("--family-a", help="First family name (e.g. dacic)")
    parser.add_argument("--family-b", help="Second family name (e.g. padodor)")
    parser.add_argument("--families", help="Comma-separated family list (e.g. emotet,agenttesla,qbot)")
    parser.add_argument("--top-families", type=int, help="Automatically pick top N most frequent families in this volume")
    parser.add_argument("--max-per-family", type=int, default=None, help="Maximum samples per family (default: unlimited)")
    parser.add_argument("--list-families", action="store_true", help="List family distributions in this volume and exit")
    args = parser.parse_args()

    volume_path = Path(args.volume)
    if not volume_path.exists():
        print(f"ERROR: Volume archive {volume_path} does not exist.")
        return

    seven_zip = find_7z_binary()
    print(f"Using 7-Zip binary: {seven_zip}")
    print(f"Reading index from {volume_path} ...")

    all_names = list_archive_contents(volume_path, seven_zip)
    print(f"Total reports indexed in this volume: {len(all_names)}")

    report_labels = load_report_labels()
    hash_to_arcname = {Path(n).name.replace(".json", ""): n for n in all_names}

    # Count families present in this volume
    fam_in_vol = Counter()
    for h in hash_to_arcname:
        info = report_labels.get(h)
        if info and info.get("avclass"):
            fam_in_vol[info["avclass"].strip().lower()] += 1

    if args.list_families or (not args.family_a and not args.family_b and not args.families and not args.top_families):
        print("\n--- Available Families in this Volume (AVClass) ---")
        for fam, count in fam_in_vol.most_common(30):
            print(f"  {fam:25s}: {count:5d} samples")
        if not (args.family_a or args.family_b or args.families or args.top_families):
            print("\nTip: Run with --top-families 2 or --family-a <name> --family-b <name> to extract.")
            return

    # Determine target families
    target_families = set()
    if args.top_families:
        top_fams = [f for f, _ in fam_in_vol.most_common(args.top_families)]
        target_families = set(top_fams)
        print(f"\nAuto-selected top {args.top_families} families: {', '.join(top_fams)}")
    elif args.families:
        target_families = {f.strip().lower() for f in args.families.split(",") if f.strip()}
    else:
        if args.family_a:
            target_families.add(args.family_a.strip().lower())
        if args.family_b:
            target_families.add(args.family_b.strip().lower())

    print(f"Target families: {target_families}")

    # Select samples per family
    selected_by_fam = {fam: [] for fam in target_families}
    for h, arcname in hash_to_arcname.items():
        info = report_labels.get(h)
        if not info:
            continue
        fam = str(info.get("avclass", "")).strip().lower()
        if fam in target_families and (args.max_per_family is None or len(selected_by_fam[fam]) < args.max_per_family):
            selected_by_fam[fam].append((h, arcname, info))

    total_selected = sum(len(v) for v in selected_by_fam.values())
    print("\nSelected samples for extraction:")
    for fam, items in selected_by_fam.items():
        print(f"  {fam:25s}: {len(items):4d} / {fam_in_vol.get(fam, 0)} available")

    if total_selected == 0:
        print("ERROR: No matching samples found for the selected families in this volume.")
        return

    OUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Write filelist for 7z extraction
    all_matched_arcnames = []
    csv_rows = []
    for fam, items in selected_by_fam.items():
        for h, arcname, info in items:
            all_matched_arcnames.append(arcname)
            csv_rows.append({
                "filename": f"{h}.json",
                "md5": h,
                "avclass_family": info.get("avclass", fam),
                "cape_family": info.get("cape", ""),
            })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        for name in all_matched_arcnames:
            tf.write(name + "\n")
        filelist_path = tf.name

    print(f"\nExtracting {len(all_matched_arcnames)} JSON reports directly into {OUT_REPORTS_DIR} ...")
    cmd_extract = [
        seven_zip, "e", f"-p{SEVENZIP_PASSWORD}", str(volume_path),
        f"@{filelist_path}", f"-o{OUT_REPORTS_DIR}", "-y"
    ]
    subprocess.run(cmd_extract, check=True)

    try:
        os.remove(filelist_path)
    except OSError:
        pass

    # Write CSVs
    for out_csv_path in (OUT_CSV, OUT_WINMET_CSV):
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "md5", "avclass_family", "cape_family"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Batch metadata written to: {out_csv_path}")

    print("\nSuccess! Selective extraction complete.")
    print("Next step: run training and evaluation with:")
    print("  python -m scripts.train_full_dataset_classifier")
    print("  python -m scripts.evaluate_dataset")


if __name__ == "__main__":
    main()