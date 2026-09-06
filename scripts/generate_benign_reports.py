"""Generate parser-compatible synthetic benign CAPE reports for pipeline tests.

These reports are synthetic telemetry, not evidence of benign execution. Each
file-operation block preserves create/use/close order so the project's
resource-lifetime checks can inspect it without manufacturing invalid traces.
"""
import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

SYSTEM_DLLS = [
    "kernel32.dll", "ntdll.dll", "user32.dll", "gdi32.dll",
    "advapi32.dll", "ole32.dll", "shell32.dll", "combase.dll",
]
APP_LABELS = ["NotepadPlusPlus", "GoogleUpdate", "OfficeClickToRun", "AdobeAcrobat", "VSCode", "Slack", "Zoom", "Dropbox", "WinRAR", "SevenZip"]
USER_NAMES = ["jsmith", "asharma", "mchen", "kpatel", "rwilson", "tuser"]
REGKEYS = [
    r"\\Registry\\MACHINE\\Software\\Microsoft\\Windows NT\\CurrentVersion",
    r"\\Registry\\USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders",
]


def arg(name, value):
    return {"name": name, "value": value}


def hex_value(length=8):
    return "0x" + "".join(random.choice("0123456789abcdef") for _ in range(length))


def build_calls(app, user, target_calls):
    calls = []
    for dll in random.sample(SYSTEM_DLLS, 4):
        calls.append(("LdrLoadDll", [arg("ModuleName", dll)]))

    blocks = []
    for index in range(max(2, target_calls // 8)):
        path = f"C:\\Users\\{user}\\AppData\\Local\\{app}\\config_{index}.json"
        handle = hex_value(6)
        blocks.append([
            ("NtCreateFile", [arg("FileHandle", handle), arg("FileName", path)]),
            (random.choice(["NtReadFile", "NtWriteFile"]), [arg("FileHandle", handle)]),
            ("NtClose", [arg("Handle", handle)]),
        ])

    for index in range(random.randint(2, 5)):
        key = random.choice(REGKEYS).replace("{app}", app)
        handle = hex_value(6)
        blocks.append([
            ("NtOpenKey", [arg("KeyHandle", handle), arg("ObjectAttributesName", key)]),
            ("RegCloseKey", [arg("Handle", handle)]),
        ])

    random.shuffle(blocks)
    for block in blocks:
        calls.extend(block)

    while len(calls) < target_calls:
        path = f"C:\\Users\\{user}\\AppData\\Local\\{app}\\cache_{len(calls)}.dat"
        calls.append(("NtQueryAttributesFile", [arg("FileName", path)]))
    return calls[:target_calls]


def generate_report(seed):
    random.seed(seed)
    app = random.choice(APP_LABELS)
    user = random.choice(USER_NAMES)
    pid = random.randint(1000, 9000)
    base = datetime(2024, random.randint(1, 12), random.randint(1, 28), 12, 0, 0)
    raw_calls = build_calls(app, user, random.randint(30, 80))
    calls = []
    for index, (api, arguments) in enumerate(raw_calls):
        timestamp = base + timedelta(milliseconds=index * random.randint(5, 25))
        calls.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
            "api": api,
            "status": True,
            "arguments": arguments,
        })
    payload = {
        "target": {"category": "file", "file": {"name": f"{app.lower()}.exe"}},
        "behavior": {"processes": [{"pid": pid, "process_id": pid, "calls": calls}]},
        "signatures": [],
        "detections": "(n/a)",
        "avclass_detection": "None",
        "synthetic_metadata": {"benign": True, "generator": "generate_benign_reports.py", "seed": seed},
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return payload, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--out-dir", default="data/benign_reports")
    parser.add_argument("--csv-path", default="data/benign_batch.csv")
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(args.count):
        report, digest = generate_report(args.seed + index)
        filename = f"{digest}.json"
        (out_dir / filename).write_text(json.dumps(report, indent=2), encoding="utf-8")
        rows.append({"filename": filename, "md5": digest, "avclass_family": "benign", "cape_family": "benign"})

    with Path(args.csv_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic benign reports to {out_dir}")


if __name__ == "__main__":
    main()
