import json
from collections import defaultdict

p = r"data\real_cape_report.json"

with open(p, encoding="utf-8") as f:
    d = json.load(f)

calls = d["behavior"]["processes"][0]["calls"]

acquisition_apis = {
    "NtOpenFile": ("file", "FileHandle"),
    "NtCreateFile": ("file", "FileHandle"),
    "NtOpenKey": ("registry", "KeyHandle"),
    "NtOpenKeyEx": ("registry", "KeyHandle"),
    "RegOpenKeyExA": ("registry", "Handle"),
    "RegOpenKeyExW": ("registry", "Handle"),
    "NtCreateSection": ("section", "SectionHandle"),
    "NtOpenSection": ("section", "SectionHandle"),
    "NtCreateMutant": ("mutant", "Handle"),
    "NtOpenEvent": ("event", "Handle"),
}

handles = defaultdict(list)

for i, c in enumerate(calls):
    api = c.get("api")

    if api not in acquisition_apis:
        continue

    if not c.get("status"):
        continue

    resource_type, arg_name = acquisition_apis[api]

    value = next(
        (a.get("value") for a in c.get("arguments", [])
         if a.get("name") == arg_name),
        None
    )

    if not value or value == "0x00000000":
        continue

    handles[value].append(
        (i, api, resource_type, c.get("timestamp"))
    )

print("=" * 80)
print("HANDLE REUSE ACROSS RESOURCE TYPES")
print("=" * 80)

found = 0

for handle, entries in handles.items():
    types = {x[2] for x in entries}

    if len(types) > 1:
        found += 1
        print(f"\nHANDLE: {handle}")

        for seq, api, rtype, ts in entries:
            print(
                f"  #{seq:<5} "
                f"{rtype:<10} "
                f"{api:<20} "
                f"{ts}"
            )

print("\nTotal handles reused across resource types:", found)
