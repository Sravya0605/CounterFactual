import json

p = r"data\real_cape_report.json"

with open(p, encoding="utf-8") as f:
    d = json.load(f)

calls = d["behavior"]["processes"][0]["calls"]

apis = [
    "NtOpenKey",
    "NtOpenKeyEx",
    "RegOpenKeyExA",
    "RegOpenKeyExW",
    "NtCreateSection",
    "NtOpenSection",
    "NtCreateMutant",
    "NtOpenEvent",
]

for api in apis:
    print(f"\n=== {api} ===")

    matches = [c for c in calls if c.get("api") == api]

    print("Count:", len(matches))

    for i, c in enumerate(matches[:3]):
        print(f"\n[{i}]")
        print("Timestamp:", c.get("timestamp"))
        print("Return:", c.get("return"))
        print("Status:", c.get("status"))
        print("Arguments:")

        for arg in c.get("arguments", []):
            print("  ", arg)
