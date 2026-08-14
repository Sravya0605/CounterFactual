import csv
import random

random.seed(42)  # reproducible sample selection -- log this seed in docs/decisions.md

TARGET_FAMILIES = ['emotet', 'agenttesla', 'qbot']
PER_FAMILY = 30

with open('malware_families.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

selected = []
for fam in TARGET_FAMILIES:
    matches = [r for r in rows if r['avclass_family'].strip().lower() == fam]
    random.shuffle(matches)
    chosen = matches[:PER_FAMILY]
    selected.extend(chosen)
    print(f'{fam}: selected {len(chosen)} of {len(matches)} available')

with open('data/training_batch.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['filename', 'md5', 'avclass_family', 'cape_family'])
    writer.writeheader()
    writer.writerows(selected)

print('')
print(f'Total selected: {len(selected)}')
print('Written to data/training_batch.csv')