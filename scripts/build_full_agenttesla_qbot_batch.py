import csv

TARGET_FAMILIES = ['agenttesla', 'qbot']

with open('malware_families.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

selected = [r for r in rows if r['avclass_family'].strip().lower() in TARGET_FAMILIES]

for fam in TARGET_FAMILIES:
    count = sum(1 for r in selected if r['avclass_family'].strip().lower() == fam)
    print(f'{fam}: {count} samples selected')

with open('data/training_batch_full.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['filename', 'md5', 'avclass_family', 'cape_family'])
    writer.writeheader()
    writer.writerows(selected)

print('')
print(f'Total selected: {len(selected)}')
print('Written to data/training_batch_full.csv')