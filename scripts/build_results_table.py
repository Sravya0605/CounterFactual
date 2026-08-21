import csv
import statistics

RESULTS_PATH = 'data/held_out_search_results.csv'

with open(RESULTS_PATH, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f'Total held-out samples: {len(rows)}')
print('')

by_status = {}
for row in rows:
    by_status.setdefault(row['status'], []).append(row)

for status, group in sorted(by_status.items()):
    print(f'{status}: {len(group)}')

print('')

malicious = [r for r in rows if r['true_label'] == '1']
not_malicious = [r for r in rows if r['true_label'] == '0']

print(f'True malicious (agenttesla) samples: {len(malicious)}')
print(f'True not-malicious (qbot) samples: {len(not_malicious)}')
print('')

# Classification accuracy check: not_malicious status should match true_label=0,
# and (no_flip_found OR completed) should match true_label=1 -- i.e. verify the
# classifier's own accuracy on this held-out set via these search results.
correctly_classified_malicious = sum(1 for r in malicious if r['status'] in ('no_flip_found', 'completed'))
correctly_classified_benign = sum(1 for r in not_malicious if r['status'] == 'not_malicious')
print(f'Classifier accuracy check: {correctly_classified_malicious}/{len(malicious)} malicious correctly flagged, {correctly_classified_benign}/{len(not_malicious)} qbot correctly not-flagged')
print('')

completed = [r for r in malicious if r['status'] == 'completed']
no_flip = [r for r in malicious if r['status'] == 'no_flip_found']

feasibility_rate = len(completed) / len(malicious) if malicious else 0
print(f'FEASIBILITY RATE: {len(completed)}/{len(malicious)} = {feasibility_rate:.1%}')
print('')

if completed:
    costs = [float(r['candidate_cost']) for r in completed]
    print(f'Cost of successful flips: min={min(costs)}, max={max(costs)}, mean={statistics.mean(costs):.2f}')
else:
    print('No successful flips -- no cost distribution to report.')

elapsed = [float(r['elapsed_seconds']) for r in rows]
print('')
print(f'Total search runtime: {sum(elapsed)/3600:.2f} hours')
print(f'Per-sample time -- min={min(elapsed):.1f}s, max={max(elapsed):.1f}s, mean={statistics.mean(elapsed):.1f}s, median={statistics.median(elapsed):.1f}s')

print('')
print('--- Per-sample table (malicious samples only) ---')
print(f'{"md5":<34} {"status":<15} {"orig_prob":>10} {"cost":>6} {"time(s)":>8}')
for r in malicious:
    print(f'{r["md5"]:<34} {r["status"]:<15} {float(r["orig_prob"]):>10.4f} {r["candidate_cost"] or "-":>6} {float(r["elapsed_seconds"]):>8.1f}')