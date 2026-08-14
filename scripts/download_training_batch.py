import csv
import json
import os
import time
import urllib.request
import urllib.error

OUT_DIR = 'data/training_reports'
os.makedirs(OUT_DIR, exist_ok=True)

with open('data/training_batch.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

BASE_URL = 'https://huggingface.co/datasets/unileon-robotics/malware-samples/resolve/main/JSON/'
MAX_RETRIES = 3


def is_valid_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False


failed = []
for i, row in enumerate(rows, 1):
    md5 = row['md5']
    out_path = os.path.join(OUT_DIR, f'{md5}.json')

    if os.path.exists(out_path):
        if is_valid_json(out_path):
            print(f'[{i}/{len(rows)}] {md5} -- already downloaded, valid, skipping')
            continue
        else:
            print(f'[{i}/{len(rows)}] {md5} -- existing file is corrupted/truncated, re-downloading')
            os.remove(out_path)

    url = BASE_URL + md5 + '.json'
    success = False
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            urllib.request.urlretrieve(url, out_path)
            # Do not trust urlretrieve's silence -- verify the file actually parses.
            if not is_valid_json(out_path):
                raise urllib.error.ContentTooShortError('downloaded file failed JSON validation (likely silent truncation)', None)
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f'[{i}/{len(rows)}] {md5} -- {size_mb:.2f} MB, validated ({row["avclass_family"]})')
            success = True
            break
        except (urllib.error.HTTPError, urllib.error.ContentTooShortError, urllib.error.URLError) as exc:
            if os.path.exists(out_path):
                os.remove(out_path)
            print(f'[{i}/{len(rows)}] {md5} -- attempt {attempt}/{MAX_RETRIES} failed: {exc}')
            time.sleep(2)

    if not success:
        failed.append(md5)

print('')
print(f'Done. {len(rows) - len(failed)} succeeded, {len(failed)} failed.')
if failed:
    print('Failed hashes:', failed)