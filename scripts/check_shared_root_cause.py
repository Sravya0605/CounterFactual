import csv
import pickle
import os
import numpy as np

from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.utils.graph_features import graph_to_api_counts

MODEL_PATH = 'models/full_dataset_agenttesla_vs_qbot_lgbm.pkl'
RESULTS_PATH = 'data/held_out_search_results.csv'
REPORTS_DIR = 'data/training_reports'

with open(MODEL_PATH, 'rb') as f:
    state = pickle.load(f)
model = state['model']
vocab = state['feature_vocab']

imp_gain = model.feature_importance(importance_type='gain')
order = np.argsort(-imp_gain)
top_features = [(vocab[i], imp_gain[i]) for i in order if imp_gain[i] > 0][:5]

print('Top features by gain (this 397-sample model):')
for name, gain in top_features:
    print(f'  {name!r} gain={gain:.2f}')
print('')

top_feature_name = top_features[0][0]

df = model.trees_to_dataframe()
splits_on_top = df[df['split_feature'] == top_feature_name]
thresholds = sorted(splits_on_top['threshold'].unique())
print(f'Distinct thresholds learned for top feature {top_feature_name!r}: {thresholds}')
print('')

with open(RESULTS_PATH, newline='', encoding='utf-8') as f:
    rows = [r for r in csv.DictReader(f) if r['status'] == 'no_flip_found']

print(f'Checking top feature value for all {len(rows)} no_flip_found samples...')
print(f'{"md5":<34} {"top_feat_value":>15} {"vs main threshold":>20}')

below_all_thresholds = 0
for r in rows:
    md5 = r['md5']
    path = os.path.join(REPORTS_DIR, f'{md5}.json')
    events = parse_cape_json(path)
    G = build_behavior_graph(events)
    counts = graph_to_api_counts(G)
    val = counts.get(top_feature_name, 0)
    main_thresh = thresholds[0] if thresholds else None
    side = 'BELOW (needs increase)' if main_thresh is not None and val <= main_thresh else 'above'
    if side.startswith('BELOW'):
        below_all_thresholds += 1
    print(f'{md5:<34} {val:>15} {side:>20}')

print('')
print(f'{below_all_thresholds}/{len(rows)} samples sit BELOW the main threshold (would need their count to INCREASE to flip)')