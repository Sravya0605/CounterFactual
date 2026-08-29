import pickle
import os

from src.ingestion.parser import parse_cape_json
from src.graph.graph_builder import build_behavior_graph
from src.utils.graph_features import graph_to_api_counts

MD5 = '83a3340793cec4cb51526c5f4d6711b9'
MODEL_PATH = 'models/full_dataset_agenttesla_vs_qbot_lgbm.pkl'

with open(MODEL_PATH, 'rb') as f:
    state = pickle.load(f)
model = state['model']

df = model.trees_to_dataframe()
splits_on_findresourceexa = df[df['split_feature'] == 'findresourceexa']
thresholds = sorted(splits_on_findresourceexa['threshold'].unique())
print(f'Number of splits on findresourceexa: {len(splits_on_findresourceexa)}')
print(f'Distinct thresholds learned for findresourceexa: {thresholds}')
print('')

path = os.path.join('data/training_reports', f'{MD5}.json')
events = parse_cape_json(path)
G = build_behavior_graph(events)
counts = graph_to_api_counts(G)

ct_val = counts.get('createtoolhelp32snapshot', 0)
fr_val = counts.get('findresourceexa', 0)
print(f'Sample {MD5}:')
print(f'  createtoolhelp32snapshot = {ct_val} (threshold ~3.0, needs >3 to flip toward qbot)')
print(f'  findresourceexa = {fr_val} (threshold(s) shown above)')