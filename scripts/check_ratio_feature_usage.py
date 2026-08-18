import pickle
import numpy as np

with open('models/full_dataset_agenttesla_vs_qbot_lgbm.pkl', 'rb') as f:
    state = pickle.load(f)

model = state['model']
vocab = state['feature_vocab']

ratio_features = [v for v in vocab if v.startswith('ratio_')]
target_in_vocab = 'ratio_createtoolhelp32snapshot' in vocab

print(f'Total vocab size: {len(vocab)}')
print(f'Number of ratio_ features in vocab: {len(ratio_features)}')
print(f'Is ratio_createtoolhelp32snapshot in vocab? {target_in_vocab}')

imp_gain = model.feature_importance(importance_type='gain')
nonzero_idx = np.where(imp_gain > 0)[0]
print(f'Features with nonzero gain: {len(nonzero_idx)}')
for i in nonzero_idx:
    print(f'  {vocab[i]!r}: gain={imp_gain[i]:.2f}')
    