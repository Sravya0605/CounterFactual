import pandas as pd

X = pd.read_csv('data/feature_matrix.csv')
labels_df = pd.read_csv('data/labels.csv')

for feat in ['createprocessw', 'createthread', 'cryptencrypt', 'connectex']:
    if feat not in X.columns:
        print(f'{feat}: NOT IN VOCAB')
        continue
    present = (X[feat] > 0)
    print(f'\n--- {feat} ---')
    for fam in ['emotet', 'agenttesla', 'qbot']:
        mask = labels_df['label'] == fam
        pct_present = present[mask].mean() * 100
        print(f'  {fam}: present in {pct_present:.1f}% of samples')