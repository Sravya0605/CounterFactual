import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

X = pd.read_csv('data/feature_matrix.csv')
labels_df = pd.read_csv('data/labels.csv')
y_raw = labels_df['label']

le = LabelEncoder()
y = le.fit_transform(y_raw)
print('Classes:', list(le.classes_))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f'Train size: {len(X_train)}, Test size: {len(X_test)}')

params = {
    'objective': 'multiclass',
    'num_class': len(le.classes_),
    'metric': 'multi_logloss',
    'verbosity': -1,
}
train_data = lgb.Dataset(X_train, label=y_train)
model = lgb.train(params, train_data, num_boost_round=100)

# Sanity-check the model isn't degenerate before trusting anything else
print('')
print('--- Model sanity check ---')
tree_info = model.dump_model()['tree_info']
print(f'Number of trees: {len(tree_info)}')
leaf_counts = [t['num_leaves'] for t in tree_info]
print(f'Leaves per tree: min={min(leaf_counts)}, max={max(leaf_counts)}, mean={sum(leaf_counts)/len(leaf_counts):.1f}')

importance = model.feature_importance(importance_type='gain')
nonzero = sum(1 for v in importance if v > 0)
print(f'Features with nonzero gain: {nonzero} / {len(importance)}')

top_idx = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)[:15]
print('')
print('Top 15 features by gain:')
for i in top_idx:
    print(f'  {X.columns[i]:40s} {importance[i]:.1f}')

# Real held-out evaluation
print('')
print('--- Held-out test set evaluation ---')
preds = model.predict(X_test)
pred_labels = preds.argmax(axis=1)
print(classification_report(y_test, pred_labels, target_names=le.classes_))
print('Confusion matrix:')
print(confusion_matrix(y_test, pred_labels))

import pickle
with open('models/family_classifier_lgbm.pkl', 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': le, 'feature_columns': list(X.columns)}, f)
print('')
print('Saved models/family_classifier_lgbm.pkl')

# --- Cross-validation: does the near-perfect separation survive different splits? ---
from sklearn.model_selection import StratifiedKFold
import numpy as np

print('')
print('--- 5-fold cross-validation on original (non-normalized) features ---')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []
for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    fold_data = lgb.Dataset(X_tr, label=y_tr)
    fold_model = lgb.train(params, fold_data, num_boost_round=100)
    fold_preds = fold_model.predict(X_te).argmax(axis=1)
    acc = (fold_preds == y_te).mean()
    fold_accuracies.append(acc)
    print(f'  Fold {fold_i}: accuracy = {acc:.3f} (test size {len(test_idx)})')

print(f'Mean CV accuracy: {np.mean(fold_accuracies):.3f} (+/- {np.std(fold_accuracies):.3f})')

# --- Shortcut-learning check: does normalizing by graph size collapse accuracy? ---
print('')
print('--- Shortcut-learning control: size-normalized features ---')
graph_sizes = X.sum(axis=1)  # total feature count per sample, proxy for graph/trace size
X_normalized = X.div(graph_sizes.replace(0, 1), axis=0)

X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(
    X_normalized, y, test_size=0.3, random_state=42, stratify=y
)
train_data_n = lgb.Dataset(X_train_n, label=y_train_n)
model_n = lgb.train(params, train_data_n, num_boost_round=100)

preds_n = model_n.predict(X_test_n)
pred_labels_n = preds_n.argmax(axis=1)
print(classification_report(y_test_n, pred_labels_n, target_names=le.classes_))

importance_n = model_n.feature_importance(importance_type='gain')
nonzero_n = sum(1 for v in importance_n if v > 0)
print(f'Features with nonzero gain (normalized): {nonzero_n} / {len(importance_n)}')
top_idx_n = sorted(range(len(importance_n)), key=lambda i: importance_n[i], reverse=True)[:10]
print('Top 10 features by gain (normalized):')
for i in top_idx_n:
    print(f'  {X.columns[i]:40s} {importance_n[i]:.1f}')