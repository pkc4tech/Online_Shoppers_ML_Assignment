"""Train five classification pipelines using the original Online Shoppers CSV.

Run once from the project directory: python train_models.py
The Streamlit app ONLY loads these saved fitted pipelines; it never retrains.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / 'online_shoppers_intention.csv'
TARGET = 'Revenue'
SEED = 42
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression.joblib',
    'Decision Tree': 'decision_tree.joblib',
    'KNN': 'knn.joblib',
    'Naive Bayes': 'naive_bayes.joblib',
    'Support Vector Machine': 'support_vector_machine.joblib',
}
# Treat numeric category IDs as categories rather than continuous quantities.
CATEGORICAL = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']

def target_to_int(column):
    """Handle bool, 0/1, and True/False labels from CSV consistently."""
    labels = column.astype(str).str.strip().str.lower()
    mapping = {'true': 1, 'false': 0, '1': 1, '0': 0}
    if not labels.isin(mapping).all():
        raise ValueError('Revenue must contain only True/False or 1/0.')
    return labels.map(mapping).astype(int)

def prepare_features(frame, features, numeric, categorical):
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f'Missing feature columns: {missing}')
    x = frame[features].copy()
    if x.empty or x.isna().any().any():
        raise ValueError('Feature columns cannot contain missing values or zero rows.')
    for name in numeric:
        x[name] = pd.to_numeric(x[name], errors='raise')
    if not np.isfinite(x[numeric].to_numpy(dtype=float)).all():
        raise ValueError('Numeric features must be finite.')
    for name in categorical:
        x[name] = x[name].astype(str).str.strip()
    return x

def get_scores(model, x, actual):
    pred = model.predict(x)
    if hasattr(model, 'decision_function'):
        confidence = model.decision_function(x)
    else:
        confidence = model.predict_proba(x)[:, list(model.classes_).index(1)]
    metrics = {
        'Accuracy': accuracy_score(actual, pred),
        'AUC': roc_auc_score(actual, confidence) if len(np.unique(actual)) == 2 else np.nan,
        'Precision': precision_score(actual, pred, zero_division=0),
        'Recall': recall_score(actual, pred, zero_division=0),
        'F1': f1_score(actual, pred, zero_division=0),
        'MCC': matthews_corrcoef(actual, pred),
    }
    return {key: round(float(value), 4) for key, value in metrics.items()}

def main():
    raw = pd.read_csv(DATA_FILE)
    if len(raw) < 500 or len(raw.columns) - 1 < 12:
        raise ValueError('Dataset does not meet the 500-instance / 12-feature minimum.')
    if TARGET not in raw.columns:
        raise ValueError(f'Expected target column {TARGET!r}.')
    features = [c for c in raw.columns if c != TARGET]
    categorical = [c for c in CATEGORICAL if c in features]
    numeric = [c for c in features if c not in categorical]
    y = target_to_int(raw[TARGET]); x = prepare_features(raw, features, numeric, categorical)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=SEED, stratify=y)
    # The preprocessing is saved *inside* each pipeline and fitted on training rows only.
    def preprocessing():
        return ColumnTransformer([
            ('continuous', StandardScaler(), numeric),
            ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
        ])
    estimators = {
        'Logistic Regression': LogisticRegression(max_iter=3000, random_state=SEED),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=SEED),
        'KNN': KNeighborsClassifier(n_neighbors=11),
        'Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(kernel='rbf', C=1.0, gamma='scale'),
    }
    model_dir = ROOT / 'model'; model_dir.mkdir(exist_ok=True)
    # Preserve original readable CSV labels/types for the app and reproducibility.
    test_raw = raw.loc[x_test.index, features + [TARGET]].copy()
    test_raw.to_csv(ROOT / 'test_data.csv', index=False)
    results = {}
    for name, estimator in estimators.items():
        print('Training:', name, flush=True)
        pipeline = Pipeline([('preprocess', preprocessing()), ('classifier', estimator)])
        pipeline.fit(x_train, y_train)
        results[name] = get_scores(pipeline, x_test, y_test)
        joblib.dump(pipeline, model_dir / MODEL_FILES[name])
    summary = pd.DataFrame.from_dict(results, orient='index').rename_axis('ML Model Name')
    summary.to_csv(ROOT / 'metrics.csv')
    info = {
        'dataset': 'Online Shoppers Purchasing Intention', 'target': TARGET,
        'features': features, 'numeric_features': numeric, 'categorical_features': categorical,
        'train_rows': len(x_train), 'test_rows': len(x_test), 'all_rows': len(raw),
        'positive_class': 'Revenue=True (purchase)', 'test_size': 0.2,
        'random_state': SEED, 'model_files': MODEL_FILES,
        'notes': 'Train/test split is stratified. All five models use the same split. Preprocessing fitted only on training rows.'
    }
    (model_dir / 'metadata.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print('\nHeld-out test metrics (positive class: purchase):\n')
    print(summary.to_string())
    print('\nCreated model/*.joblib, model/metadata.json, metrics.csv and test_data.csv')

if __name__ == '__main__':
    main()
