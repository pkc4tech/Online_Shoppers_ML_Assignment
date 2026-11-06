"""Interactive evaluation of pretrained Online Shoppers classifiers; no model training."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from train_models import get_scores, prepare_features, target_to_int

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Shopper Purchase Prediction Lab', page_icon='🛒', layout='wide')
st.title('🛒 Shopper Purchase Prediction Lab')
st.caption('BITS ML Assignment 2 | Five pretrained classifiers | Test-set-only model evaluation')
st.markdown('Predict whether an online browsing session results in a purchase (**Revenue = True**).')

@st.cache_resource
def load_model(filename):
    return joblib.load(ROOT / 'model' / filename)

try:
    meta = json.loads((ROOT / 'model' / 'metadata.json').read_text(encoding='utf-8'))
except FileNotFoundError:
    st.error('Saved model files not found. Run `python train_models.py` before opening this app.')
    st.stop()

with st.sidebar:
    st.header('Experiment settings')
    selected = st.selectbox('Select classification model', list(meta['model_files']))
    st.info(f"Dataset: {meta['all_rows']:,} records, {len(meta['features'])} features. Model training uses {meta['train_rows']:,} training sessions.")
    st.markdown('**Positive class:** purchase (Revenue = True)')

upload = st.file_uploader('Upload a held-out test CSV (17 input features + Revenue target)', type=['csv'])
st.caption('Use the included `test_data.csv` for results that match the README. Do not upload training records to claim independent test performance.')
if upload is None:
    st.warning('Please upload `test_data.csv` to view evaluation results.')
    st.stop()

try:
    raw = pd.read_csv(upload)
    required = meta['features'] + [meta['target']]
    absent = [c for c in required if c not in raw.columns]
    if absent:
        raise ValueError(f'Missing required columns: {absent}')
    if raw[required].isna().any().any():
        raise ValueError('Missing values found in required columns.')
    x = prepare_features(raw, meta['features'], meta['numeric_features'], meta['categorical_features'])
    y = target_to_int(raw[meta['target']])
    model = load_model(meta['model_files'][selected])
    result = get_scores(model, x, y)
    st.subheader(f'{selected} — evaluation on {len(raw):,} rows')
    panels = st.columns(3)
    for i, (name, value) in enumerate(result.items()):
        panels[i % 3].metric(name, 'N/A' if pd.isna(value) else f'{value:.4f}')
    if y.nunique() < 2:
        st.warning('AUC is undefined because your uploaded test CSV has only one target class.')
    left, right = st.columns(2)
    preds = model.predict(x)
    with left:
        st.subheader('Confusion matrix')
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(y, preds, labels=[0, 1]), display_labels=['No purchase (0)', 'Purchase (1)']).plot(ax=ax, colorbar=False, cmap='Blues')
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader('Classification report')
        report = classification_report(y, preds, labels=[0, 1], target_names=['No purchase', 'Purchase'], output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)
    st.subheader('All five models on the uploaded test CSV')
    summary = pd.DataFrame.from_dict({name: get_scores(load_model(filename), x, y) for name, filename in meta['model_files'].items()}, orient='index')
    summary.index.name = 'ML Model Name'
    st.dataframe(summary, use_container_width=True)
    st.download_button('Download evaluation table', summary.to_csv().encode('utf-8'), 'uploaded_test_metrics.csv', mime='text/csv')
    st.caption('The application only loads saved fitted pipelines and calculates predictions and metrics. It does not call fit() or retrain.')
except (ValueError, KeyError, TypeError) as error:
    st.error(f'Invalid test CSV: {error}')
