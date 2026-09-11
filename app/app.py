from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_data, split_features_target  # noqa: E402

st.set_page_config(page_title="Alzheimer's ML Demo", page_icon="🧠", layout="wide")
st.title("Alzheimer's disease classification demo")
st.warning(
    "Educational demonstration only. This model was trained on synthetic data and must not be used "
    "for diagnosis, treatment, screening, or other clinical decisions."
)

model_path = PROJECT_ROOT / "models/model.joblib"
if not model_path.exists():
    st.error("Model artifact missing. Run `python -m src.train` first.")
    st.stop()

artifact = joblib.load(model_path)
data = load_data()
X, _ = split_features_target(data)
defaults = X.median(numeric_only=True).to_dict()

st.sidebar.header("Patient features")
values: dict[str, float | int] = {}
for column in artifact["feature_columns"]:
    series = X[column]
    if column in {"Gender", "Ethnicity", "EducationLevel"} or set(series.dropna().unique()).issubset({0, 1}):
        options = sorted(int(v) for v in series.dropna().unique())
        values[column] = st.sidebar.selectbox(column, options, index=0)
    else:
        values[column] = st.sidebar.number_input(
            column,
            value=float(defaults.get(column, series.dropna().iloc[0])),
            min_value=float(series.min()),
            max_value=float(series.max()),
        )

row = pd.DataFrame([values], columns=artifact["feature_columns"])
probability = artifact["pipeline"].predict_proba(row)[0, 1]
st.metric("Model-estimated probability of positive label", f"{probability:.1%}")
st.caption(f"Selected model: {artifact['model_name']}. Output is not a clinical probability.")

