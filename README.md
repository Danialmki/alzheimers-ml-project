# Alzheimer's Disease ML Project

An end-to-end, reproducible binary-classification project using Rabie El Kharoua's synthetic [Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset).

> **Not for clinical use.** The source data is synthetic. Model outputs are educational and are not diagnoses, medical advice, or validated risk estimates.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
python -m src.explain
streamlit run app/app.py
```

Run commands from the repository root. Generated metrics, tables, and figures are written under `results/`.

## Dataset acquisition

The downloaded Kaggle CSV is intentionally not stored in this repository. Download the [Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset) under its CC BY 4.0 terms, then place it at:

```text
data/raw/alzheimers_disease_data.csv
```

With the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) already authenticated, this can be done from the repository root:

```bash
mkdir -p data/raw
kaggle datasets download \
  -d rabieelkharoua/alzheimers-disease-dataset \
  -p data/raw \
  --unzip
```

The training pipeline writes the fitted model to `models/model.joblib`; that reproducible binary is also intentionally excluded from version control.

## Design

- Drops `PatientID` and constant `DoctorInCharge` to prevent identifier leakage.
- Uses stratified train/validation/test partitions.
- Compares logistic regression and random forest using validation ROC AUC.
- Evaluates the selected model once on a held-out test set.
- Applies 5-fold out-of-fold sigmoid probability calibration.
- Reports discrimination and calibration metrics with bootstrap intervals.
- Keeps preprocessing inside the fitted scikit-learn pipeline.
- Provides a clearly labelled educational Streamlit interface.

## Project layout

```text
data/raw/                         source CSV
data/processed/                   optional derived data
notebooks/exploratory_analysis.ipynb
src/                              pipeline modules
app/app.py                        Streamlit demo
models/                           local artifacts (ignored)
results/{tables,figures,metrics}/ generated outputs
paper/                            manuscript and citation
tests/                            smoke tests
```

## Private NotebookLM references

The supplied NotebookLM collections informed the leakage controls, cross-validated calibration, reliability reporting, explainability choices, reproducibility safeguards, and claim discipline. They remain private links and are not treated as publicly retrievable citations; the manuscript must cite the original works selected from those collections.

## Data license and citation

The Kaggle data card lists CC BY 4.0. Cite the dataset using `paper/references/dataset.bib`. Keep the original source attribution in derivative work.
