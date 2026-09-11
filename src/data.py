from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import PROJECT_ROOT

DEFAULT_DATA_PATH = PROJECT_ROOT / "data/raw/alzheimers_disease_data.csv"
TARGET = "Diagnosis"
NON_PREDICTIVE_COLUMNS = ("PatientID", "DoctorInCharge")


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing")
    if frame.empty:
        raise ValueError("Dataset is empty")
    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop_columns = [TARGET, *[c for c in NON_PREDICTIVE_COLUMNS if c in frame.columns]]
    features = frame.drop(columns=drop_columns)
    target = frame[TARGET].astype(int)
    if not set(target.unique()).issubset({0, 1}):
        raise ValueError("Diagnosis must be binary (0/1)")
    return features, target

