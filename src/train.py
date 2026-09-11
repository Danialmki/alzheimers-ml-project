from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import DEFAULT_DATA_PATH, load_data, split_features_target
from src.preprocessing import build_preprocessor
from src.utils import PROJECT_ROOT, RANDOM_STATE, ensure_directories, seed_everything, write_json


def candidate_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train(data_path: str | Path = DEFAULT_DATA_PATH) -> dict[str, object]:
    seed_everything()
    ensure_directories()
    frame = load_data(data_path)
    X, y = split_features_target(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )

    scores: dict[str, float] = {}
    fitted: dict[str, Pipeline] = {}
    for name, model in candidate_models().items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor(list(X.columns))), ("classifier", model)]
        )
        pipeline.fit(X_train, y_train)
        scores[name] = float(roc_auc_score(y_validation, pipeline.predict_proba(X_validation)[:, 1]))
        fitted[name] = pipeline

    best_name = max(scores, key=scores.get)
    development_X = pd.concat([X_train, X_validation])
    development_y = pd.concat([y_train, y_validation])
    explanation_pipeline = fitted[best_name]
    explanation_pipeline.fit(development_X, development_y)
    # Cross-validated sigmoid calibration avoids fitting the calibration map on
    # the same predictions used to train each underlying estimator.
    calibrated_pipeline = CalibratedClassifierCV(
        estimator=Pipeline(
            [
                ("preprocessor", build_preprocessor(list(X.columns))),
                ("classifier", candidate_models()[best_name]),
            ]
        ),
        method="sigmoid",
        cv=5,
        ensemble=True,
        n_jobs=-1,
    )
    calibrated_pipeline.fit(development_X, development_y)

    artifact = {
        "pipeline": calibrated_pipeline,
        "explanation_pipeline": explanation_pipeline,
        "model_name": best_name,
        "feature_columns": list(X.columns),
        "test_X": X_test,
        "test_y": y_test,
        "validation_roc_auc": scores,
        "random_state": RANDOM_STATE,
        "calibration": "5-fold cross-validated sigmoid",
    }
    joblib.dump(artifact, PROJECT_ROOT / "models/model.joblib")
    write_json(
        {"selected_model": best_name, "validation_roc_auc": scores, "random_state": RANDOM_STATE},
        PROJECT_ROOT / "results/metrics/training.json",
    )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Alzheimer's classifier")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()
    artifact = train(args.data)
    print(f"Saved {artifact['model_name']} to models/model.joblib")


if __name__ == "__main__":
    main()
