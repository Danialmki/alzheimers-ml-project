from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    brier_score_loss,
    log_loss,
)

from src.utils import PROJECT_ROOT, RANDOM_STATE, ensure_directories, write_json


def expected_calibration_error(y_true, probabilities, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            error += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(error)


def bootstrap_intervals(y_true, probabilities, iterations: int = 1000) -> dict[str, list[float]]:
    rng = np.random.default_rng(RANDOM_STATE)
    y_array = np.asarray(y_true)
    p_array = np.asarray(probabilities)
    samples: dict[str, list[float]] = {"roc_auc": [], "brier_score": []}
    for _ in range(iterations):
        indices = rng.integers(0, len(y_array), len(y_array))
        if np.unique(y_array[indices]).size < 2:
            continue
        samples["roc_auc"].append(roc_auc_score(y_array[indices], p_array[indices]))
        samples["brier_score"].append(brier_score_loss(y_array[indices], p_array[indices]))
    return {
        name: [float(v) for v in np.percentile(values, [2.5, 97.5])]
        for name, values in samples.items()
    }


def evaluate(model_path: str | Path = PROJECT_ROOT / "models/model.joblib") -> dict[str, float]:
    ensure_directories()
    artifact = joblib.load(model_path)
    pipeline, X_test, y_test = artifact["pipeline"], artifact["test_X"], artifact["test_y"]
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities)),
        "expected_calibration_error_10_bins": expected_calibration_error(
            y_test.to_numpy(), probabilities
        ),
        "test_samples": int(len(y_test)),
        "bootstrap_95_ci": bootstrap_intervals(y_test, probabilities),
    }
    write_json(metrics, PROJECT_ROOT / "results/metrics/test_metrics.json")
    report = pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).T
    report.to_csv(PROJECT_ROOT / "results/tables/classification_report.csv")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual", title="Test-set confusion matrix")
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "results/figures/confusion_matrix.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax)
    ax.set_title("Test-set ROC curve")
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "results/figures/roc_curve.png", dpi=200)
    plt.close(fig)

    fraction_positive, mean_predicted = calibration_curve(
        y_test, probabilities, n_bins=10, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.plot(mean_predicted, fraction_positive, marker="o", label="Model")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title="Test-set reliability diagram",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "results/figures/reliability_diagram.png", dpi=200)
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/model.joblib")
    args = parser.parse_args()
    print(evaluate(args.model))


if __name__ == "__main__":
    main()
