from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils import PROJECT_ROOT, ensure_directories


def generate(model_path: Path = PROJECT_ROOT / "models/model.joblib") -> pd.DataFrame:
    ensure_directories()
    artifact = joblib.load(model_path)
    pipeline = artifact["explanation_pipeline"]
    X_test = artifact["test_X"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(X_test)
    names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(classifier)
    raw = explainer.shap_values(transformed)
    if isinstance(raw, list):
        values = np.asarray(raw[1])
        expected = np.asarray(explainer.expected_value)[1]
    elif np.asarray(raw).ndim == 3:
        values = np.asarray(raw)[:, :, 1]
        expected = np.asarray(explainer.expected_value)[1]
    else:
        values = np.asarray(raw)
        expected = np.asarray(explainer.expected_value).reshape(-1)[0]

    table = pd.DataFrame({"feature": names, "mean_absolute_shap": np.abs(values).mean(axis=0)})
    table = table.sort_values("mean_absolute_shap", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table.to_csv(PROJECT_ROOT / "results/tables/shap_importance.csv", index=False)

    shap.summary_plot(values, transformed, feature_names=names, show=False, max_display=15)
    plt.title("SHAP summary: base random forest on locked test observations")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "results/figures/shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    top = table.head(15).sort_values("mean_absolute_shap")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["mean_absolute_shap"])
    ax.set(xlabel="Mean absolute SHAP value", title="Global SHAP importance: base random forest")
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "results/figures/shap_bar.png", dpi=300)
    plt.close(fig)

    probabilities = artifact["pipeline"].predict_proba(X_test)[:, 1]
    representative_position = int(np.argmin(np.abs(probabilities - np.median(probabilities))))
    explanation = shap.Explanation(
        values=values[representative_position],
        base_values=expected,
        data=transformed[representative_position],
        feature_names=list(names),
    )
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.title(f"Representative test observation (row position {representative_position})")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "results/figures/shap_individual_waterfall.png", dpi=300, bbox_inches="tight")
    plt.close()
    (PROJECT_ROOT / "results/metrics/shap_metadata.json").write_text(
        pd.Series({
            "observations": "locked test set",
            "n_observations": len(X_test),
            "explainer": "shap.TreeExplainer",
            "model": "uncalibrated selected random forest refit on development set",
            "feature_names": "transformed feature names from fitted preprocessor",
            "representative_test_position": representative_position,
            "representative_probability_from_final_calibrated_model": float(probabilities[representative_position]),
        }).to_json(indent=2),
        encoding="utf-8",
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/model.joblib")
    args = parser.parse_args()
    print(generate(args.model).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
