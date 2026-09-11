from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.utils import PROJECT_ROOT, ensure_directories


def explain(model_path: str | Path = PROJECT_ROOT / "models/model.joblib") -> pd.DataFrame:
    ensure_directories()
    artifact = joblib.load(model_path)
    pipeline = artifact["explanation_pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = abs(classifier.coef_[0])
    else:
        raise TypeError("Selected classifier exposes no supported importance values")
    importance = pd.DataFrame({"feature": names, "importance": values}).sort_values(
        "importance", ascending=False
    )
    importance.to_csv(PROJECT_ROOT / "results/tables/feature_importance.csv", index=False)
    top = importance.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set(title="Top model feature importances", xlabel="Importance")
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "results/figures/feature_importance.png", dpi=200)
    plt.close(fig)
    return importance


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a global model explanation")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/model.joblib")
    args = parser.parse_args()
    print(explain(args.model).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
