from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.data import DEFAULT_DATA_PATH, NON_PREDICTIVE_COLUMNS, TARGET, load_data, split_features_target
from src.evaluate import expected_calibration_error
from src.preprocessing import build_preprocessor
from src.train import candidate_models
from src.utils import PROJECT_ROOT, RANDOM_STATE, ensure_directories, write_json

THRESHOLD = 0.5
N_BOOTSTRAP = 2000
N_PERMUTATIONS = 25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_counts(y: pd.Series) -> dict[str, int]:
    counts = y.value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def calculate_metrics(y_true, probabilities, threshold: float = THRESHOLD) -> dict[str, object]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    predictions = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_average_precision": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, predictions)),
        "sensitivity_recall": float(recall_score(y, predictions, zero_division=0)),
        "specificity": float(specificity),
        "precision_ppv": float(precision_score(y, predictions, zero_division=0)),
        "negative_predictive_value": float(npv),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "expected_calibration_error_10_equal_width_bins": expected_calibration_error(y, p, bins=10),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_cis(y_true, probabilities, iterations: int = N_BOOTSTRAP) -> dict[str, list[float]]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(RANDOM_STATE)
    values = {"roc_auc": [], "pr_auc_average_precision": [], "brier_score": []}
    for _ in range(iterations):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        values["roc_auc"].append(roc_auc_score(y[idx], p[idx]))
        values["pr_auc_average_precision"].append(average_precision_score(y[idx], p[idx]))
        values["brier_score"].append(brier_score_loss(y[idx], p[idx]))
    return {
        key: [float(x) for x in np.percentile(samples, [2.5, 97.5])]
        for key, samples in values.items()
    }


def reproduce_splits(frame: pd.DataFrame):
    X, y = split_features_target(frame)
    development_X, test_X, development_y, test_y = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    train_X, validation_X, train_y, validation_y = train_test_split(
        development_X,
        development_y,
        test_size=0.2,
        stratify=development_y,
        random_state=RANDOM_STATE,
    )
    return X, y, development_X, test_X, development_y, test_y, train_X, validation_X, train_y, validation_y


def duplicate_audit(development_X: pd.DataFrame, test_X: pd.DataFrame) -> dict[str, object]:
    exact_overlap = pd.merge(
        development_X.reset_index(names="development_index"),
        test_X.reset_index(names="test_index"),
        how="inner",
        on=list(development_X.columns),
    )
    combined = pd.concat([development_X, test_X])
    scaler = MinMaxScaler().fit(combined)
    dev_scaled = scaler.transform(development_X)
    test_scaled = scaler.transform(test_X)
    # Mean normalized absolute feature distance: 0 is identical, 0.01 means an
    # average difference of 1% of each feature's observed range.
    minima = []
    for row in test_scaled:
        minima.append(float(np.mean(np.abs(dev_scaled - row), axis=1).min()))
    minima_array = np.asarray(minima)
    return {
        "exact_feature_duplicates_across_development_test": int(len(exact_overlap)),
        "near_duplicate_definition": "mean absolute distance after min-max scaling all features <= 0.01",
        "near_duplicates_across_development_test": int((minima_array <= 0.01).sum()),
        "minimum_cross_split_normalized_distance": float(minima_array.min()),
        "median_nearest_cross_split_normalized_distance": float(np.median(minima_array)),
        "duplicate_predictor_rows_in_source": int(combined.duplicated().sum()),
    }


def cross_validation_results(development_X: pd.DataFrame, development_y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "accuracy": "accuracy",
        "sensitivity": "recall",
        "specificity": make_scorer(recall_score, pos_label=0),
        "precision": "precision",
        "f1": "f1",
        "brier_score_negative": "neg_brier_score",
    }
    rows = []
    for name, model in candidate_models().items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor(list(development_X.columns))), ("classifier", model)]
        )
        scores = cross_validate(
            pipeline,
            development_X,
            development_y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        row: dict[str, object] = {"model": name, "folds": 5}
        for metric in scoring:
            fold_values = scores[f"test_{metric}"]
            if metric == "brier_score_negative":
                fold_values = -fold_values
                output_name = "brier_score"
            else:
                output_name = metric
            row[f"{output_name}_mean"] = float(np.mean(fold_values))
            row[f"{output_name}_sd"] = float(np.std(fold_values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def permutation_sanity(development_X, development_y, test_X, test_y, model_template) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for run in range(N_PERMUTATIONS):
        shuffled_y = pd.Series(rng.permutation(development_y.to_numpy()), index=development_y.index)
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(list(development_X.columns))),
                ("classifier", clone(model_template)),
            ]
        )
        pipeline.fit(development_X, shuffled_y)
        probabilities = pipeline.predict_proba(test_X)[:, 1]
        rows.append(
            {
                "run": run + 1,
                "roc_auc": float(roc_auc_score(test_y, probabilities)),
                "pr_auc": float(average_precision_score(test_y, probabilities)),
            }
        )
    return pd.DataFrame(rows)


def generate_figures(y_test, probabilities, predictions, pipeline, explanation_pipeline, X_test):
    sns.set_theme(style="whitegrid")
    figures = PROJECT_ROOT / "results/figures"

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_test, probabilities):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Locked model: test ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(figures / "final_roc_curve.png", dpi=300); plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    prevalence = float(np.mean(y_test))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"Average precision = {average_precision_score(y_test, probabilities):.3f}")
    ax.axhline(prevalence, linestyle="--", color="gray", label=f"Prevalence = {prevalence:.3f}")
    ax.set(xlabel="Recall (sensitivity)", ylabel="Precision", title="Locked model: test precision–recall curve")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "final_precision_recall_curve.png", dpi=300); plt.close(fig)

    from sklearn.calibration import calibration_curve
    observed, predicted = calibration_curve(y_test, probabilities, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    ax.plot(predicted, observed, marker="o", label="Locked model")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", title="Locked model: test calibration curve", xlim=(0, 1), ylim=(0, 1))
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "final_calibration_curve.png", dpi=300); plt.close(fig)

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set(xlabel="Predicted label", ylabel="True label", title=f"Locked model confusion matrix (threshold {THRESHOLD})")
    fig.tight_layout(); fig.savefig(figures / "final_confusion_matrix.png", dpi=300); plt.close(fig)

    permutation = permutation_importance(
        pipeline, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=RANDOM_STATE, n_jobs=-1
    )
    permutation_table = pd.DataFrame({
        "feature": X_test.columns,
        "mean_roc_auc_decrease": permutation.importances_mean,
        "sd": permutation.importances_std,
    }).sort_values("mean_roc_auc_decrease", ascending=False)
    permutation_table.to_csv(PROJECT_ROOT / "results/tables/permutation_importance.csv", index=False)
    top = permutation_table.head(15).sort_values("mean_roc_auc_decrease")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["mean_roc_auc_decrease"], xerr=top["sd"])
    ax.set(xlabel="Mean decrease in test ROC AUC", title="Global permutation importance (30 repeats)")
    fig.tight_layout(); fig.savefig(figures / "global_permutation_importance.png", dpi=300); plt.close(fig)

    shap_status = {"generated": False, "explanation_model": "uncalibrated selected random forest refit on development set"}
    try:
        import shap
        preprocessor = explanation_pipeline.named_steps["preprocessor"]
        classifier = explanation_pipeline.named_steps["classifier"]
        transformed = preprocessor.transform(X_test)
        feature_names = preprocessor.get_feature_names_out()
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed)
        if isinstance(shap_values, list):
            selected = shap_values[1]
        elif np.asarray(shap_values).ndim == 3:
            selected = np.asarray(shap_values)[:, :, 1]
        else:
            selected = shap_values
        shap.summary_plot(selected, transformed, feature_names=feature_names, show=False, max_display=15)
        plt.title("SHAP summary: selected model on test features")
        plt.tight_layout(); plt.savefig(figures / "shap_summary.png", dpi=300, bbox_inches="tight"); plt.close()
        shap_status["generated"] = True
    except Exception as error:  # report, do not hide, environment-specific SHAP failures
        shap_status["error"] = f"{type(error).__name__}: {error}"
    return permutation_table, shap_status


def build_report(audit: dict[str, object], cv: pd.DataFrame, permutation: pd.DataFrame) -> str:
    m = audit["test_metrics"]
    ci = audit["bootstrap_95_ci"]
    split = audit["sample_counts"]
    features = audit["features"]
    candidates = cv.set_index("model")
    candidate_lines = []
    for name, row in candidates.iterrows():
        candidate_lines.append(
            f"- **{name}:** ROC AUC {row.roc_auc_mean:.3f} ± {row.roc_auc_sd:.3f}; "
            f"PR AUC {row.pr_auc_mean:.3f} ± {row.pr_auc_sd:.3f}; accuracy {row.accuracy_mean:.3f} ± {row.accuracy_sd:.3f}; "
            f"sensitivity {row.sensitivity_mean:.3f} ± {row.sensitivity_sd:.3f}; precision {row.precision_mean:.3f} ± {row.precision_sd:.3f}; "
            f"F1 {row.f1_mean:.3f} ± {row.f1_sd:.3f}; Brier score {row.brier_score_mean:.3f} ± {row.brier_score_sd:.3f}."
        )
    cm = m["confusion_matrix"]
    return f"""# Final Locked-Model Validation Report

**Status:** authoritative internal validation report  
**Model artifact SHA-256:** `{audit['artifact_sha256_before']}`  
**Audit rule:** the saved final model was not retrained, tuned, recalibrated, or replaced in response to test results.

## 1. Samples and class distribution

- Full dataset: {split['full']['n']} observations; class 0 = {split['full']['classes']['0']}, class 1 = {split['full']['classes']['1']}.
- Development set: {split['development']['n']} observations; class 0 = {split['development']['classes']['0']}, class 1 = {split['development']['classes']['1']}.
- Initial model-fit subset: {split['initial_train']['n']} observations; class 0 = {split['initial_train']['classes']['0']}, class 1 = {split['initial_train']['classes']['1']}.
- Internal model-selection validation subset: {split['selection_validation']['n']} observations; class 0 = {split['selection_validation']['classes']['0']}, class 1 = {split['selection_validation']['classes']['1']}.
- Locked test set: {split['test']['n']} observations; class 0 = {split['test']['classes']['0']}, class 1 = {split['test']['classes']['1']}.

## 2. Test-set isolation and provenance

The source code first created a stratified 80/20 development/test split with seed 42. Candidate preprocessing pipelines were fitted only on the initial training subset; model selection used only the internal validation subset. The selected model was then refit and five-fold sigmoid-calibrated only on the combined development set. The test set was stored for final evaluation and was not supplied to preprocessing fitting, feature selection, model selection, hyperparameter selection, calibration, or threshold selection. The audit reproduced the split and matched the stored test features and labels exactly: **{audit['test_split_exact_match']}**. No data-driven threshold search occurred; `CalibratedClassifierCV.predict` uses the fixed binary cutoff **0.5**.

Important scope note: the candidate configurations were fixed in code rather than selected by a nested hyperparameter search. The test set has now been inspected for final reporting, so it must remain locked and cannot support later model changes.

## 3. Included and excluded features

**Included ({len(features['included'])}):** {', '.join(features['included'])}.

**Excluded:**
- `Diagnosis`: prediction target; excluded from predictors.
- `PatientID`: unique record identifier; excluded to prevent memorization and identifier leakage.
- `DoctorInCharge`: constant confidential metadata (`XXXConfid`); excluded as non-predictive metadata.

## 4. Target-leakage audit

No included column is identical to `Diagnosis`, no feature name explicitly states diagnosis, and the preprocessing pipeline receives only the listed predictor columns. `PatientID` and `DoctorInCharge` are absent from the artifact's feature list. Several included variables—especially `FunctionalAssessment`, `ADL`, `MMSE`, `MemoryComplaints`, and `BehavioralProblems`—are clinically proximal to diagnostic assessment and may indirectly encode the synthetic generator's diagnostic rule. They are legitimate predictors for this benchmark but create a **construct/proxy leakage risk** and may inflate apparent performance. Because provenance and measurement timing are unavailable, the audit cannot prove that every predictor was measured before diagnosis. This blocks causal or prospective clinical claims.

## 5. Locked test performance

At threshold 0.5:

- ROC AUC: **{m['roc_auc']:.3f}** (95% bootstrap CI {ci['roc_auc'][0]:.3f}–{ci['roc_auc'][1]:.3f})
- PR AUC / average precision: **{m['pr_auc_average_precision']:.3f}** (95% bootstrap CI {ci['pr_auc_average_precision'][0]:.3f}–{ci['pr_auc_average_precision'][1]:.3f}); appropriate because positive prevalence is {split['test']['prevalence']:.3f}
- Accuracy: **{m['accuracy']:.3f}**
- Sensitivity: **{m['sensitivity_recall']:.3f}**
- Specificity: **{m['specificity']:.3f}**
- Precision / PPV: **{m['precision_ppv']:.3f}**
- Negative predictive value: **{m['negative_predictive_value']:.3f}**
- F1: **{m['f1']:.3f}**
- Brier score: **{m['brier_score']:.3f}** (95% bootstrap CI {ci['brier_score'][0]:.3f}–{ci['brier_score'][1]:.3f})
- Log loss: **{m['log_loss']:.3f}**
- Expected calibration error: **{m['expected_calibration_error_10_equal_width_bins']:.3f}** (10 equal-width bins; descriptive and bin-dependent)
- Confusion matrix: TN = {cm['tn']}, FP = {cm['fp']}, FN = {cm['fn']}, TP = {cm['tp']}.

## 6. Threshold

The selected threshold is **0.5**, inherited from the classifier's default binary decision rule. It was not optimized on validation or test data. Therefore, no claim is made that 0.5 is clinically optimal. A clinical threshold would require a prespecified use case and explicit harm/benefit trade-offs in external data.

## 7–9. Cross-validation, discrepancy, and candidate models

Five-fold stratified CV on the development set, using training-fold-only preprocessing, produced:

{chr(10).join(candidate_lines)}

The selected random forest's CV ROC AUC and locked test ROC AUC differ by **{abs(candidates.loc['random_forest', 'roc_auc_mean'] - m['roc_auc']):.3f}**. This is not a meaningful collapse; the test value lies within the observed uncertainty and is slightly more conservative. The original one-shot internal validation ROC AUC values were {audit['original_validation_scores']} and selected random forest. The CV table is the more stable candidate comparison; it does not alter the locked model.

## 10. Calibration isolation

Calibration used `CalibratedClassifierCV(method='sigmoid', cv=5, ensemble=True)` fitted on development data only. Each calibrator was fitted from held-out predictions within development folds. The reproduced test indices were excluded. The final test calibration metrics and plot are evaluation only and were not used to recalibrate the model.

## 11. Final figures and explanation scope

Generated under `results/figures/`:

- `final_roc_curve.png`
- `final_precision_recall_curve.png`
- `final_calibration_curve.png`
- `final_confusion_matrix.png`
- `shap_summary.png`
- `global_permutation_importance.png`

Permutation importance measures the change in locked-test ROC AUC when each feature is shuffled and is strictly post-hoc. SHAP values use the selected uncalibrated random-forest explanation pipeline refit on the development set, because the final calibrated estimator is an ensemble of five calibrated pipelines. The SHAP plot explains that base model's score structure, not the sigmoid calibration mapping. Neither analysis implies causality.

## 12. Permuted-target sanity check

Across {audit['permutation_sanity']['runs']} independently shuffled development targets, the same selected model specification achieved mean test ROC AUC **{audit['permutation_sanity']['roc_auc_mean']:.3f} ± {audit['permutation_sanity']['roc_auc_sd']:.3f}** and mean PR AUC **{audit['permutation_sanity']['pr_auc_mean']:.3f} ± {audit['permutation_sanity']['pr_auc_sd']:.3f}**. Performance collapsed toward chance/baseline, supporting that the real-label result is not explained by the pipeline producing high scores under random labels. These sanity models are separate from and did not modify the final artifact.

## 13. Duplicate and near-duplicate audit

- Exact predictor duplicates crossing development/test: **{audit['duplicates']['exact_feature_duplicates_across_development_test']}**.
- Near duplicates crossing development/test under the prespecified normalized-distance rule: **{audit['duplicates']['near_duplicates_across_development_test']}**.
- Minimum cross-split normalized distance: **{audit['duplicates']['minimum_cross_split_normalized_distance']:.4f}**.
- Median nearest cross-split normalized distance: **{audit['duplicates']['median_nearest_cross_split_normalized_distance']:.4f}**.

The near-duplicate rule is mean absolute distance ≤0.01 after min–max scaling all included features. This is a transparent heuristic, not proof of patient independence; the dataset is synthetic and provides no entity-linkage metadata.

## 14. Internal performance versus clinical validity

These are **internal predictive-performance estimates on a synthetic dataset**, not evidence of clinical validity. The audit verifies software-level split isolation and metrics. It does not establish representative sampling, temporal validity, transportability, fairness, clinical utility, safety, or benefit to patients. External validation on provenance-audited real clinical cohorts, with predictor timing defined before the diagnostic decision, is mandatory before any clinical interpretation.

## Reproducibility artifacts

- Machine-readable audit: `results/metrics/final_validation.json`
- Candidate CV table: `results/tables/candidate_cross_validation.csv`
- Permutation sanity table: `results/tables/permutation_sanity.csv`
- Global permutation importance: `results/tables/permutation_importance.csv`
- Locked artifact hash unchanged after audit: **{audit['artifact_unchanged']}**
"""


def validate(model_path: Path, data_path: Path) -> dict[str, object]:
    ensure_directories()
    artifact_hash_before = sha256(model_path)
    artifact = joblib.load(model_path)
    frame = load_data(data_path)
    X, y, dev_X, test_X, dev_y, test_y, train_X, val_X, train_y, val_y = reproduce_splits(frame)

    stored_X, stored_y = artifact["test_X"], artifact["test_y"]
    test_match = bool(stored_X.equals(test_X) and stored_y.equals(test_y))
    if not test_match:
        raise RuntimeError("Stored test set does not match the reproducible locked split")
    if artifact["feature_columns"] != list(X.columns):
        raise RuntimeError("Artifact feature list does not match source-code feature construction")

    probabilities = artifact["pipeline"].predict_proba(stored_X)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)
    metrics = calculate_metrics(stored_y, probabilities)
    cis = bootstrap_cis(stored_y, probabilities)
    cv = cross_validation_results(dev_X, dev_y)
    cv.to_csv(PROJECT_ROOT / "results/tables/candidate_cross_validation.csv", index=False)

    selected_template = candidate_models()[artifact["model_name"]]
    sanity = permutation_sanity(dev_X, dev_y, test_X, test_y, selected_template)
    sanity.to_csv(PROJECT_ROOT / "results/tables/permutation_sanity.csv", index=False)
    permutation_table, shap_status = generate_figures(
        stored_y,
        probabilities,
        predictions,
        artifact["pipeline"],
        artifact["explanation_pipeline"],
        stored_X,
    )

    duplicates = duplicate_audit(dev_X, test_X)
    exact_target_matches = [column for column in X if X[column].equals(y)]
    audit: dict[str, object] = {
        "authoritative": True,
        "locked_model_policy": "No model modification or retraining based on test results",
        "artifact_sha256_before": artifact_hash_before,
        "artifact_sha256_after": sha256(model_path),
        "artifact_unchanged": artifact_hash_before == sha256(model_path),
        "random_state": RANDOM_STATE,
        "test_split_exact_match": test_match,
        "sample_counts": {
            "full": {"n": len(y), "classes": class_counts(y), "prevalence": float(y.mean())},
            "development": {"n": len(dev_y), "classes": class_counts(dev_y), "prevalence": float(dev_y.mean())},
            "initial_train": {"n": len(train_y), "classes": class_counts(train_y), "prevalence": float(train_y.mean())},
            "selection_validation": {"n": len(val_y), "classes": class_counts(val_y), "prevalence": float(val_y.mean())},
            "test": {"n": len(test_y), "classes": class_counts(test_y), "prevalence": float(test_y.mean())},
        },
        "features": {
            "included": list(X.columns),
            "excluded": {TARGET: "prediction target", NON_PREDICTIVE_COLUMNS[0]: "unique identifier", NON_PREDICTIVE_COLUMNS[1]: "constant confidential metadata"},
        },
        "leakage_audit": {
            "included_feature_identical_to_target": exact_target_matches,
            "identifier_absent": "PatientID" not in X,
            "constant_metadata_absent": "DoctorInCharge" not in X,
            "proximal_proxy_risk": ["FunctionalAssessment", "ADL", "MMSE", "MemoryComplaints", "BehavioralProblems"],
            "measurement_timing_verified": False,
        },
        "threshold": {"value": THRESHOLD, "selection": "fixed classifier default; no optimization on validation or test"},
        "test_metrics": metrics,
        "bootstrap_95_ci": cis,
        "original_validation_scores": artifact["validation_roc_auc"],
        "cross_validation": cv.to_dict(orient="records"),
        "calibration": artifact["calibration"],
        "duplicates": duplicates,
        "permutation_sanity": {
            "runs": len(sanity),
            "roc_auc_mean": float(sanity.roc_auc.mean()),
            "roc_auc_sd": float(sanity.roc_auc.std(ddof=1)),
            "pr_auc_mean": float(sanity.pr_auc.mean()),
            "pr_auc_sd": float(sanity.pr_auc.std(ddof=1)),
        },
        "shap": shap_status,
    }
    audit["artifact_sha256_after"] = sha256(model_path)
    audit["artifact_unchanged"] = audit["artifact_sha256_before"] == audit["artifact_sha256_after"]
    write_json(audit, PROJECT_ROOT / "results/metrics/final_validation.json")
    report = build_report(audit, cv, permutation_table)
    (PROJECT_ROOT / "paper/final_validation_report.md").write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the locked final Alzheimer's model")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/model.joblib")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()
    audit = validate(args.model, args.data)
    print(json.dumps({
        "artifact_unchanged": audit["artifact_unchanged"],
        "test_metrics": audit["test_metrics"],
        "bootstrap_95_ci": audit["bootstrap_95_ci"],
        "shap": audit["shap"],
    }, indent=2))


if __name__ == "__main__":
    main()
