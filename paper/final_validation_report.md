# Final Locked-Model Validation Report

**Status:** authoritative internal validation report  
**Model artifact SHA-256:** `6d6b1ad1274d917d4cd4a169412ba369e4848505729d6124788a8768a86e6cd7`  
**Audit rule:** the saved final model was not retrained, tuned, recalibrated, or replaced in response to test results.

## 1. Samples and class distribution

- Full dataset: 2149 observations; class 0 = 1389, class 1 = 760.
- Development set: 1719 observations; class 0 = 1111, class 1 = 608.
- Initial model-fit subset: 1375 observations; class 0 = 889, class 1 = 486.
- Internal model-selection validation subset: 344 observations; class 0 = 222, class 1 = 122.
- Locked test set: 430 observations; class 0 = 278, class 1 = 152.

## 2. Test-set isolation and provenance

The source code first created a stratified 80/20 development/test split with seed 42. Candidate preprocessing pipelines were fitted only on the initial training subset; model selection used only the internal validation subset. The selected model was then refit and five-fold sigmoid-calibrated only on the combined development set. The test set was stored for final evaluation and was not supplied to preprocessing fitting, feature selection, model selection, hyperparameter selection, calibration, or threshold selection. The audit reproduced the split and matched the stored test features and labels exactly: **True**. No data-driven threshold search occurred; `CalibratedClassifierCV.predict` uses the fixed binary cutoff **0.5**.

Important scope note: the candidate configurations were fixed in code rather than selected by a nested hyperparameter search. The test set has now been inspected for final reporting, so it must remain locked and cannot support later model changes.

## 3. Included and excluded features

**Included (32):** Age, Gender, Ethnicity, EducationLevel, BMI, Smoking, AlcoholConsumption, PhysicalActivity, DietQuality, SleepQuality, FamilyHistoryAlzheimers, CardiovascularDisease, Diabetes, Depression, HeadInjury, Hypertension, SystolicBP, DiastolicBP, CholesterolTotal, CholesterolLDL, CholesterolHDL, CholesterolTriglycerides, MMSE, FunctionalAssessment, MemoryComplaints, BehavioralProblems, ADL, Confusion, Disorientation, PersonalityChanges, DifficultyCompletingTasks, Forgetfulness.

**Excluded:**
- `Diagnosis`: prediction target; excluded from predictors.
- `PatientID`: unique record identifier; excluded to prevent memorization and identifier leakage.
- `DoctorInCharge`: constant confidential metadata (`XXXConfid`); excluded as non-predictive metadata.

## 4. Target-leakage audit

No included column is identical to `Diagnosis`, no feature name explicitly states diagnosis, and the preprocessing pipeline receives only the listed predictor columns. `PatientID` and `DoctorInCharge` are absent from the artifact's feature list. Several included variables—especially `FunctionalAssessment`, `ADL`, `MMSE`, `MemoryComplaints`, and `BehavioralProblems`—are clinically proximal to diagnostic assessment and may indirectly encode the synthetic generator's diagnostic rule. They are legitimate predictors for this benchmark but create a **construct/proxy leakage risk** and may inflate apparent performance. Because provenance and measurement timing are unavailable, the audit cannot prove that every predictor was measured before diagnosis. This blocks causal or prospective clinical claims.

## 5. Locked test performance

At threshold 0.5:

- ROC AUC: **0.941** (95% bootstrap CI 0.909–0.969)
- PR AUC / average precision: **0.922** (95% bootstrap CI 0.873–0.962); appropriate because positive prevalence is 0.353
- Accuracy: **0.951**
- Sensitivity: **0.921**
- Specificity: **0.968**
- Precision / PPV: **0.940**
- Negative predictive value: **0.957**
- F1: **0.930**
- Brier score: **0.053** (95% bootstrap CI 0.037–0.072)
- Log loss: **0.229**
- Expected calibration error: **0.050** (10 equal-width bins; descriptive and bin-dependent)
- Confusion matrix: TN = 269, FP = 9, FN = 12, TP = 140.

## 6. Threshold

The selected threshold is **0.5**, inherited from the classifier's default binary decision rule. It was not optimized on validation or test data. Therefore, no claim is made that 0.5 is clinically optimal. A clinical threshold would require a prespecified use case and explicit harm/benefit trade-offs in external data.

## 7–9. Cross-validation, discrepancy, and candidate models

Five-fold stratified CV on the development set, using training-fold-only preprocessing, produced:

- **logistic_regression:** ROC AUC 0.903 ± 0.014; PR AUC 0.846 ± 0.033; accuracy 0.823 ± 0.020; sensitivity 0.814 ± 0.033; precision 0.724 ± 0.044; F1 0.765 ± 0.015; Brier score 0.123 ± 0.015.
- **random_forest:** ROC AUC 0.955 ± 0.011; PR AUC 0.935 ± 0.020; accuracy 0.952 ± 0.009; sensitivity 0.913 ± 0.024; precision 0.950 ± 0.026; F1 0.930 ± 0.012; Brier score 0.087 ± 0.005.

The selected random forest's CV ROC AUC and locked test ROC AUC differ by **0.014**. This is not a meaningful collapse; the test value lies within the observed uncertainty and is slightly more conservative. The original one-shot internal validation ROC AUC values were {'logistic_regression': 0.895030276177817, 'random_forest': 0.9496012405848472} and selected random forest. The CV table is the more stable candidate comparison; it does not alter the locked model.

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

Across 25 independently shuffled development targets, the same selected model specification achieved mean test ROC AUC **0.508 ± 0.056** and mean PR AUC **0.369 ± 0.040**. Performance collapsed toward chance/baseline, supporting that the real-label result is not explained by the pipeline producing high scores under random labels. These sanity models are separate from and did not modify the final artifact.

## 13. Duplicate and near-duplicate audit

- Exact predictor duplicates crossing development/test: **0**.
- Near duplicates crossing development/test under the prespecified normalized-distance rule: **0**.
- Minimum cross-split normalized distance: **0.0825**.
- Median nearest cross-split normalized distance: **0.1476**.

The near-duplicate rule is mean absolute distance ≤0.01 after min–max scaling all included features. This is a transparent heuristic, not proof of patient independence; the dataset is synthetic and provides no entity-linkage metadata.

## 14. Internal performance versus clinical validity

These are **internal predictive-performance estimates on a synthetic dataset**, not evidence of clinical validity. The audit verifies software-level split isolation and metrics. It does not establish representative sampling, temporal validity, transportability, fairness, clinical utility, safety, or benefit to patients. External validation on provenance-audited real clinical cohorts, with predictor timing defined before the diagnostic decision, is mandatory before any clinical interpretation.

## Reproducibility artifacts

- Machine-readable audit: `results/metrics/final_validation.json`
- Candidate CV table: `results/tables/candidate_cross_validation.csv`
- Permutation sanity table: `results/tables/permutation_sanity.csv`
- Global permutation importance: `results/tables/permutation_importance.csv`
- Locked artifact hash unchanged after audit: **True**
