# Manuscript Evidence and Citation Verification Report

**Manuscript:** `paper/manuscript.md`  
**Verification date:** 2026-09-11  
**Status:** Passed with scope caveats stated below

## 1. Verification scope

The manuscript was prepared using a staged research-writing workflow: literature collection, source screening, synthesis, drafting, peer-style claim review, and citation verification. The five supplied NotebookLM notebooks were queried for research guidance:

- `52ccf200-6b40-408c-a305-796684a1b202`
- `19e4113f-6acc-4859-bc1f-dd5760981324`
- `991ff8c6-f499-4cd8-b561-23db785c5598`
- `db3e5834-377b-42cc-be52-7da706063b28`
- `5370ff4c-2644-4359-9fab-fae60863522a`

All five queries succeeded. NotebookLM output was treated as discovery and synthesis guidance only. It was not treated as authoritative bibliographic evidence, and private notebook citations do not appear in the manuscript.

Candidate references were retained only when their identity and relevant claim scope could be independently checked against at least one authoritative source: a publisher article page, PubMed/PMC, official conference proceedings, JMLR, or the primary dataset page/DOI. Notebook-only summaries, uncertain 2026 works, and sources with incomplete or mismatched metadata were excluded.

## 2. Empirical-source lock

No experiment, retraining, tuning, recalibration, threshold optimization, or empirical-output modification was performed during manuscript writing. Empirical claims were checked only against:

- `src/data.py`
- `src/preprocessing.py`
- `src/train.py`
- `src/evaluate.py`
- `src/explain.py`
- `src/shap_report.py`
- `src/validate.py`
- `results/metrics/final_validation.json`
- `results/metrics/shap_metadata.json`
- `results/tables/candidate_cross_validation.csv`
- `results/tables/classification_report.csv`
- `results/tables/dataset_summary.csv`
- `results/tables/feature_importance.csv`
- `results/tables/permutation_importance.csv`
- `results/tables/permutation_sanity.csv`
- `results/tables/shap_importance.csv`
- `paper/final_validation_report.md`
- Existing figures under `results/figures/`

The authoritative model artifact hash reported by the project is `6d6b1ad1274d917d4cd4a169412ba369e4848505729d6124788a8768a86e6cd7`, unchanged before and after the existing final validation.

## 3. Empirical consistency checklist

| Required fact | Manuscript value | Project evidence | Status |
|---|---|---|---|
| Full sample | 2,149 | `final_validation.json`; `dataset_summary.csv` | Verified |
| Positives | 760 (35.4%) | `final_validation.json`; `dataset_summary.csv` | Verified |
| Development/test | 1,719 / 430 | `final_validation.json` | Verified |
| Candidates | Logistic regression; random forest only | `src/train.py`; `src/validate.py` | Verified |
| CV ROC-AUC | RF 0.955 ± 0.011; LR 0.903 ± 0.014 | `candidate_cross_validation.csv` | Verified |
| Final model | Five-fold sigmoid-calibrated random forest | `src/train.py`; `final_validation.json` | Verified |
| Test ROC-AUC | 0.9408 (0.9093–0.9694) | `final_validation.json` | Verified |
| Test PR-AUC | 0.9222 (0.8732–0.9624) | `final_validation.json` | Verified |
| Accuracy | 0.9512 | `final_validation.json` | Verified |
| Sensitivity/specificity | 0.9211 / 0.9676 | `final_validation.json` | Verified |
| Precision/NPV/F1 | 0.9396 / 0.9573 / 0.9302 | `final_validation.json` | Verified |
| Brier score | 0.0532 (0.0366–0.0717) | `final_validation.json` | Verified |
| Log loss/ECE | 0.2293 / 0.0499 | `final_validation.json` | Verified |
| Confusion matrix | TN 269; FP 9; FN 12; TP 140 | `final_validation.json` | Verified |
| Threshold | 0.5; not optimized | `src/validate.py`; `final_validation.json` | Verified |
| SHAP top five | FunctionalAssessment, ADL, MemoryComplaints, MMSE, BehavioralProblems | `shap_importance.csv` | Verified |
| Excluded metadata | PatientID; DoctorInCharge | `src/data.py`; `final_validation.json` | Verified |
| Duplicate audit | No source predictor duplicates; no exact or near cross-split duplicates | `final_validation.json` | Verified |
| Target shuffle | ROC-AUC 0.508 ± 0.056 | `final_validation.json`; `permutation_sanity.csv` | Verified |
| Test isolation | Excluded from selection, preprocessing fitting, calibration, tuning, threshold selection | `src/train.py`; `src/validate.py`; validation report | Verified |
| Hyperparameter optimization | None | Fixed candidate constructors in `src/train.py`; validation report | Verified |
| SHAP model scope | Underlying uncalibrated development-refit RF, not calibration layer | `src/shap_report.py`; `shap_metadata.json`; validation report | Verified |

## 4. Citation-by-citation verification

| Ref. | Reference | Authoritative verification page | Verified claim scope | Status |
|---:|---|---|---|---|
| 1 | McKhann et al., 2011 | [PubMed PMID 21514250](https://pubmed.ncbi.nlm.nih.gov/21514250/) and DOI `10.1016/j.jalz.2011.03.005` | Clinical diagnostic framing; cognition and function in dementia assessment | Verified |
| 2 | Folstein et al., 1975 | [PubMed PMID 1202204](https://pubmed.ncbi.nlm.nih.gov/1202204/) and [Elsevier landing page](https://www.sciencedirect.com/science/article/abs/pii/0022395675900266) | Origin and purpose of MMSE | Verified |
| 3 | Jo et al., 2019 | [Frontiers/DOI landing page](https://doi.org/10.3389/fnagi.2019.00220) | Alzheimer ML review focused on neuroimaging classification and prediction | Verified |
| 4 | Collins et al., 2024 | [BMJ article](https://www.bmj.com/content/385/bmj-2023-078378) | Reporting guidance for regression and ML prediction models | Verified |
| 5 | Moons et al., 2025 | [BMJ article](https://www.bmj.com/content/388/bmj-2024-082505) | Prediction-model quality, risk-of-bias, and applicability assessment | Verified |
| 6 | Andaur Navarro et al., 2021 | [BMJ article](https://www.bmj.com/content/375/bmj.n2281) | High risk of bias and common analysis limitations in supervised-ML prediction studies | Verified |
| 7 | Collins et al., 2014 | [PubMed Central article](https://pmc.ncbi.nlm.nih.gov/articles/PMC3999945/) | Conduct and reporting of external validation; independent-evaluation context | Verified |
| 8 | Van Calster et al. and the STRATOS Topic Group, 2019 | [BMC Medicine/Springer article](https://link.springer.com/article/10.1186/s12916-019-1466-7) | Calibration definition, importance, assessment, and updating | Verified |
| 9 | Saito and Rehmsmeier, 2015 | [PLOS ONE article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) | Precision–recall analysis for imbalanced binary classification | Verified |
| 10 | Breiman, 2001 | [Springer article](https://link.springer.com/article/10.1023/A:1010933404324) | Random-forest method | Verified |
| 11 | Niculescu-Mizil and Caruana, 2005 | [ACM Digital Library](https://dl.acm.org/doi/10.1145/1102351.1102430) and [ICML proceedings PDF](https://icml.cc/Conferences/2005/proceedings/papers/079_GoodProbabilities_NiculescuMizilCaruana.pdf) | Supervised-classifier probability calibration, including sigmoid/Platt-style calibration | Verified |
| 12 | Lundberg and Lee, 2017 | [Official NeurIPS proceedings](https://papers.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) | SHAP additive feature-attribution framework | Verified |
| 13 | Lundberg et al., 2020 | [Nature Machine Intelligence article](https://www.nature.com/articles/s42256-019-0138-9) | Tree-specific SHAP and aggregation from local to global explanations | Verified |
| 14 | Kapoor and Narayanan, 2023 | [Patterns/Cell Press article](https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9) | Leakage taxonomy and performance inflation/reproducibility risk | Verified |
| 15 | Chen et al., 2021 | [Nature Biomedical Engineering article](https://www.nature.com/articles/s41551-021-00751-8) | Opportunities and limitations of synthetic medical data | Verified |
| 16 | Pedregosa et al., 2011 | [JMLR article](https://jmlr.org/papers/v12/pedregosa11a.html) | scikit-learn software framework | Verified |
| 17 | El Kharoua, 2024 | [Kaggle dataset page](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset) and DOI `10.34740/KAGGLE/DSV/8668279` | Dataset identity, authorship, synthetic educational source | Verified primary dataset source; not peer reviewed |

## 5. Screening exclusions

The following classes of candidate sources were not used:

- private NotebookLM synthesis documents;
- references returned with incomplete authorship, venue, DOI, or publication status;
- unverified 2026 preprints and articles when established primary sources were available;
- secondary web summaries when an authoritative publisher or index page was available;
- claims requiring experiments not present in the locked project outputs;
- literature suggesting models not evaluated by this project, where inclusion could imply an unperformed comparison.

## 6. Peer-style claim review

The draft was checked for the following prohibited overclaims:

- **No clinical-validity claim:** the manuscript repeatedly identifies the data as synthetic and the evaluation as internal.
- **No broad algorithm-superiority claim:** random forest is described as outperforming logistic regression only within the prespecified two-model candidate set.
- **No untested-model claim:** no additional algorithm is described as evaluated.
- **No causal SHAP claim:** SHAP is described as attribution for the fitted underlying random forest, not causality.
- **No calibration-layer explanation claim:** SHAP is explicitly separated from the sigmoid calibration maps and calibrated ensemble probability.
- **No threshold-optimality claim:** 0.5 is described as fixed and non-optimized, not clinically optimal.
- **No external-validation claim:** the locked test set is correctly identified as internal because it comes from the same synthetic source.
- **No post-test model-change recommendation using the current test set:** development-only ablation is proposed; any revised model requires a new untouched test set.

## 7. Caveats

1. Verification establishes bibliographic identity and appropriate high-level claim scope; it is not a formal systematic review.
2. The dataset reference is a primary repository/data-card citation rather than a peer-reviewed article.
3. The empirical audit relies on the locked project outputs and source code supplied in the repository. Manuscript preparation did not regenerate those outputs.
4. The calibration curve uses ten quantile bins, whereas ECE uses ten equal-width bins; the manuscript preserves this distinction.
5. SHAP and permutation importance use locked-test observations post hoc. They are descriptive audits, not model-development evidence or causal analyses.
