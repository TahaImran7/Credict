# Credict — Project Overview

**Full title:** Credict — Explainable AI Credit Risk Assessment &
Decision Support Platform

**Type:** Semester project (AI/ML + Accounting & Finance)

**Status:** Complete — all implementation phases finished and verified

**Last updated:** 29 - October - 2026

---

## Purpose

Predict the probability that a credit-card customer will default on their
next-month payment, and present the result with model comparison,
explainability, what-if simulation, portfolio analytics, fairness
auditing, and downloadable reports.

Credict is a **decision-support tool** for credit-risk analysis. It is
not a lending authority and does not approve or reject real applicants.

---

## Problem

Financial institutions need systematic methods to identify customers who
may fail to make required payments. Black-box AI models produce risk
estimates that are difficult for analysts, auditors, and regulators to
interpret. Credict addresses this by combining predictive accuracy with
SHAP-based per-prediction explanations, fairness auditing, and scenario
analysis.

---

## Dataset

**Source:** UCI Machine Learning Repository — *Default of Credit Card
Clients* (Yeh & Lien, 2009)

- 30,000 records from a Taiwanese bank (April–September 2005)
- 23 input features (credit limit, demographics, repayment history, bill
  amounts, payment amounts)
- 1 binary target: `default.payment.next.month` (1 = default next month,
  0 = no default)
- Class distribution: 22.12% positive (6,636 defaults), 77.88% negative
  (23,364 non-defaults) — imbalanced

**Data split:** Stratified 80/20 (24,000 train / 6,000 test), random seed
42.

---

## Machine Learning Approach

Two models trained and compared:

| Model | Role | Selection rationale |
|---|---|---|
| **Logistic Regression** | Interpretable baseline | Simple, well-established credit scoring method |
| **Random Forest** | Advanced model | Nonlinear pattern capture, ensemble robustness |

**Hyperparameter tuning:** 5-fold cross-validation on the training set
(ROC-AUC scoring). Small grids (6 candidates for LR, 4 for RF) to avoid
overfitting the selection process.

**Model selection rule:** Primary metric PR-AUC (appropriate for imbalanced
data), tie-break ROC-AUC.

**Probability calibration:** The Random Forest is trained with
`class_weight='balanced'`, which improves recall but inflates predicted
probabilities (mean predicted ~0.35 vs observed base rate ~0.22). Isotonic
regression is fitted on 5-fold out-of-fold training predictions to correct
this. The calibrated model is deployed for prediction; the raw model
remains for SHAP explanations (tree explainer requires raw classifier).

---

## Results (Held-Out Test Set — n = 6,000)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.7740 | 0.4903 | 0.5546 | 0.5205 | 0.7590 | 0.5267 | 0.1864 |
| Random Forest (raw) | 0.7970 | 0.5414 | 0.5365 | 0.5390 | 0.7740 | 0.5523 | 0.1536 |
| **Random Forest (calibrated) — deployed** | **0.7950** | **0.5358** | **0.5463** | **0.5410** | **0.7732** | **0.5391** | **0.1361** |

**Benchmark context:** Published accuracy on this dataset ranges 79–83%
for state-of-the-art methods. Credict sits within this range, with
substantially better explainability.

**Calibration impact:** Brier score improved 11.4% (0.1536 → 0.1361).
ROC-AUC and PR-AUC dropped marginally due to ties in the isotonic step
function — a documented trade-off. F1 and recall improved slightly.

---

## Risk Bands (Analytical Thresholds)

Not regulatory classifications — analytical choices documented in
`config/settings.py`.

| Band | Probability range |
|---|---|
| Low | 0.00 – 0.20 |
| Moderate | 0.20 – 0.50 |
| High | 0.50 – 0.80 |
| Very High | 0.80 – 1.00 |

**Observed default rate per band** (test set): 7.6% / 16.4% / 46.9% /
73.1% — monotone increase, confirming the bands are informative.

---

## Application Features (8 UI Pages)

1. **Overview** — portfolio-level KPIs, risk distribution, model summary
2. **Risk Assessment** — single-customer prediction with SHAP explanation
3. **What-if Simulator** — scenario analysis (change features, compare
   risk estimates)
4. **Model Performance** — ROC, PR, calibration curves, confusion matrices,
   metric interpretation
5. **Explainability** — global feature importance, individual SHAP
   waterfall
6. **Portfolio Analytics** — customer distributions, default rates by
   category, financial patterns
7. **Fairness Audit** — group-level metrics across SEX, EDUCATION,
   MARRIAGE, AGE_BAND
8. **Reports** — downloadable CSV/JSON exports for all analysis types

**UI characteristics:**
- Professional design system (light and dark themes, both switchable in-app)
- No emoji anywhere
- Plotly interactive charts
- Local-only execution (no cloud services)

---

## Explainability (SHAP)

**Method:** TreeExplainer on the raw Random Forest classifier. SHAP values
are aggregated from expanded (one-hot) feature names back to the original
23 feature names for interpretability.

**Global importance:** `PAY_0` (most recent repayment status) dominates,
consistent with credit-risk literature.

**Per-prediction explanations:** Every prediction in Risk Assessment and
Explainability tabs shows the top contributing features with sign and
magnitude.

**Caveat:** SHAP describes model behaviour, not causality. A feature
contributing +0.05 to the predicted risk does not cause a customer to
default — it indicates the model's reasoning.

---

## Fairness Audit

Group-level metrics computed for **SEX, EDUCATION, MARRIAGE, AGE_BAND**
across the held-out test set:

- True positive rate (TPR)
- False positive rate (FPR)
- Precision, NPV, F1, accuracy
- ROC-AUC per group
- Differences from a reference group (largest by sample size)

**Status:** Descriptive audit only. Does not certify compliance with any
fairness standard. Groups with fewer than 30 records are flagged as
low-confidence.

---

## Technical Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Data | pandas, numpy, pyarrow |
| ML | scikit-learn 1.5+ |
| Explainability | SHAP 0.45+ |
| Dashboard | Streamlit 1.32+ |
| Charts | Plotly 5.18+ |
| Testing | pytest 8.0+ |

**No cloud services, no paid APIs, no external AI services.** All
computation is local.

---

## Project Structure

Credict/
├── app.py # Streamlit entry point
├── config/settings.py # Central configuration
├── src/ # Core logic (UI-independent)
│ ├── data/ # Loading, validation, preprocessing
│ ├── training/ # Model training, calibration, evaluation
│ ├── prediction/ # Prediction interface
│ ├── explainability/ # SHAP explanations
│ ├── simulation/ # What-if analysis
│ ├── analytics/ # Portfolio analytics
│ ├── fairness/ # Fairness audit
│ ├── reporting/ # Report generation
│ └── utils/ # Paths, logging
├── ui/ # Streamlit pages
├── tests/ # 147 automated tests
├── scripts/ # One-shot pipeline scripts
├── models/ # Saved models + evaluation artefacts
├── data/ # Raw + processed datasets
└── docs/ # Documentation


---

## Verification & Testing

- **147 automated tests** passing (unit, integration, end-to-end)
- **Manual UI verification** across all 8 pages in both themes
- **Reproducible** from random seed 42 — re-running scripts produces
  identical results
- **No data leakage** — preprocessor fitted on training data only; test
  set never used for tuning or selection
- **Cross-validation stability** — fold-to-fold std in ROC-AUC: 0.0005

See `docs/self_audit.md` for detailed evidence.

---

## Documentation Index

| Document | Purpose |
|---|---|
| `docs/project_overview.md` | This file — high-level project summary |
| `docs/secondary_research.md` | Problem validation from academic and regulatory sources |
| `docs/methodology.md` | Machine learning methodology (detailed) |
| `docs/testing.md` | Test evidence summary |
| `docs/limitations.md` | Consolidated honest limitations |
| `docs/self_audit.md` | Verification evidence |
| `docs/feasibility_analysis.md` | Financial and market feasibility |
| `docs/pitch_deck.md` | Pitch deck content and speaker notes |
| `docs/qa_prep.md` | Likely judge questions with prepared answers |
| `docs/demo_script.md` | 5-minute walkthrough script |
| `docs/cv_entry.md` | Verified CV-ready project entry |
| `README.md` | Setup, installation, running, troubleshooting |

---

## Honest Limitations

1. **Dataset is historical and geographically limited.** UCI data is from
   a Taiwanese bank in 2005. Results may not generalise to current
   Pakistani customers or other populations.

2. **Academic decision-support prototype.** Must not be used to approve or
   reject real applicants.

3. **SHAP describes model behaviour, not causality.**

4. **Fairness audit is descriptive only.** Does not certify legal
   compliance or prove discrimination in any direction.

5. **PR-AUC dropped slightly after calibration** (0.552 → 0.539) due to
   ties introduced by isotonic regression — an expected trade-off for
   improved probability quality.

6. **No regulatory approval.** SBP compliance for use in actual lending
   decisions would require a multi-year process. Credict is positioned
   as decision-support, not decision-making.

7. **No user-facing validation interviews.** Secondary research and
   desk-based market analysis were conducted instead (see
   `docs/secondary_research.md`) — appropriate for a semester-scope
   project.

8. **Model drift not monitored.** Deployed model reflects 2005 training
   data. Production deployment would require quarterly retraining and
   monitoring.

---

## Reproducibility Commands

From the project root with the virtual environment activated:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
python -c "from src.data.loader import download_dataset; download_dataset()"

# 3. Prepare data (split, fit preprocessor)
python -m scripts.prepare_data

# 4. Train models (LR, RF, calibrated RF)
python -m scripts.train_models

# 5. Compute dashboard artefacts (SHAP importance, test predictions)
python -m scripts.build_dashboard_artifacts

# 6. Run fairness audit
python -m scripts.run_fairness_audit

# 7. Launch the application
streamlit run app.py

# 8. Run tests (optional — verifies setup)
pytest tests -v