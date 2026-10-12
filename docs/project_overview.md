# CreditLens — Project Overview

**Full title:** CreditLens — Explainable AI Credit Risk Assessment & Decision Support Platform
**Type:** Final-year university project (AI/ML + Accounting & Finance)
**Status:** Phase 1 complete (project skeleton). Phases 2–8 pending.

## Purpose
Predict the probability that a credit-card customer will default on their
next-month payment, and present the result with model comparison,
explainability, what-if simulation, portfolio analytics, fairness auditing,
and downloadable reports.

## Dataset
UCI Machine Learning Repository — *Default of Credit Card Clients*
(30,000 records, 23 input features, 1 binary target).
Target column: `default.payment.next.month` (1 = default next month).

## Risk bands (analytical thresholds, not regulatory)
| Band | Probability range |
|---|---|
| Low | 0.00 – 0.20 |
| Moderate | 0.20 – 0.50 |
| High | 0.50 – 0.80 |
| Very High | 0.80 – 1.00 |

## Phase 1 scope (this document)
- Reproducible Python environment.
- Project folder structure.
- Central path resolution (`src/utils/paths.py`).
- Central settings (`config/settings.py`).
- Logging configuration (`src/utils/logging_config.py`).
- Smoke tests (`tests/test_environment.py`).
- Verification dashboard (`app.py`).

## Non-goals for Phase 1
- No dataset loading, training, prediction, SHAP, or analytics yet.
- No Streamlit pages beyond the verification screen.

## Limitations acknowledged
- Dataset population is historical and geographically limited; results are
  not expected to generalise to other populations or time periods.
- The system is an academic decision-support prototype and must not be used
  to approve or reject real applicants.
- SHAP explains model behaviour, not causality.