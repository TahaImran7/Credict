# Credict

**Explainable AI Credit Risk Assessment & Decision Support Platform**

Predicts next-month credit-card default probability with per-prediction
SHAP explanations, what-if scenario simulation, model comparison,
portfolio analytics, and fairness auditing.

- Runs entirely **locally** — no cloud, no external APIs, no data leaves
  your machine
- **8-page** Streamlit dashboard with light and dark themes
- **147 automated tests** passing
- Reproducible from `RANDOM_SEED = 42`

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Requirements](#requirements)
3. [Project Structure](#project-structure)
4. [Application Pages](#application-pages)
5. [Running Tests](#running-tests)
6. [Dataset](#dataset)
7. [Model Summary](#model-summary)
8. [Troubleshooting](#troubleshooting)
9. [Reproducibility](#reproducibility)
10. [Known Limitations](#known-limitations)
11. [Documentation](#documentation)
12. [Licence](#licence)

---

## Quick Start

```bash
# 1. Clone / extract this project, then:
python -m venv .Credict_venv

# Windows:
.Credict_venv\Scripts\activate

# macOS / Linux:
source .Credict_venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download the UCI dataset
python -c "from src.data.loader import download_dataset; download_dataset()"

# 4. Prepare the data (split, preprocessor)
python -m scripts.prepare_data

# 5. Train models (LR, RF, calibrated RF) — takes 3–5 minutes
python -m scripts.train_models

# 6. Build dashboard artefacts (SHAP importance, test predictions)
python -m scripts.build_dashboard_artifacts

# 7. Run fairness audit
python -m scripts.run_fairness_audit

# 8. Launch the app
streamlit run app.py
```

Open <http://localhost:8501>. Hard-refresh (`Ctrl+Shift+R` on Windows/Linux,
`Cmd+Shift+R` on macOS) on first load to clear any cached CSS.

---

## Requirements

- **Python:** 3.10, 3.11, or 3.12
- **RAM:** 4 GB minimum (8 GB recommended for the Random Forest and SHAP)
- **Disk:** ~500 MB (including models and dataset)
- **OS:** Windows 10+, macOS 12+, or any modern Linux
- **Internet:** required only for dependency installation and the dataset
  download. Not required at runtime.

**Tested on:** Windows 11, Python 3.12.10, `.Credict_venv`.

---

## Project Structure

```
Credict/
├── app.py                          # Streamlit entry point
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml                 # Streamlit theme configuration
├── config/
│   ├── __init__.py
│   └── settings.py                 # Single source of truth
├── data/
│   ├── raw/                        # Original UCI .xls file
│   └── processed/                  # Parquet splits
├── models/
│   ├── saved/                      # Joblib artefacts
│   │   ├── preprocessor.joblib
│   │   ├── logistic_regression.joblib
│   │   ├── random_forest.joblib
│   │   └── random_forest_calibrated.joblib
│   └── evaluation/                 # JSON + CSV evaluation artefacts
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py               # Dataset download + loading
│   │   ├── validation.py           # Data validation
│   │   └── preprocessing.py        # Cleaning, splitting, ColumnTransformer
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                # Model training + CV tuning
│   │   ├── calibrate.py            # Isotonic calibration
│   │   ├── evaluate.py             # Metrics, curves, calibration data
│   │   └── model_comparison.py     # Comparison + selection rules
│   ├── prediction/
│   │   ├── __init__.py
│   │   └── predictor.py            # Prediction interface
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── shap_explainer.py       # SHAP-based explanations
│   ├── simulation/
│   │   ├── __init__.py
│   │   └── what_if.py              # What-if scenario simulation
│   ├── analytics/
│   │   ├── __init__.py
│   │   └── portfolio.py            # Portfolio analytics
│   ├── fairness/
│   │   ├── __init__.py
│   │   └── audit.py                # Fairness audit
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── report_generator.py     # CSV/JSON export
│   └── utils/
│       ├── __init__.py
│       ├── paths.py                # Central path resolution
│       └── logging_config.py       # Logging setup
├── ui/
│   ├── __init__.py
│   ├── styles.py                   # Design system (light + dark)
│   ├── components.py               # Reusable components
│   ├── charts.py                   # Plotly chart builders
│   ├── _cache.py                   # Cached data / model loading
│   ├── overview.py
│   ├── risk_assessment.py
│   ├── simulator.py
│   ├── model_performance.py
│   ├── explainability.py
│   ├── portfolio_analytics.py
│   ├── fairness_audit.py
│   └── reports.py
├── scripts/
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── train_models.py
│   ├── build_dashboard_artifacts.py
│   └── run_fairness_audit.py
├── tests/
│   ├── __init__.py
│   ├── test_environment.py
│   ├── test_data_loading.py
│   ├── test_preprocessing.py
│   ├── test_training.py
│   ├── test_calibration.py
│   ├── test_prediction.py
│   ├── test_explainability.py
│   ├── test_simulation.py
│   ├── test_portfolio.py
│   ├── test_fairness.py
│   ├── test_reporting.py
│   └── test_ui_imports.py
└── docs/
    ├── project_overview.md
    ├── secondary_research.md
    ├── methodology.md
    ├── testing.md
    ├── limitations.md
    ├── self_audit.md
    ├── feasibility_analysis.md
    ├── pitch_deck.md
    ├── qa_prep.md
    ├── demo_script.md
    └── cv_entry.md
```

---

## Application Pages

| Page | What it does |
|---|---|
| **Overview** | Portfolio KPIs, risk distribution, model comparison |
| **Risk Assessment** | Single-customer prediction with SHAP breakdown |
| **What-if Simulator** | Scenario analysis — change features, compare risk |
| **Model Performance** | ROC, PR, calibration curves, confusion matrices |
| **Explainability** | Global feature importance + individual explanations |
| **Portfolio Analytics** | Customer distributions, default rates by category |
| **Fairness Audit** | Group-level metrics across SEX, EDUCATION, MARRIAGE, AGE |
| **Reports** | CSV/JSON exports for all analysis types |

Toggle between **light and dark** themes via the sidebar.

---

## Running Tests

```bash
pytest tests -v
```

**Expected:** ~147 passed in ~60 seconds. Some tests are skipped on a
fresh checkout (models not trained); they run once
`scripts/train_models.py` has been executed.

Test categories:

- `test_data_loading.py` — dataset loading, validation
- `test_preprocessing.py` — cleaning, splits, unseen categories
- `test_training.py` — pipelines, evaluation, comparison, persistence
- `test_calibration.py` — isotonic calibration, threshold mapping
- `test_prediction.py` — risk bands, record preparation, determinism
- `test_explainability.py` — SHAP shape, aggregation, consistency
- `test_simulation.py` — validation, no-mutation, multi-change scenarios
- `test_portfolio.py` — summaries, distributions, low-confidence flags
- `test_fairness.py` — group metrics, undefined handling
- `test_reporting.py` — JSON/CSV round-trips
- `test_ui_imports.py` — every UI module imports, palette, no emoji

---

## Dataset

**UCI Machine Learning Repository — Default of Credit Card Clients**

- Source: <https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients>
- 30,000 records, 23 input features, 1 binary target
- Class distribution: 22.12% positive, 77.88% negative (imbalanced)
- Data description: customers from a Taiwanese bank, April–September 2005

The dataset is downloaded automatically by `download_dataset()`. To place
it manually, save it as:

```
data/raw/default of credit card clients.xls
```

and skip step 3 of Quick Start.

---

## Model Summary

| Model | Accuracy | ROC-AUC | PR-AUC | Brier | Deployed? |
|---|---|---|---|---|---|
| Logistic Regression | 0.7740 | 0.7590 | 0.5267 | 0.1864 | No |
| Random Forest (raw) | 0.7970 | 0.7740 | 0.5523 | 0.1536 | No (SHAP only) |
| **Random Forest (calibrated)** | **0.7950** | **0.7732** | **0.5391** | **0.1361** | **Yes** |

**Selection rule:** Base model family chosen by PR-AUC (imbalanced data
needs PR-AUC), then calibrated variant preferred if Brier improves by ≥5%.

The calibrated model is deployed for prediction. The raw Random Forest is
retained for SHAP explanations — TreeExplainer requires the raw estimator,
not the isotonic wrapper.

**Calibration impact:**

- Brier score: 0.1536 → **0.1361** (−11.4% improvement)
- ROC-AUC: 0.7740 → 0.7732 (small drop due to isotonic ties)
- PR-AUC: 0.5523 → 0.5391 (small drop due to isotonic ties)
- Threshold: 0.50 → 0.31 (operating point preserved)

See `docs/methodology.md` §7 for full discussion of the calibration
trade-off.

---

## Troubleshooting

### `FileNotFoundError: Dataset not found`

Run step 3 of Quick Start:

```bash
python -c "from src.data.loader import download_dataset; download_dataset()"
```

### `FileNotFoundError: Model 'random_forest_calibrated' is missing`

Run step 5:

```bash
python -m scripts.train_models
```

### `FileNotFoundError: Fairness files missing`

Run step 7:

```bash
python -m scripts.run_fairness_audit
```

### Charts render but text is unreadable in dark mode

Hard-refresh the browser:

- Windows / Linux: `Ctrl+Shift+R`
- macOS: `Cmd+Shift+R`
- Or: open DevTools (`F12`) → right-click reload button → "Empty Cache
  and Hard Reload"

Streamlit caches CSS aggressively.

### `ImportError: pyarrow requires NumPy 2.0 or newer`

Your NumPy is too old. Reinstall from `requirements.txt`:

```bash
pip uninstall -y numpy pandas scikit-learn shap pyarrow
pip install -r requirements.txt
pip check
```

`pip check` should report `No broken requirements found.`

### Tests fail with `IndexError: index 1 is out of bounds`

You're likely using an old version of `tests/test_calibration.py`. Update
from the current version. The `_tiny_data` helper must produce two-class
`y` — the current version includes a guard test
(`test_tiny_data_has_both_classes`) that fails loudly if it doesn't.

### `streamlit: command not found`

The virtual environment is not activated. Activate it:

- Windows: `.Credict_venv\Scripts\activate`
- macOS/Linux: `source .Credict_venv/bin/activate`

### App runs but slow on first prediction

SHAP TreeExplainer initialization for a 200-tree Random Forest takes a few
seconds. Subsequent predictions are cached. This is expected — not a bug.

### Download buttons are invisible (white text on white background)

Fixed in the current `ui/styles.py`. If you still see this, you have stale
cached CSS. Hard-refresh the browser (see above). If it persists, ensure
`ui/styles.py` is the version from the current repository — the button
CSS block must be at the end of the stylesheet for cascade priority.

### Dark mode has black text on dark background in some places

Same root cause: stale CSS cache. Hard-refresh. If the issue persists after
hard-refresh, check that `.streamlit/config.toml` has `base = "light"` (our
custom themes are injected via CSS, not the Streamlit theme system) and
that `ui/styles.py` contains the full `_build_css` function.

---

## Reproducibility

All training runs use `RANDOM_SEED = 42`. Re-running from a clean state
produces identical results:

```bash
# Remove all artefacts
rm -rf models/saved models/evaluation data/processed

# Regenerate everything
python -m scripts.prepare_data
python -m scripts.train_models
python -m scripts.build_dashboard_artifacts
python -m scripts.run_fairness_audit
```

This should reproduce the exact metrics in the Model Summary table above.

**Verification:** run `pytest tests -v` after regeneration; the model-
dependent tests that were skipped should now activate and pass.

---

## Known Limitations

See `docs/limitations.md` for the complete list. Highlights:

- Dataset is historical (2005) and geographically limited (Taiwan)
- No model drift monitoring — production deployment needs retraining
- Fairness audit is descriptive, not a compliance certification
- Calibration improves probability quality at the cost of tiny ranking
  metric drops
- No live user validation interviews — secondary research only
- Not regulatory certified for use in actual lending decisions
- Single-user local application — no authentication, no multi-tenancy

---

## Documentation

| File | Purpose |
|---|---|
| `docs/project_overview.md` | High-level project summary |
| `docs/secondary_research.md` | Problem validation from literature and regulation |
| `docs/methodology.md` | ML methodology (detailed) |
| `docs/testing.md` | Test evidence summary |
| `docs/limitations.md` | Consolidated honest limitations |
| `docs/self_audit.md` | Verification evidence |
| `docs/feasibility_analysis.md` | Financial and market feasibility |
| `docs/pitch_deck.md` | Pitch deck content with speaker notes |
| `docs/qa_prep.md` | Q&A preparation (15 likely questions) |
| `docs/demo_script.md` | 5-minute walkthrough script |
| `docs/cv_entry.md` | CV-ready project entry |

---

## Technical Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.10–3.12 |
| Data processing | pandas | 2.2+ |
| Numerical | numpy | 2.0+ |
| Machine learning | scikit-learn | 1.5+ |
| Explainability | SHAP | 0.45+ |
| Dashboard | Streamlit | 1.32+ |
| Visualisation | Plotly | 5.18+ |
| Data persistence | pyarrow | 15.0+ |
| Excel reader | xlrd | 2.0+ |
| Testing | pytest | 8.0+ |

Install all pinned versions with `pip install -r requirements.txt`.

---

## Reproducible Build Order

The scripts have strict dependency ordering. Run them in this sequence:

```
1. prepare_data.py            → data/processed/, models/saved/preprocessor.joblib
2. train_models.py            → models/saved/*.joblib, models/evaluation/*_eval.json,
                                model_comparison.csv
3. build_dashboard_artifacts  → global_importance.csv, test_predictions.csv
4. run_fairness_audit         → fairness_metrics.csv, fairness_differences.csv,
                                fairness_report.json
```

Running `train_models.py` before `prepare_data.py` will fail with a clear
`FileNotFoundError`. Running `build_dashboard_artifacts.py` before
`train_models.py` will fail the same way.

---

## Academic Integrity

All metrics reported in this README and in `docs/` are from actual model
runs on the held-out test set. No results are fabricated or estimated.
Published benchmarks are cited with sources. Where estimates are used
(e.g., Pakistan credit card figures in `docs/feasibility_analysis.md`),
they are explicitly marked as estimates with a recommendation to verify
from primary regulatory sources.

---

## Licence

Academic project — for educational and demonstration use.

**Not licensed for production deployment** without additional hardening,
security audit, and regulatory review. Credict is a decision-support
prototype, not a lending authority. Predictions must not be used to
approve or reject real applicants.

---

## Contact

For questions about this project, open an issue on the project repository
or contact the author directly.