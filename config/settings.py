"""
CreditLens — central configuration.

Single source of truth for:
- Random seed
- Dataset URL, filename, and header row
- Target column (canonical name + known variants)
- Feature columns in fixed order
- Cleaning maps for undocumented category codes
- Risk-band thresholds (analytical, documented in docs/project_overview.md)
- Split parameters and logging level

All modules must import constants from here rather than hard-coding values.
"""

from src.utils.paths import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    SAVED_MODELS_DIR,
    EVALUATION_DIR,
    DOCS_DIR,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_FILENAME: str = "default of credit card clients.xls"
DATASET_URL: str = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)

# Row 0 in the .xls file is X1, X2, ..., Y (a legacy numbering row).
# Row 1 contains the real column names.
DATASET_HEADER_ROW: int = 1

# Canonical target column name used throughout CreditLens.
TARGET_COLUMN: str = "default payment next month"

# Variants observed in other mirrors of the same dataset. The loader renames
# whichever variant it finds to TARGET_COLUMN.
TARGET_COLUMN_CANDIDATES: tuple[str, ...] = (
    "default payment next month",
    "default.payment.next.month",
    "Y",
)

POSITIVE_CLASS_LABEL: int = 1
NEGATIVE_CLASS_LABEL: int = 0

# Identifier columns dropped before modeling.
ID_COLUMNS: tuple[str, ...] = ("ID", "id")

# The 23 feature columns in fixed order. Every model and preprocessor must
# consume the features in this exact order.
FEATURE_COLUMNS: tuple[str, ...] = (
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
)

# Continuous / numeric features.
NUMERIC_FEATURES: tuple[str, ...] = (
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
)

# Categorical features. Repayment-status fields (PAY_*) are treated as
# categorical here because the numeric codes are not a linear severity scale:
# negative values carry distinct semantic meanings (-2 = no consumption,
# -1 = paid in full, 0 = revolving credit). One-hot encoding avoids imposing
# a false ordinal structure on Logistic Regression.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
)

# Documented cleaning decisions for undocumented category codes.
# EDUCATION: 0, 5, 6 are not in the UCI data dictionary → merged into 4 (others).
# MARRIAGE: 0 is not in the UCI data dictionary → merged into 3 (others).
EDUCATION_MAP: dict[int, int] = {0: 4, 5: 4, 6: 4}
MARRIAGE_MAP: dict[int, int] = {0: 3}

# ---------------------------------------------------------------------------
# Risk bands — analytical thresholds, not regulatory classifications.
# ---------------------------------------------------------------------------
RISK_BANDS: dict[str, tuple[float, float]] = {
    "Low":       (0.00, 0.20),
    "Moderate":  (0.20, 0.50),
    "High":      (0.50, 0.80),
    "Very High": (0.80, 1.01),
}

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.20

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = "INFO"