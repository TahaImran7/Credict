"""
CreditLens — central configuration.

Single source of truth for:
- Random seed
- Target column and class labels
- Dataset filename convention
- Risk-band thresholds (analytical choice, documented)
- Paths (re-exported from src.utils.paths)

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
# The UCI file is distributed as an .xls workbook. Confirm the real filename
# after download in Phase 2; this is the expected default.
DATASET_FILENAME: str = "default of credit card clients.xls"

# Original target column name in the UCI file. Some mirrors rename this to
# "default.payment.next.month". The loader in Phase 2 will reconcile both.
TARGET_COLUMN_CANDIDATES: tuple[str, ...] = (
    "default.payment.next.month",
    "default payment next month",
    "Y",
)

# Positive class = default occurred next month.
POSITIVE_CLASS_LABEL: int = 1
NEGATIVE_CLASS_LABEL: int = 0

# ---------------------------------------------------------------------------
# Risk bands — analytical thresholds, not regulatory classifications.
# Documented in docs/project_overview.md and shown in the UI.
# ---------------------------------------------------------------------------
RISK_BANDS: dict[str, tuple[float, float]] = {
    "Low":       (0.00, 0.20),
    "Moderate":  (0.20, 0.50),
    "High":      (0.50, 0.80),
    "Very High": (0.80, 1.01),   # upper bound exclusive in code
}

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.20

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = "INFO"