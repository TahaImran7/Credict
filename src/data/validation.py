"""
Dataset validation for CreditLens.

Checks structural integrity (shape, columns, target, value ranges) and emits
a structured report. Used in Phase 2 tests and shown in the Phase 7 UI.
"""

from dataclasses import dataclass, field

import pandas as pd

from config import settings
from src.utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class ValidationReport:
    n_rows: int
    n_columns: int
    missing_per_column: dict[str, int]
    duplicate_rows: int
    target_present: bool
    target_values: list[int]
    education_codes: list[int]
    marriage_codes: list[int]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [
            f"Rows: {self.n_rows}",
            f"Columns: {self.n_columns}",
            f"Duplicate rows: {self.duplicate_rows}",
            f"Target present: {self.target_present}",
            f"Target values: {self.target_values}",
            f"Missing cells (total): {sum(self.missing_per_column.values())}",
            f"EDUCATION codes present: {self.education_codes}",
            f"MARRIAGE codes present: {self.marriage_codes}",
        ]
        if self.errors:
            lines.append("ERRORS:")
            lines.extend(f"  - {e}" for e in self.errors)
        if self.warnings:
            lines.append("WARNINGS:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    """
    Run structural and semantic checks.

    Errors indicate the dataset cannot be used. Warnings indicate issues
    that the preprocessing pipeline is expected to handle.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Target presence
    target_present = settings.TARGET_COLUMN in df.columns
    if not target_present:
        errors.append(f"Target column '{settings.TARGET_COLUMN}' is missing.")

    # Required feature columns
    for col in settings.FEATURE_COLUMNS:
        if col not in df.columns:
            errors.append(f"Expected feature column '{col}' is missing.")

    # Missing values
    missing_per_column = {c: int(df[c].isna().sum()) for c in df.columns}
    if sum(missing_per_column.values()) > 0:
        warnings.append("Missing values detected; review before training.")

    # Duplicates
    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows > 0:
        warnings.append(f"{duplicate_rows} duplicate rows detected.")

    # Target values
    target_values: list[int] = []
    if target_present:
        target_values = sorted(df[settings.TARGET_COLUMN].dropna().unique().tolist())
        unexpected = set(target_values) - {
            settings.POSITIVE_CLASS_LABEL,
            settings.NEGATIVE_CLASS_LABEL,
        }
        if unexpected:
            errors.append(f"Unexpected target values: {unexpected}")

    # Category code inventory (informational; the pipeline handles known oddities)
    education_codes = (
        sorted(df["EDUCATION"].dropna().unique().tolist())
        if "EDUCATION" in df.columns else []
    )
    marriage_codes = (
        sorted(df["MARRIAGE"].dropna().unique().tolist())
        if "MARRIAGE" in df.columns else []
    )

    report = ValidationReport(
        n_rows=int(df.shape[0]),
        n_columns=int(df.shape[1]),
        missing_per_column=missing_per_column,
        duplicate_rows=duplicate_rows,
        target_present=target_present,
        target_values=target_values,
        education_codes=education_codes,
        marriage_codes=marriage_codes,
        errors=errors,
        warnings=warnings,
    )
    log.info("Validation report:\n%s", report.summary())
    return report