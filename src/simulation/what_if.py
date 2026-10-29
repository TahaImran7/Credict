"""
What-if simulator for Credict.

Allows controlled modification of selected customer features and returns a
side-by-side comparison of original and modified model estimates.

Scope rules (per spec §6):
  - Only permitted features may change.
  - Types, categories, and numeric ranges are validated.
  - The same model and preprocessing pipeline used for prediction are reused.
  - The original record is never mutated.
  - The result is a model-based scenario, not a financial guarantee.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from config import settings
from src.prediction.predictor import (
    DEFAULT_MODEL_NAME,
    _prepare_record,
    assign_risk_band,
    predict_single,
)
from src.training.train import load_model
from src.utils.logging_config import get_logger

log = get_logger(__name__)

DISCLAIMER: str = (
    "Results are model-based scenarios, not guaranteed financial outcomes. "
    "Changing one feature while holding others constant may produce "
    "combinations that are unlikely in real customer behaviour. Correlation "
    "between historical repayment behaviour and payment amounts means an "
    "isolated change in one field does not imply the same change in risk "
    "in real life."
)


# ---------------------------------------------------------------------------
# Validation of permitted changes
# ---------------------------------------------------------------------------

def _int_in(min_v: int, max_v: int) -> Callable[[Any], int]:
    def _check(v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, (int,)):
            try:
                v_int = int(v)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Expected an integer, got {v!r}") from exc
            v = v_int
        if not (min_v <= v <= max_v):
            raise ValueError(f"Value {v} out of allowed range [{min_v}, {max_v}]")
        return int(v)
    return _check


def _number_in(min_v: float, max_v: float) -> Callable[[Any], float]:
    def _check(v: Any) -> float:
        try:
            v_f = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Expected a number, got {v!r}") from exc
        if v_f != v_f:  # NaN check
            raise ValueError("Value must not be NaN")
        if not (min_v <= v_f <= max_v):
            raise ValueError(f"Value {v_f} out of allowed range [{min_v}, {max_v}]")
        return v_f
    return _check


def _in_set(allowed: set) -> Callable[[Any], Any]:
    def _check(v: Any) -> Any:
        if v not in allowed:
            raise ValueError(f"Value {v!r} not in allowed set {sorted(allowed)}")
        return v
    return _check


# Permitted changes: feature -> validator. Any feature NOT in this map
# cannot be modified by the simulator.
VALIDATORS: dict[str, Callable[[Any], Any]] = {
    "LIMIT_BAL": _number_in(0.0, 1_000_000.0),
    "AGE": _int_in(18, 100),
    "SEX": _in_set({1, 2}),
    "EDUCATION": _in_set({1, 2, 3, 4}),
    "MARRIAGE": _in_set({1, 2, 3}),
    "PAY_0": _int_in(-2, 9),
    "PAY_2": _int_in(-2, 9),
    "PAY_3": _int_in(-2, 9),
    "PAY_4": _int_in(-2, 9),
    "PAY_5": _int_in(-2, 9),
    "PAY_6": _int_in(-2, 9),
    **{f"BILL_AMT{i}": _number_in(-1_000_000.0, 10_000_000.0) for i in range(1, 7)},
    **{f"PAY_AMT{i}": _number_in(0.0, 10_000_000.0) for i in range(1, 7)},
}

PERMITTED_FEATURES: tuple[str, ...] = tuple(VALIDATORS.keys())


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ScenarioChange:
    feature: str
    original: Any
    new: Any


@dataclass
class WhatIfResult:
    model_name: str
    original_probability: float
    modified_probability: float
    original_band: str
    modified_band: str
    changes: list[ScenarioChange] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    @property
    def difference_pp(self) -> float:
        """Difference in percentage points (modified − original) × 100."""
        return (self.modified_probability - self.original_probability) * 100.0

    @property
    def direction(self) -> str:
        if abs(self.difference_pp) < 0.5:
            return "negligible"
        return "increased" if self.difference_pp > 0 else "decreased"


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def validate_changes(changes: dict[str, Any]) -> dict[str, Any]:
    """
    Validate that each change targets a permitted feature and passes its
    validator. Returns a dict of validated values. Raises ValueError on the
    first invalid entry.
    """
    if not isinstance(changes, dict) or not changes:
        raise ValueError("changes must be a non-empty dict of {feature: new_value}")
    validated: dict[str, Any] = {}
    for feature, value in changes.items():
        if feature not in VALIDATORS:
            raise ValueError(
                f"Feature '{feature}' is not permitted for simulation. "
                f"Permitted features: {sorted(PERMITTED_FEATURES)}"
            )
        validated[feature] = VALIDATORS[feature](value)
    return validated


def _check_plausibility(changes: dict[str, Any]) -> list[str]:
    """
    Return a list of human-readable warnings for combinations that are
    likely unrealistic. Does NOT block the simulation.
    """
    warnings: list[str] = []
    pay_fields = [f for f in changes if f.startswith("PAY_") and f != "PAY_AMT1"
                  and not f.startswith("PAY_AMT")]
    # ^ pay-status fields only (PAY_0, PAY_2..PAY_6), not PAY_AMT*
    if pay_fields and all(int(changes[f]) <= 0 for f in pay_fields):
        has_high_pay_amt = any(
            f.startswith("PAY_AMT") and float(changes[f]) > 0 for f in changes
        )
        if has_high_pay_amt:
            warnings.append(
                "Setting all repayment-status fields to 'paid in full' while "
                "also changing historical payment amounts may not represent a "
                "realistic combined scenario."
            )
    if "LIMIT_BAL" in changes and float(changes["LIMIT_BAL"]) == 0:
        warnings.append(
            "A credit limit of zero is unusual and may not reflect a real "
            "customer record."
        )
    return warnings


def simulate(
    record: dict | pd.Series | pd.DataFrame,
    changes: dict[str, Any],
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[WhatIfResult, list[str]]:
    """
    Compare original vs. modified prediction for one record.

    Returns (WhatIfResult, plausibility_warnings).
    The input `record` is never mutated.
    """
    validated = validate_changes(changes)
    warnings = _check_plausibility(validated)

    original_df = _prepare_record(record)
    original_prob = float(load_model(model_name).predict_proba(original_df)[0, 1])

    modified_df = original_df.copy()
    scenario_changes: list[ScenarioChange] = []
    for feature, new_value in validated.items():
        old_value = modified_df.iloc[0][feature]
        if old_value == new_value:
            continue
        modified_df.loc[modified_df.index[0], feature] = new_value
        scenario_changes.append(
            ScenarioChange(feature=feature, original=old_value, new=new_value)
        )

    modified_prob = float(load_model(model_name).predict_proba(modified_df)[0, 1])

    result = WhatIfResult(
        model_name=model_name,
        original_probability=original_prob,
        modified_probability=modified_prob,
        original_band=assign_risk_band(original_prob),
        modified_band=assign_risk_band(modified_prob),
        changes=scenario_changes,
    )
    log.info(
        "Simulated %d change(s) on %s: %.4f -> %.4f (%+.2f pp)",
        len(scenario_changes), model_name,
        original_prob, modified_prob, result.difference_pp,
    )
    return result, warnings


def simulate_grid(
    record: dict | pd.Series | pd.DataFrame,
    feature: str,
    values: list[Any],
    model_name: str = DEFAULT_MODEL_NAME,
) -> pd.DataFrame:
    """
    Compute modified probability for a single feature across a list of values.
    Useful for plotting risk-vs-feature curves. Returns a DataFrame with
    columns: value, probability, risk_band.
    """
    if feature not in VALIDATORS:
        raise ValueError(f"Feature '{feature}' is not permitted for simulation.")
    original_df = _prepare_record(record)
    model = load_model(model_name)
    rows = []
    for v in values:
        validated = VALIDATORS[feature](v)
        modified = original_df.copy()
        modified.loc[modified.index[0], feature] = validated
        p = float(model.predict_proba(modified)[0, 1])
        rows.append({
            "value": validated,
            "probability": p,
            "risk_band": assign_risk_band(p),
        })
    return pd.DataFrame(rows)