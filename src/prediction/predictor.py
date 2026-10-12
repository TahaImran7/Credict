"""
Prediction interface for CreditLens.

Single source of truth for turning a raw customer record into a
next-month default probability and risk band. Uses the fitted pipeline saved
by Phase 3 (preprocessor + classifier are one joblib artefact, so prediction
cannot drift from training).
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import settings
from src.training.train import load_model
from src.utils.logging_config import get_logger

log = get_logger(__name__)

# Phase 3's documented selection rule chose random_forest.
DEFAULT_MODEL_NAME: str = "random_forest"
AVAILABLE_MODELS: tuple[str, ...] = ("logistic_regression", "random_forest")


@dataclass
class PredictionResult:
    model_name: str
    probability: float
    risk_band: str
    predicted_class: int
    threshold: float
    timestamp_utc: str

    def as_dict(self) -> dict:
        return asdict(self)


def _ensure_feature_order(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in settings.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns for prediction: {missing}")
    return df[list(settings.FEATURE_COLUMNS)].copy()


def assign_risk_band(probability: float) -> str:
    """Map a probability to the configured risk band. Bands are analytical."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probability out of range: {probability}")
    for band, (low, high) in settings.RISK_BANDS.items():
        if band == "Very High":
            if low <= probability <= 1.0:
                return band
        elif low <= probability < high:
            return band
    # Defensive: unreachable if bands cover [0, 1].
    raise RuntimeError(f"No risk band covers probability {probability}")


def _prepare_record(record: dict | pd.Series | pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a single customer record into a 1-row DataFrame in the canonical
    feature order. Accepts dict, Series, or single-row DataFrame.
    """
    if isinstance(record, pd.DataFrame):
        if record.shape[0] != 1:
            raise ValueError(
                f"Expected a single-row DataFrame, got shape {record.shape}."
            )
        df = record.copy()
    elif isinstance(record, pd.Series):
        df = record.to_frame().T
    elif isinstance(record, dict):
        df = pd.DataFrame([record])
    else:
        raise TypeError(
            f"Unsupported record type: {type(record).__name__}. "
            "Pass dict, Series, or single-row DataFrame."
        )
    return _ensure_feature_order(df)


def predict_single(
    record: dict | pd.Series | pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
    threshold: float = 0.5,
) -> PredictionResult:
    """Return the next-month default probability and risk band for one record."""
    model = load_model(model_name)
    X = _prepare_record(record)
    proba = float(model.predict_proba(X)[0, 1])
    result = PredictionResult(
        model_name=model_name,
        probability=proba,
        risk_band=assign_risk_band(proba),
        predicted_class=int(proba >= threshold),
        threshold=float(threshold),
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    log.info(
        "Predicted %s for one record: p=%.4f, band=%s",
        model_name, proba, result.risk_band,
    )
    return result


def predict_batch(
    X: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
) -> np.ndarray:
    """Return positive-class probabilities for a DataFrame of records."""
    model = load_model(model_name)
    X_ordered = _ensure_feature_order(X)
    return model.predict_proba(X_ordered)[:, 1]


def predict_batch_with_bands(
    X: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
) -> pd.DataFrame:
    """Return a DataFrame with probability and risk band per row."""
    probs = predict_batch(X, model_name=model_name)
    bands = [assign_risk_band(float(p)) for p in probs]
    return pd.DataFrame(
        {"probability": probs, "risk_band": bands},
        index=X.index,
    )