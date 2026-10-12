"""
SHAP-based explanations for CreditLens.

The fitted pipeline (preprocessor + classifier) is used directly:
  - Preprocessing is applied to the input record.
  - TreeExplainer explains the classifier on the preprocessed features.
  - SHAP values are aggregated from one-hot expanded names back to the
    original 23 feature names, so explanations are interpretable.

SHAP describes how features contribute to the model's output under its
explanation framework. It does not establish causality.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
import shap

from config import settings
from src.prediction.predictor import DEFAULT_MODEL_NAME, _prepare_record
from src.training.train import load_model
from src.utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass
class ShapExplanation:
    model_name: str
    base_value: float
    probability: float
    expanded_features: list[str] = field(default_factory=list)
    expanded_shap_values: list[float] = field(default_factory=list)
    original_features: list[str] = field(default_factory=list)
    original_shap_values: list[float] = field(default_factory=list)
    original_input_values: list = field(default_factory=list)

    def top_contributors(self, n: int = 5) -> list[dict]:
        """
        Return the n features with the largest |SHAP| contribution,
        preserving sign so the caller can distinguish risk-increasing from
        risk-decreasing contributions.
        """
        pairs = list(zip(self.original_features, self.original_shap_values,
                         self.original_input_values))
        pairs.sort(key=lambda t: abs(t[1]), reverse=True)
        return [
            {
                "feature": name,
                "shap_value": float(sv),
                "input_value": val,
                "direction": "increases risk" if sv > 0 else "decreases risk",
            }
            for name, sv, val in pairs[:n]
        ]


# ---------------------------------------------------------------------------
# SHAP value extraction — defensive across shap versions
# ---------------------------------------------------------------------------

def _positive_class_shap(shap_values) -> np.ndarray:
    """
    Normalise TreeExplainer output for binary classification to a
    (n_samples, n_features) array corresponding to the positive class.
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def _positive_class_base(expected_value) -> float:
    """Extract the positive-class base value, defensively."""
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        return float(expected_value[1])
    return float(expected_value)


# ---------------------------------------------------------------------------
# Name aggregation: expanded (num__X / cat__Y_Z) -> original feature name
# ---------------------------------------------------------------------------

def original_feature_from_expanded(expanded: str) -> str:
    """
    Map an expanded column name produced by ColumnTransformer back to the
    original feature name.

    num__LIMIT_BAL     -> LIMIT_BAL
    cat__SEX_1         -> SEX
    cat__PAY_0_2       -> PAY_0
    """
    if expanded.startswith("num__"):
        return expanded[len("num__"):]
    if expanded.startswith("cat__"):
        rest = expanded[len("cat__"):]
        # Match the longest categorical feature name that fits.
        matches = [c for c in settings.CATEGORICAL_FEATURES
                   if rest == c or rest.startswith(c + "_")]
        if matches:
            return max(matches, key=len)
        return rest
    return expanded


def aggregate_by_original_feature(
    expanded_names: list[str],
    expanded_values: np.ndarray,
) -> tuple[list[str], list[float]]:
    """
    Sum SHAP values per original feature. Preserves order of first appearance.
    """
    order: list[str] = []
    total: dict[str, float] = {}
    for name, val in zip(expanded_names, expanded_values):
        orig = original_feature_from_expanded(name)
        if orig not in total:
            order.append(orig)
            total[orig] = 0.0
        total[orig] += float(val)
    return order, [total[k] for k in order]


# ---------------------------------------------------------------------------
# Explainers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _get_explainer(model_name: str) -> tuple[object, object, list[str]]:
    """
    Cache (classifier, TreeExplainer, expanded_feature_names) per model name.
    Building TreeExplainer over 200 trees is not free; caching matters for
    Streamlit reruns in Phase 7.
    """
    pipeline = load_model(model_name)
    if "pre" not in pipeline.named_steps or "clf" not in pipeline.named_steps:
        raise ValueError(
            f"Model '{model_name}' is not a Pipeline with 'pre' and 'clf'."
        )
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]
    try:
        explainer = shap.TreeExplainer(clf)
    except Exception as exc:
        raise RuntimeError(
            f"SHAP TreeExplainer is not compatible with the '{model_name}' "
            f"classifier ({type(clf).__name__}). Details: {exc}"
        ) from exc
    expanded_names = list(pre.get_feature_names_out())
    return pipeline, explainer, expanded_names


def clear_explainer_cache() -> None:
    """Explicit cache reset (useful in tests / after model retraining)."""
    _get_explainer.cache_clear()


def explain_single(
    record: dict | pd.Series | pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
) -> ShapExplanation:
    """Produce a SHAP explanation for one customer record."""
    pipeline, explainer, expanded_names = _get_explainer(model_name)
    X_row = _prepare_record(record)
    pre = pipeline.named_steps["pre"]
    X_transformed = pre.transform(X_row)

    sv_raw = explainer.shap_values(X_transformed)
    sv = _positive_class_shap(sv_raw)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    sv_row = sv[0]

    base_value = _positive_class_base(explainer.expected_value)
    probability = float(pipeline.predict_proba(X_row)[0, 1])

    orig_names, orig_values = aggregate_by_original_feature(expanded_names, sv_row)
    input_values = [X_row.iloc[0][name] for name in orig_names]

    return ShapExplanation(
        model_name=model_name,
        base_value=base_value,
        probability=probability,
        expanded_features=expanded_names,
        expanded_shap_values=sv_row.tolist(),
        original_features=orig_names,
        original_shap_values=orig_values,
        original_input_values=input_values,
    )


def global_importance(
    X_sample: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
    max_samples: int = 500,
    random_state: int = settings.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Mean absolute SHAP value per original feature, computed on a random
    subsample. Suitable for the UI's global-importance chart.

    Returns a DataFrame sorted by importance descending with columns:
        feature, mean_abs_shap
    """
    pipeline, explainer, expanded_names = _get_explainer(model_name)
    if len(X_sample) > max_samples:
        X_used = X_sample.sample(max_samples, random_state=random_state)
    else:
        X_used = X_sample
    X_ordered = X_used[list(settings.FEATURE_COLUMNS)]
    X_transformed = pipeline.named_steps["pre"].transform(X_ordered)

    sv_raw = explainer.shap_values(X_transformed)
    sv = _positive_class_shap(sv_raw)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)

    mean_abs_expanded = np.abs(sv).mean(axis=0)
    orig_names, orig_values = aggregate_by_original_feature(
        expanded_names, mean_abs_expanded,
    )
    df = pd.DataFrame(
        {"feature": orig_names, "mean_abs_shap": orig_values}
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    log.info(
        "Computed global SHAP importance on %d samples; top feature: %s",
        len(X_used), df.iloc[0]["feature"],
    )
    return df