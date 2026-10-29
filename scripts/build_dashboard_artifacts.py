"""
Pre-compute dashboard artefacts for Credict.

Run once after Phase 3 (or after retraining):
    python -m scripts.build_dashboard_artifacts
"""

import numpy as np
import pandas as pd

from config import settings
from src.analytics.portfolio import model_risk_bands_for
from src.data import preprocessing as pp
from src.explainability.shap_explainer import global_importance
from src.prediction.predictor import (
    DEFAULT_MODEL_NAME, SHAP_MODEL_NAME, predict_batch,
)
from src.utils.logging_config import get_logger
from src.utils.paths import EVALUATION_DIR

log = get_logger(__name__)


def main() -> None:
    log.info("=== Credict: pre-computing dashboard artefacts ===")

    _, X_test, _, y_test = pp.load_splits()

    # 1. Global SHAP importance — uses RAW RF (TreeExplainer requires raw tree)
    log.info("Computing global SHAP importance on %d rows...", len(X_test))
    importance = global_importance(
        X_test, model_name=SHAP_MODEL_NAME,
        max_samples=len(X_test),
    )
    path = EVALUATION_DIR / "global_importance.csv"
    importance.to_csv(path, index=False)
    log.info("Wrote %s (%d rows)", path, len(importance))

    # 2. Test-set predictions — uses CALIBRATED model (for display + analytics)
    probs = predict_batch(X_test, model_name=DEFAULT_MODEL_NAME)
    bands_df = model_risk_bands_for(X_test, model_name=DEFAULT_MODEL_NAME)
    pred_df = pd.DataFrame({
        "probability": probs,
        "risk_band": bands_df["risk_band"].values,
        "y_true": np.asarray(y_test),
    })
    path2 = EVALUATION_DIR / "test_predictions.csv"
    pred_df.to_csv(path2, index=False)
    log.info("Wrote %s (%d rows)", path2, len(pred_df))

    log.info("Dashboard artefacts ready.")


if __name__ == "__main__":
    main()