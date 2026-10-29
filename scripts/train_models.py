"""
One-shot Phase 3 pipeline with calibration.

Run from the project root:
    python -m scripts.train_models
"""

from config import settings
from src.data import preprocessing as pp
from src.training.calibrate import calibrate_pipeline, save_calibrated_model
from src.training.evaluate import evaluate_model, save_evaluation
from src.training.model_comparison import (
    compare_models, save_comparison, select_final_model,
)
from src.training.train import (
    save_cv_results, save_model,
    train_logistic_regression, train_random_forest,
)
from src.utils.logging_config import get_logger
from src.utils.paths import EVALUATION_DIR

log = get_logger(__name__)


def main() -> None:
    log.info("=== Credict Phase 3: model training with calibration ===")

    X_train, X_test, y_train, y_test = pp.load_splits()
    log.info("Loaded splits: train=%d, test=%d", len(X_train), len(X_test))

    # --- Logistic Regression -------------------------------------------
    lr_grid = train_logistic_regression(X_train, y_train)
    lr_model = lr_grid.best_estimator_
    save_model(lr_model, "logistic_regression")
    save_cv_results(lr_grid, "logistic_regression")

    # --- Random Forest (raw) -------------------------------------------
    rf_grid = train_random_forest(X_train, y_train)
    rf_model = rf_grid.best_estimator_
    save_model(rf_model, "random_forest")
    save_cv_results(rf_grid, "random_forest")

    # --- Random Forest (calibrated) ------------------------------------
    log.info("=== Calibrating Random Forest probabilities ===")
    calibrated_model = calibrate_pipeline(rf_model, X_train, y_train)
    save_calibrated_model(calibrated_model, "random_forest_calibrated")

    # --- Evaluation ----------------------------------------------------
    evaluations = [
        evaluate_model(lr_model, X_test, y_test, "logistic_regression"),
        evaluate_model(rf_model, X_test, y_test, "random_forest"),
        evaluate_model(calibrated_model, X_test, y_test, "random_forest_calibrated"),
    ]
    for ev in evaluations:
        save_evaluation(ev, EVALUATION_DIR / f"{ev.model_name}_eval.json")

    df = compare_models(evaluations)
    save_comparison(df, EVALUATION_DIR / "model_comparison.csv")

    print("\n=== Model comparison (held-out test set) ===")
    print(df.round(4).to_string())

    # --- Calibration impact (honest narrative) -------------------------
    raw_rf = next(ev for ev in evaluations if ev.model_name == "random_forest")
    cal_rf = next(ev for ev in evaluations if ev.model_name == "random_forest_calibrated")

    print("\n=== Calibration impact (Random Forest) ===")
    print(f"  Threshold      : {raw_rf.threshold:.4f}  →  {cal_rf.threshold:.4f}")
    print(f"  Brier (lower=better): {raw_rf.metrics['brier']:.4f}  →  {cal_rf.metrics['brier']:.4f}")
    brier_delta = (raw_rf.metrics['brier'] - cal_rf.metrics['brier']) / raw_rf.metrics['brier'] * 100
    print(f"                          ({brier_delta:+.1f}% improvement)")
    print(f"  ROC-AUC        : {raw_rf.metrics['roc_auc']:.4f}  →  {cal_rf.metrics['roc_auc']:.4f}  "
          f"(isotonic ties cause tiny drop)")
    print(f"  PR-AUC         : {raw_rf.metrics['pr_auc']:.4f}  →  {cal_rf.metrics['pr_auc']:.4f}  "
          f"(isotonic ties cause small drop)")
    print(f"  Accuracy       : {raw_rf.metrics['accuracy']:.4f}  →  {cal_rf.metrics['accuracy']:.4f}")
    print(f"  Recall         : {raw_rf.metrics['recall']:.4f}  →  {cal_rf.metrics['recall']:.4f}")
    print(f"  F1             : {raw_rf.metrics['f1']:.4f}  →  {cal_rf.metrics['f1']:.4f}")

    # --- Final selection ----------------------------------------------
    final_model, rationale = select_final_model(df)
    print(f"\n=== Final selection ===")
    print(f"  Base model chosen: {rationale['best_base']}")
    print(f"  Reason           : {rationale['reason']}")
    print(f"  FINAL MODEL      : {final_model}")

    log.info("Phase 3 training with calibration complete. Final model: %s", final_model)


if __name__ == "__main__":
    main()