"""
One-shot Phase 3 pipeline:
    load splits -> tune LR -> tune RF -> save models -> evaluate -> compare.

Run from the project root:
    python -m scripts.train_models
"""

from config import settings
from src.data import preprocessing as pp
from src.training.evaluate import evaluate_model, save_evaluation
from src.training.model_comparison import (
    compare_models,
    save_comparison,
    select_best_model,
)
from src.training.train import (
    load_model,
    save_cv_results,
    save_model,
    train_logistic_regression,
    train_random_forest,
)
from src.utils.logging_config import get_logger
from src.utils.paths import EVALUATION_DIR

log = get_logger(__name__)


def main() -> None:
    log.info("=== CreditLens Phase 3: model training ===")

    X_train, X_test, y_train, y_test = pp.load_splits()
    log.info("Loaded splits: train=%d, test=%d", len(X_train), len(X_test))

    # --- Logistic Regression -------------------------------------------
    lr_grid = train_logistic_regression(X_train, y_train)
    lr_model = lr_grid.best_estimator_
    save_model(lr_model, "logistic_regression")
    save_cv_results(lr_grid, "logistic_regression")

    # --- Random Forest --------------------------------------------------
    rf_grid = train_random_forest(X_train, y_train)
    rf_model = rf_grid.best_estimator_
    save_model(rf_model, "random_forest")
    save_cv_results(rf_grid, "random_forest")

    # --- Held-out test evaluation --------------------------------------
    evaluations = [
        evaluate_model(lr_model, X_test, y_test, "logistic_regression"),
        evaluate_model(rf_model, X_test, y_test, "random_forest"),
    ]
    for ev in evaluations:
        save_evaluation(ev, EVALUATION_DIR / f"{ev.model_name}_eval.json")

    # --- Comparison & selection ----------------------------------------
    df = compare_models(evaluations)
    save_comparison(df, EVALUATION_DIR / "model_comparison.csv")
    print("\n=== Model comparison (held-out test set) ===")
    print(df.round(4).to_string())

    best = select_best_model(df)
    print(f"\nDocumented selection rule chooses: {best}")
    log.info("Phase 3 training complete.")


if __name__ == "__main__":
    main()