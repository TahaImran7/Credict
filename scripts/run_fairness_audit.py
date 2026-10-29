"""
One-shot fairness audit over the held-out test set.

Run from the project root:
    python -m scripts.run_fairness_audit

Writes three files under models/evaluation/:
    fairness_metrics.csv       (pure metrics — one row per group)
    fairness_differences.csv   (pure differences — one row per group)
    fairness_report.json       (combined report)

IMPORTANT: fairness_metrics.csv must contain ONLY metric rows. The UI loads
this file directly to compute small_group flags; appending difference rows
or separator rows produces NaN in `small_group` and `n`, which breaks the
masking logic in ui/fairness_audit.py.
"""

from config import settings
from src.analytics.portfolio import model_risk_bands_for  # noqa: F401  (documented dependency)
from src.data import preprocessing as pp
from src.fairness.audit import (
    FAIRNESS_LIMITATION,
    audit_standard_groups,
    flatten_audit_results,
)
from src.prediction.predictor import DEFAULT_MODEL_NAME, predict_batch
from src.reporting.report_generator import (
    fairness_report_csv,
    fairness_report_json,
    write_to_disk,
)
from src.utils.logging_config import get_logger
from src.utils.paths import EVALUATION_DIR, PROJECT_ROOT

log = get_logger(__name__)


def _relative_eval_path(filename: str) -> str:
    """Return the path of a file under models/evaluation/ relative to PROJECT_ROOT."""
    return str((EVALUATION_DIR / filename).relative_to(PROJECT_ROOT))


def main() -> None:
    log.info("=== Credict Phase 6: fairness audit ===")

    X_train, X_test, y_train, y_test = pp.load_splits()

    # Reconstruct the test slice with its grouping columns. Uses the same
    # seed as Phase 2 and verifies the reconstruction matches the persisted
    # split before continuing.
    from src.data.loader import load_raw_dataset
    df_raw = load_raw_dataset()
    df_clean = pp.clean_dataset(df_raw)
    X_all, y_all = pp.split_features_target(df_clean)
    _, X_test_ref, _, y_test_ref = pp.make_train_test_split(X_all, y_all)

    assert X_test_ref.reset_index(drop=True).equals(
        X_test.reset_index(drop=True)
    ), "Reconstructed test split does not match persisted split."

    test_positions = X_test_ref.index.to_numpy()
    audit_df = df_raw.iloc[test_positions][
        [c for c in ("SEX", "EDUCATION", "MARRIAGE", "AGE") if c in df_raw.columns]
    ].copy()
    audit_df = pp.clean_dataset(audit_df.assign(**{
        settings.TARGET_COLUMN: y_test_ref.values
    }))

    probs = predict_batch(X_test_ref, model_name=DEFAULT_MODEL_NAME)
    y_pred = (probs >= 0.5).astype(int)

    results = audit_standard_groups(audit_df, y_pred=y_pred, y_prob=probs)
    metrics_df, diff_df = flatten_audit_results(results)

    # Ensure clean row indices and column ordering before writing.
    metrics_df = metrics_df.reset_index(drop=True)
    diff_df = diff_df.reset_index(drop=True)

    print("\n=== Fairness metrics (test set) ===")
    print(metrics_df.round(4).to_string(index=False))
    print("\n=== Differences from reference group ===")
    print(diff_df.round(4).to_string(index=False))

    all_warnings: list[str] = []
    for name, res in results.items():
        for w in res.warnings:
            all_warnings.append(f"[{name}] {w}")
    if all_warnings:
        print("\n=== Warnings ===")
        for w in all_warnings:
            print("  -", w)

    # --- Metrics: PURE metrics only. No diff_df argument. ---------------
    write_to_disk(
        fairness_report_csv(metrics_df),
        _relative_eval_path("fairness_metrics.csv"),
    )
    # --- Differences: PURE differences only. ----------------------------
    write_to_disk(
        fairness_report_csv(diff_df),
        _relative_eval_path("fairness_differences.csv"),
    )
    # --- Combined report for downloads ----------------------------------
    write_to_disk(
        fairness_report_json(
            metrics_df, diff_df, all_warnings,
            limitation_text=FAIRNESS_LIMITATION,
        ),
        _relative_eval_path("fairness_report.json"),
    )

    log.info("Phase 6 fairness audit complete.")


if __name__ == "__main__":
    main()