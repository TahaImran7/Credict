"""
One-shot fairness audit over the held-out test set.

Run from the project root:
    python -m scripts.run_fairness_audit

Writes:
    models/evaluation/fairness_metrics.csv
    models/evaluation/fairness_differences.csv
    models/evaluation/fairness_report.json
"""

from config import settings
from src.analytics.portfolio import model_risk_bands_for
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
from src.utils.paths import EVALUATION_DIR

log = get_logger(__name__)


def main() -> None:
    log.info("=== CreditLens Phase 6: fairness audit ===")

    X_train, X_test, y_train, y_test = pp.load_splits()

    # Rebuild the labelled test frame with the raw (uncleaned) grouping
    # columns needed for audits. We rebuild from the raw file and align by
    # positional index order, which is identical to the split used in Phase 2.
    from src.data.loader import load_raw_dataset
    df_raw = load_raw_dataset()
    df_clean = pp.clean_dataset(df_raw)
    # Recreate the same split to obtain the test-slice labels + groups.
    X_all, y_all = pp.split_features_target(df_clean)
    _, X_test_ref, _, y_test_ref = pp.make_train_test_split(X_all, y_all)

    # Guard: our loaded splits and the freshly reproduced split must match.
    assert X_test_ref.reset_index(drop=True).equals(
        X_test.reset_index(drop=True)
    ), "Reconstructed test split does not match persisted split."

    # Build the audit dataframe from raw (pre-clean) columns aligned to test rows
    test_positions = X_test_ref.index.to_numpy()
    audit_df = df_raw.iloc[test_positions][
        [c for c in ("SEX", "EDUCATION", "MARRIAGE", "AGE") if c in df_raw.columns]
    ].copy()
    audit_df = pp.clean_dataset(audit_df.assign(**{
        settings.TARGET_COLUMN: y_test_ref.values
    }))

    # Predictions from the selected model
    probs = predict_batch(X_test_ref, model_name=DEFAULT_MODEL_NAME)
    y_pred = (probs >= 0.5).astype(int)

    results = audit_standard_groups(audit_df, y_pred=y_pred, y_prob=probs)
    metrics_df, diff_df = flatten_audit_results(results)

    print("\n=== Fairness metrics (test set) ===")
    print(metrics_df.round(4).to_string(index=False))
    print("\n=== Differences from reference group ===")
    print(diff_df.round(4).to_string(index=False))

    all_warnings = []
    for name, res in results.items():
        for w in res.warnings:
            all_warnings.append(f"[{name}] {w}")
    if all_warnings:
        print("\n=== Warnings ===")
        for w in all_warnings:
            print("  -", w)

    write_to_disk(
        fairness_report_csv(metrics_df, diff_df),
        str(EVALUATION_DIR.relative_to(EVALUATION_DIR.parents[1])
            / "fairness_metrics.csv"),
    )
    write_to_disk(
        fairness_report_csv(diff_df),
        str(EVALUATION_DIR.relative_to(EVALUATION_DIR.parents[1])
            / "fairness_differences.csv"),
    )
    write_to_disk(
        fairness_report_json(metrics_df, diff_df, all_warnings,
                             limitation_text=FAIRNESS_LIMITATION),
        str(EVALUATION_DIR.relative_to(EVALUATION_DIR.parents[1])
            / "fairness_report.json"),
    )

    log.info("Phase 6 fairness audit complete.")


if __name__ == "__main__":
    main()