"""Model performance page."""

import pandas as pd
import streamlit as st

from src.reporting.report_generator import comparison_report_csv
from ui import components as c
from ui import charts, styles
from ui import _cache


def render() -> None:
    c.page_title(
        "Model Performance",
        "Held-out test-set evaluation of every trained model. All values come "
        "from actual training runs.",
    )

    try:
        comparison = _cache.get_comparison()
    except FileNotFoundError as exc:
        c.error_panel(str(exc), "python -m scripts.train_models")
        return

    models = list(comparison.index)

    # ---- Comparison table -----------------------------------------------
    c.section_header("Comparison summary")
    display = comparison.reset_index().rename(columns={
        "model": "Model", "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC", "brier": "Brier",
        "n_test": "Test n", "n_positive": "Positives", "threshold": "Threshold",
    })
    display["Model"] = display["Model"].str.replace("_", " ").str.title()
    st.dataframe(display.round(4), use_container_width=True, hide_index=True)

    st.download_button(
        "Download comparison CSV",
        data=comparison_report_csv(comparison),
        file_name="creditlens_model_comparison.csv",
        mime="text/csv",
    )

    # ---- Curves ---------------------------------------------------------
    c.section_header("ROC, precision-recall, and calibration")
    evaluations = {m: _cache.get_evaluation(m) for m in models}

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            charts.roc_curve_chart({
                m.replace("_", " ").title(): ev["roc_curve"]
                for m, ev in evaluations.items()
            }),
            use_container_width=True, config={"displayModeBar": False},
        )
    with col2:
        st.plotly_chart(
            charts.pr_curve_chart({
                m.replace("_", " ").title(): ev["pr_curve"]
                for m, ev in evaluations.items()
            }),
            use_container_width=True, config={"displayModeBar": False},
        )

    st.plotly_chart(
        charts.calibration_chart({
            m.replace("_", " ").title(): ev["calibration"]
            for m, ev in evaluations.items()
        }),
        use_container_width=True, config={"displayModeBar": False},
    )

    # ---- Confusion matrices ---------------------------------------------
    c.section_header("Confusion matrices")
    cols = st.columns(len(models))
    for col, m in zip(cols, models):
        with col:
            st.plotly_chart(
                charts.confusion_matrix_chart(
                    evaluations[m]["confusion_matrix"],
                    m.replace("_", " ").title(),
                ),
                use_container_width=True, config={"displayModeBar": False},
            )

    # ---- Metric explanations --------------------------------------------
    c.section_header("What the metrics mean in financial terms")
    st.markdown(
        """
**Accuracy** — share of test customers whose outcome was classified correctly.
Can be misleading when classes are imbalanced; the majority class alone is
about 78% of the data.

**Precision** — of the customers the model flagged as likely defaulters,
what share actually defaulted. Low precision means more customers are
incorrectly flagged.

**Recall** — of the customers who did default, what share the model caught.
Low recall means more defaults are missed.

**F1** — harmonic mean of precision and recall. Useful when both matter.

**ROC-AUC** — probability that a random defaulter is scored above a random
non-defaulter. 0.5 is random; higher is better.

**PR-AUC** — area under the precision-recall curve. More informative than
ROC-AUC under class imbalance, and the primary selection metric here.

**Brier score** — mean squared error of predicted probabilities. Lower is
better; measures both discrimination and calibration.

For lending policy, the correct trade-off between precision and recall
depends on the cost of missed defaults versus the cost of wrongly rejecting
creditworthy applicants. This dashboard does not decide that trade-off.
        """
    )

    c.disclaimer(
        "Metrics are computed on a single stratified 20% hold-out split with "
        "random seed 42. They will vary slightly if the split changes. No "
        "hyperparameter was tuned on this test set."
    )