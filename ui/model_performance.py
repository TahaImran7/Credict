"""Model performance page."""

import streamlit as st

from src.reporting.report_generator import comparison_report_csv
from ui import components as c
from ui import charts
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

    c.section_header("Comparison summary")
    display = comparison.reset_index().rename(columns={
        "model": "Model", "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC", "brier": "Brier",
        "n_test": "Test n", "n_positive": "Positives", "threshold": "Threshold",
    })
    display["Model"] = display["Model"].str.replace("_", " ").str.title()
    c.data_table(display, precision=4)

    st.download_button(
        "Download comparison CSV",
        data=comparison_report_csv(comparison),
        file_name="creditlens_model_comparison.csv",
        mime="text/csv",
    )

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

    c.section_header("What the metrics mean in financial terms")
    st.markdown(
        """
**Accuracy** — share of test customers whose outcome was classified correctly.

**Precision** — of the customers flagged as likely defaulters, what share actually defaulted.

**Recall** — of the customers who did default, what share the model caught.

**F1** — harmonic mean of precision and recall.

**ROC-AUC** — probability that a random defaulter is scored above a random non-defaulter.

**PR-AUC** — area under the precision-recall curve. More informative than ROC-AUC under class imbalance.

**Brier score** — mean squared error of predicted probabilities. Lower is better.
        """
    )

    c.disclaimer(
        "Metrics are computed on a single stratified 20% hold-out split with "
        "random seed 42. No hyperparameter was tuned on this test set."
    )