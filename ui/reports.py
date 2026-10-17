"""Reports page — centralised export centre."""

import streamlit as st

from src.reporting.report_generator import (
    comparison_report_csv, fairness_report_csv, fairness_report_json,
    model_metadata_json,
)
from src.fairness.audit import FAIRNESS_LIMITATION
from ui import components as c
from ui import _cache


def render() -> None:
    c.page_title(
        "Reports",
        "Download analysis outputs. All payloads are generated from real "
        "application state.",
    )

    c.info_panel(
        "For an individual prediction report, use the <b>Risk Assessment</b> "
        "page. For a what-if scenario report, use the <b>What-if Simulator</b> "
        "page. This page collects the aggregate exports.",
    )

    c.section_header("Model metadata")
    try:
        comparison = _cache.get_comparison()
    except FileNotFoundError as exc:
        c.error_panel(str(exc), "python -m scripts.train_models")
        return

    metric_dict = comparison.to_dict(orient="index")
    st.download_button(
        "Download model metadata JSON",
        data=model_metadata_json(
            metric_dict,
            notes=[
                "Random seed 42.",
                "Stratified 80/20 split.",
                "Random Forest selected by PR-AUC then ROC-AUC.",
            ],
        ),
        file_name="creditlens_model_metadata.json",
        mime="application/json",
    )

    c.section_header("Model comparison")
    c.data_table(comparison.reset_index(), precision=4)
    st.download_button(
        "Download comparison CSV",
        data=comparison_report_csv(comparison),
        file_name="creditlens_model_comparison.csv",
        mime="text/csv",
    )

    c.section_header("Fairness audit exports")
    try:
        metrics_df, diffs_df = _cache.get_fairness()
    except FileNotFoundError:
        c.info_panel(
            "Fairness files not found. Run: "
            "<code>python -m scripts.run_fairness_audit</code>",
            kind="warning",
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download fairness CSV",
                data=fairness_report_csv(metrics_df, diffs_df),
                file_name="creditlens_fairness_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "Download fairness JSON",
                data=fairness_report_json(metrics_df, diffs_df,
                                          limitation_text=FAIRNESS_LIMITATION),
                file_name="creditlens_fairness_report.json",
                mime="application/json",
                use_container_width=True,
            )

    c.disclaimer(
        "Every export reflects the current state of the application's models, "
        "splits, and evaluation artefacts."
    )