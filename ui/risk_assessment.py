"""Individual risk assessment page."""

import streamlit as st

from src.explainability.shap_explainer import explain_single
from src.prediction.predictor import predict_single, DEFAULT_MODEL_NAME
from src.reporting.report_generator import (
    prediction_report_csv, prediction_report_json,
)
from ui import components as c
from ui import charts, styles
from ui import _cache


def render() -> None:
    c.page_title(
        "Individual Risk Assessment",
        "Score a single customer and understand which factors drove the estimate.",
    )

    try:
        _cache.get_model(DEFAULT_MODEL_NAME)
    except FileNotFoundError as exc:
        c.error_panel(str(exc), "python -m scripts.train_models")
        return

    mode = st.radio(
        "Input source",
        options=["Manual entry", "Load from dataset"],
        horizontal=True,
        label_visibility="collapsed",
    )

    default_record = None
    if mode == "Load from dataset":
        try:
            raw = _cache.get_raw_data()
        except FileNotFoundError as exc:
            c.error_panel(str(exc))
            return
        idx = st.number_input(
            "Dataset row index", min_value=0, max_value=len(raw) - 1,
            value=0, step=1,
        )
        row = raw.iloc[int(idx)]
        default_record = {
            col: row[col] for col in
            ("LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
             "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
             "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
             "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
             "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
             "PAY_AMT4", "PAY_AMT5", "PAY_AMT6")
        }
        actual = int(row["default payment next month"]) if "default payment next month" in row.index else None
        if actual is not None:
            st.caption(
                f"Actual outcome in dataset: "
                f"{'defaulted next month' if actual == 1 else 'did not default next month'}."
            )

    c.section_header("Customer inputs")
    record = c.record_input_form("ra", defaults=default_record)

    col_a, col_b = st.columns([1, 3])
    with col_a:
        predict_clicked = st.button("Predict and explain", type="primary", use_container_width=True)
    with col_b:
        if "ra_result" in st.session_state:
            if st.button("Clear result", type="secondary"):
                st.session_state.pop("ra_result", None)
                st.session_state.pop("ra_explanation", None)
                st.rerun()

    if predict_clicked:
        with st.spinner("Scoring and computing explanations..."):
            prediction = predict_single(record, model_name=DEFAULT_MODEL_NAME)
            explanation = explain_single(record, model_name=DEFAULT_MODEL_NAME)
            st.session_state["ra_result"] = prediction
            st.session_state["ra_explanation"] = explanation

    if "ra_result" not in st.session_state:
        c.info_panel(
            "Enter customer information above and click <b>Predict and explain</b> "
            "to see the estimated default probability and its SHAP-based breakdown.",
        )
        return

    result = st.session_state["ra_result"]
    explanation = st.session_state["ra_explanation"]

    # ---- Result card -----------------------------------------------------
    st.markdown("---")
    left, right = st.columns([1, 1])

    with left:
        st.markdown(c.probability_block(result.probability, result.risk_band),
                    unsafe_allow_html=True)

    with right:
        st.markdown(
            f"""
            <div class="cl-card">
                <div class="cl-kpi-label">Details</div>
                <table style="width:100%; border-collapse: collapse; margin-top:8px;
                              font-size: 14px; color: {styles.NAVY};">
                    <tr><td style="padding:6px 0; color:{styles.MUTED};">Model</td>
                        <td style="text-align:right;">{result.model_name.replace('_',' ').title()}</td></tr>
                    <tr><td style="padding:6px 0; color:{styles.MUTED};">Decision threshold</td>
                        <td style="text-align:right;">{result.threshold:.2f}</td></tr>
                    <tr><td style="padding:6px 0; color:{styles.MUTED};">Predicted class</td>
                        <td style="text-align:right;">{'Default' if result.predicted_class == 1 else 'No default'}</td></tr>
                    <tr><td style="padding:6px 0; color:{styles.MUTED};">Scored (UTC)</td>
                        <td style="text-align:right;">{result.timestamp_utc}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- SHAP explanation -----------------------------------------------
    c.section_header(
        "Why the model produced this estimate",
        "SHAP values describe how each feature contributes to the model's "
        "output under its explanation framework. They do not establish "
        "causation.",
    )

    st.plotly_chart(
        charts.shap_waterfall(
            explanation.original_features,
            explanation.original_shap_values,
            explanation.original_input_values,
            explanation.base_value,
        ),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    top = explanation.top_contributors(n=5)
    st.markdown("**Strongest contributors**")
    for t in top:
        direction_colour = styles.DANGER if t["shap_value"] > 0 else styles.ACCENT
        st.markdown(
            f"""
            <div style="padding: 8px 0; border-bottom: 1px solid {styles.BORDER};">
                <span style="font-weight:600; color:{styles.NAVY};">{t['feature']}</span>
                <span style="color:{styles.MUTED};"> &mdash; input value: {t['input_value']}</span>
                <span style="float:right; color:{direction_colour}; font-weight:600;">
                    {t['shap_value']:+.4f}
                </span>
                <div style="color:{styles.MUTED}; font-size:13px; margin-top:2px;">
                    {t['direction'].capitalize()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Downloads -------------------------------------------------------
    st.markdown("")
    st.markdown("**Download this assessment**")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download JSON report",
            data=prediction_report_json(result, explanation, top_k=10),
            file_name=f"creditlens_prediction_{result.timestamp_utc[:10]}.json",
            mime="application/json",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "Download CSV summary",
            data=prediction_report_csv(result),
            file_name=f"creditlens_prediction_{result.timestamp_utc[:10]}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    c.disclaimer(
        "This is a model-based estimate intended for analysis and "
        "demonstration. It is not a lending decision and must not be used "
        "to approve or reject a real applicant."
    )