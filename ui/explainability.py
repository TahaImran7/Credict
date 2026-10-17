"""Explainability page."""

import streamlit as st

from src.explainability.shap_explainer import explain_single
from src.prediction.predictor import DEFAULT_MODEL_NAME, predict_single
from ui import components as c
from ui import charts
from ui import _cache


def render() -> None:
    c.page_title(
        "Explainability",
        "Global feature importance and per-customer SHAP-based explanations.",
    )

    try:
        _cache.get_model(DEFAULT_MODEL_NAME)
        importance = _cache.get_global_importance()
    except FileNotFoundError as exc:
        c.error_panel(str(exc), "python -m scripts.build_dashboard_artifacts")
        return

    c.section_header(
        "Global feature importance",
        "Mean absolute SHAP value across the full test set.",
    )
    top_n = st.slider("How many features to display", 5, 23, 15, key="expl_topn")
    st.plotly_chart(charts.global_importance_bar(importance, top_n=top_n),
                    use_container_width=True, config={"displayModeBar": False})

    with st.expander("Full importance table"):
        c.data_table(importance, precision=4)

    c.section_header(
        "Individual explanation",
        "Enter a customer record and see which features pushed the estimate up or down.",
    )

    mode = st.radio(
        "Input source",
        options=["Manual entry", "Load from dataset"],
        horizontal=True, label_visibility="collapsed", key="expl_mode",
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
            value=0, step=1, key="expl_idx",
        )
        row = raw.iloc[int(idx)]
        default_record = {col: row[col] for col in (
            "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
            "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
            "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
            "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
        )}

    record = c.record_input_form("expl", defaults=default_record)

    if st.button("Compute explanation", type="primary"):
        with st.spinner("Computing SHAP values..."):
            prediction = predict_single(record, model_name=DEFAULT_MODEL_NAME)
            explanation = explain_single(record, model_name=DEFAULT_MODEL_NAME)
            st.session_state["expl_pred"] = prediction
            st.session_state["expl_exp"] = explanation

    if "expl_exp" not in st.session_state:
        return

    prediction = st.session_state["expl_pred"]
    explanation = st.session_state["expl_exp"]

    st.markdown("---")
    st.markdown(c.probability_block(prediction.probability, prediction.risk_band),
                unsafe_allow_html=True)
    st.markdown("")

    st.plotly_chart(
        charts.shap_waterfall(
            explanation.original_features,
            explanation.original_shap_values,
            explanation.original_input_values,
            explanation.base_value,
        ),
        use_container_width=True, config={"displayModeBar": False},
    )

    c.disclaimer(
        "SHAP explains how the model used each feature under its own "
        "explanation framework. It does not prove causality."
    )