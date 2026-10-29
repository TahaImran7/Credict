"""
Plotly chart builders for Credict.

Every chart is theme-aware. Multi-line traces use styles.CHART_SEQUENCE.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import settings
from ui import styles


def _palette() -> dict:
    theme = st.session_state.get("theme", "light")
    return styles.get_palette(theme)


def _theme_name() -> str:
    return st.session_state.get("theme", "light")


def _risk_map() -> dict:
    p = _palette()
    return {
        "Low": p["RISK_LOW"],
        "Moderate": p["RISK_MODERATE"],
        "High": p["RISK_HIGH"],
        "Very High": p["RISK_VERY_HIGH"],
    }


def _base_layout(fig: go.Figure, height: int = 380,
                 showlegend: bool = False) -> go.Figure:
    p = _palette()
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=48, b=20),
        font=dict(family=styles.FONT_STACK, size=12, color=p["NAVY"]),
        plot_bgcolor=p["SURFACE"],
        paper_bgcolor=p["SURFACE"],
        showlegend=showlegend,
        # ---- Main title colour (fixes the "black title in dark mode" issue)
        title=dict(
            font=dict(
                family=styles.FONT_STACK,
                size=14,
                color=p["NAVY"],
            ),
            x=0.02,
            xanchor="left",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=12, color=p["MUTED"]),
        ),
        hoverlabel=dict(
            bgcolor=p["SURFACE"], bordercolor=p["BORDER"],
            font=dict(family=styles.FONT_STACK, color=p["NAVY"]),
        ),
        colorway=styles.CHART_SEQUENCE,
    )
    fig.update_xaxes(
        gridcolor=p["BORDER"], zerolinecolor=p["BORDER"],
        linecolor=p["BORDER"], tickfont=dict(color=p["MUTED"]),
        title_font=dict(color=p["SLATE"], size=12),
    )
    fig.update_yaxes(
        gridcolor=p["BORDER"], zerolinecolor=p["BORDER"],
        linecolor=p["BORDER"], tickfont=dict(color=p["MUTED"]),
        title_font=dict(color=p["SLATE"], size=12),
    )
    return fig


# ---------------------------------------------------------------------------
# Portfolio charts
# ---------------------------------------------------------------------------

def risk_band_bar(counts_df: pd.DataFrame) -> go.Figure:
    p = _palette()
    rmap = _risk_map()
    colors = [rmap.get(b, p["ACCENT"]) for b in counts_df["risk_band"]]
    fig = go.Figure(go.Bar(
        x=counts_df["risk_band"], y=counts_df["count"],
        marker=dict(color=colors),
        text=counts_df["count"], textposition="outside",
        textfont=dict(color=p["NAVY"]),
        hovertemplate="<b>%{x}</b><br>Customers: %{y}<extra></extra>",
    ))
    fig.update_layout(title="Customers per predicted risk band", bargap=0.35)
    fig.update_yaxes(title="Customers")
    fig.update_xaxes(title="")
    return _base_layout(fig, height=380)


def probability_histogram(probs: np.ndarray, bins: int = 40) -> go.Figure:
    p = _palette()
    fig = go.Figure(go.Histogram(
        x=probs, nbinsx=bins,
        marker=dict(color=p["ACCENT"], line=dict(color=p["SURFACE"], width=0.5)),
        hovertemplate="Probability %{x:.2f}<br>Customers: %{y}<extra></extra>",
    ))
    fig.update_layout(title="Distribution of predicted default probability", bargap=0.02)
    fig.update_xaxes(title="Predicted probability", tickformat=".0%")
    fig.update_yaxes(title="Customers")
    return _base_layout(fig, height=380)


def default_rate_by_band_bar(df: pd.DataFrame) -> go.Figure:
    p = _palette()
    rmap = _risk_map()
    colors = [rmap.get(b, p["ACCENT"]) for b in df["risk_band"]]
    fig = go.Figure(go.Bar(
        x=df["risk_band"], y=df["observed_default_rate"],
        marker=dict(color=colors),
        text=[f"{v:.1%}" for v in df["observed_default_rate"]],
        textposition="outside",
        textfont=dict(color=p["NAVY"]),
        hovertemplate="<b>%{x}</b><br>Observed default rate: %{y:.2%}"
                      "<br>Customers: %{customdata[0]}<extra></extra>",
        customdata=df[["n"]].values,
    ))
    fig.update_layout(title="Observed default rate per predicted risk band")
    fig.update_yaxes(title="Observed default rate", tickformat=".0%")
    fig.update_xaxes(title="")
    return _base_layout(fig, height=380)


def category_default_rate_bar(
    df: pd.DataFrame, category_col: str, title: str,
) -> go.Figure:
    p = _palette()
    fig = go.Figure(go.Bar(
        x=df["category"].astype(str), y=df["default_rate"],
        marker=dict(color=[
            p["WARNING"] if lc else p["ACCENT"]
            for lc in df["low_confidence"]
        ]),
        text=[f"{v:.1%}" for v in df["default_rate"]],
        textposition="outside",
        textfont=dict(color=p["NAVY"]),
        hovertemplate=("<b>%{x}</b><br>Default rate: %{y:.2%}"
                       "<br>Records: %{customdata[0]}<extra></extra>"),
        customdata=df[["n"]].values,
    ))
    fig.update_layout(title=title, bargap=0.35)
    fig.update_yaxes(title="Default rate", tickformat=".0%")
    fig.update_xaxes(title=category_col)
    return _base_layout(fig, height=380)


# ---------------------------------------------------------------------------
# Model performance charts
# ---------------------------------------------------------------------------

def roc_curve_chart(curves: dict[str, dict]) -> go.Figure:
    p = _palette()
    fig = go.Figure()
    for i, (name, data) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"], mode="lines",
            name=name,
            line=dict(
                color=styles.CHART_SEQUENCE[i % len(styles.CHART_SEQUENCE)],
                width=2,
            ),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random (0.5)",
        line=dict(color=p["LIGHT"], width=1, dash="dash"),
        hoverinfo="skip",
    ))
    fig.update_layout(
        title="ROC curves",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return _base_layout(fig, height=420, showlegend=True)


def pr_curve_chart(curves: dict[str, dict]) -> go.Figure:
    p = _palette()
    fig = go.Figure()
    for i, (name, data) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(
            x=data["recall"], y=data["precision"], mode="lines",
            name=name,
            line=dict(
                color=styles.CHART_SEQUENCE[i % len(styles.CHART_SEQUENCE)],
                width=2,
            ),
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        title="Precision-recall curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return _base_layout(fig, height=420, showlegend=True)


def calibration_chart(curves: dict[str, dict]) -> go.Figure:
    p = _palette()
    fig = go.Figure()
    for i, (name, data) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(
            x=data["mean_predicted"], y=data["fraction_positive"],
            mode="lines+markers", name=name,
            line=dict(
                color=styles.CHART_SEQUENCE[i % len(styles.CHART_SEQUENCE)],
                width=2,
            ),
            marker=dict(size=8),
            hovertemplate="Mean predicted: %{x:.3f}"
                          "<br>Fraction positive: %{y:.3f}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
        line=dict(color=p["LIGHT"], width=1, dash="dash"),
        hoverinfo="skip",
    ))
    fig.update_layout(
        title="Calibration curve",
        xaxis_title="Mean predicted probability",
        yaxis_title="Fraction of positives",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return _base_layout(fig, height=420, showlegend=True)


def confusion_matrix_chart(cm: list[list[int]], model_name: str) -> go.Figure:
    p = _palette()
    theme = _theme_name()
    cm_arr = np.array(cm)
    labels = ["Non-default", "Default"]
    x_labels = [f"Predicted<br>{l}" for l in labels]
    y_labels = [f"Actual<br>{l}" for l in labels]

    fig = go.Figure(go.Heatmap(
        z=cm_arr,
        x=x_labels, y=y_labels,
        colorscale=[[0, p["SURFACE_ALT"]], [1, p["ACCENT"]]],
        showscale=False,
        hovertemplate="Actual %{y}<br>Predicted %{x}"
                      "<br>Count: %{z}<extra></extra>",
    ))

    vmax = int(cm_arr.max()) if cm_arr.size else 1
    vmax = max(vmax, 1)
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            val = int(cm_arr[i, j])
            intensity = val / vmax
            if intensity > 0.5:
                text_colour = "white" if theme == "light" else p["SIDEBAR_BG"]
            else:
                text_colour = p["NAVY"]
            fig.add_annotation(
                x=x_labels[j], y=y_labels[i],
                text=str(val), showarrow=False,
                font=dict(size=18, color=text_colour),
            )

    fig.update_layout(
        title=f"{model_name}: confusion matrix",
        yaxis=dict(autorange="reversed"),
    )
    return _base_layout(fig, height=360)


# ---------------------------------------------------------------------------
# Explainability charts
# ---------------------------------------------------------------------------

def global_importance_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    p = _palette()
    sub = df.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=sub["mean_abs_shap"], y=sub["feature"], orientation="h",
        marker=dict(color=p["ACCENT"]),
        text=[f"{v:.4f}" for v in sub["mean_abs_shap"]],
        textposition="outside",
        textfont=dict(color=p["NAVY"]),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(title=f"Top {top_n} features by mean |SHAP|")
    fig.update_xaxes(title="Mean |SHAP value|")
    fig.update_yaxes(title="")
    return _base_layout(fig, height=max(360, 22 * top_n))


def shap_waterfall(
    features: list[str], shap_values: list[float], input_values: list,
    base_value: float,
) -> go.Figure:
    p = _palette()
    pairs = sorted(zip(features, shap_values, input_values),
                   key=lambda t: abs(t[1]))
    feats = [pr[0] for pr in pairs]
    svs = [pr[1] for pr in pairs]
    inputs = [pr[2] for pr in pairs]
    colors = [p["DANGER"] if v > 0 else p["ACCENT"] for v in svs]
    text = [f"  {v:+.4f}  (input: {i})" for v, i in zip(svs, inputs)]

    fig = go.Figure(go.Bar(
        x=svs, y=feats, orientation="h",
        marker=dict(color=colors),
        text=text, textposition="outside",
        textfont=dict(color=p["NAVY"]),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=p["BORDER_STRONG"], width=1))
    fig.update_layout(
        title=f"SHAP contributions (base value = {base_value:.4f})",
        xaxis_title="SHAP value (positive increases predicted risk)",
        yaxis_title="",
    )
    return _base_layout(fig, height=max(380, 28 * len(feats)))


# ---------------------------------------------------------------------------
# Fairness charts
# ---------------------------------------------------------------------------

def fairness_group_bar(
    df: pd.DataFrame, metric: str, audit: str, reference: object,
) -> go.Figure:
    p = _palette()
    sub = df[df["audit"] == audit].copy()
    sub["group"] = sub["group"].astype(str)
    ref_str = str(reference)
    colors = [p["MUTED"] if g == ref_str else p["ACCENT"] for g in sub["group"]]
    fig = go.Figure(go.Bar(
        x=sub["group"], y=sub[metric],
        marker=dict(color=colors),
        text=[f"{v:.3f}" if pd.notna(v) else "n/a" for v in sub[metric]],
        textposition="outside",
        textfont=dict(color=p["NAVY"]),
        hovertemplate="<b>%{x}</b><br>" + metric + ": %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(title=f"{metric} by {audit} (reference = {ref_str})")
    fig.update_xaxes(title=audit)
    fig.update_yaxes(title=metric)
    return _base_layout(fig, height=380)


def fairness_difference_bar(
    diff_df: pd.DataFrame, metric: str, audit: str,
) -> go.Figure | None:
    p = _palette()
    sub = diff_df[diff_df["audit"] == audit].copy()
    sub["group"] = sub["group"].astype(str)
    col = f"{metric}_minus_ref"
    if col not in sub.columns:
        return None
    values = sub[col].astype(float)
    colors = [
        p["DANGER"] if (pd.notna(v) and v > 0)
        else p["ACCENT"] if (pd.notna(v) and v < 0)
        else p["MUTED"]
        for v in values
    ]
    fig = go.Figure(go.Bar(
        x=sub["group"], y=values,
        marker=dict(color=colors),
        text=[f"{v:+.4f}" if pd.notna(v) else "n/a" for v in values],
        textposition="outside",
        textfont=dict(color=p["NAVY"]),
        hovertemplate="<b>%{x}</b><br>Difference: %{y:+.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=p["BORDER_STRONG"], width=1))
    fig.update_layout(title=f"{metric}: difference from reference")
    fig.update_xaxes(title=audit)
    fig.update_yaxes(title=f"{metric} minus reference")
    return _base_layout(fig, height=380)