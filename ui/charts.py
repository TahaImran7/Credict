"""
Plotly chart builders for CreditLens.

Every chart uses the same layout factory so the whole dashboard looks
consistent. No emoji, no default Plotly colors.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config import settings
from ui import styles


def _base_layout(fig: go.Figure, height: int = 380, showlegend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family=styles.FONT_STACK, size=12, color=styles.NAVY),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=showlegend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=12),
        ),
        hoverlabel=dict(
            bgcolor="white", bordercolor=styles.BORDER,
            font=dict(family=styles.FONT_STACK, color=styles.NAVY),
        ),
        colorway=styles.CHART_SEQUENCE,
    )
    fig.update_xaxes(
        gridcolor=styles.BORDER, zerolinecolor=styles.BORDER,
        linecolor=styles.BORDER, tickfont=dict(color=styles.MUTED),
        title_font=dict(color=styles.SLATE, size=12),
    )
    fig.update_yaxes(
        gridcolor=styles.BORDER, zerolinecolor=styles.BORDER,
        linecolor=styles.BORDER, tickfont=dict(color=styles.MUTED),
        title_font=dict(color=styles.SLATE, size=12),
    )
    return fig


# ---------------------------------------------------------------------------
# Portfolio charts
# ---------------------------------------------------------------------------

def risk_band_bar(counts_df: pd.DataFrame) -> go.Figure:
    colors = [styles.RISK_COLORS.get(b, styles.ACCENT) for b in counts_df["risk_band"]]
    fig = go.Figure(go.Bar(
        x=counts_df["risk_band"], y=counts_df["count"],
        marker=dict(color=colors),
        text=counts_df["count"], textposition="outside",
        hovertemplate="<b>%{x}</b><br>Customers: %{y}<extra></extra>",
    ))
    fig.update_layout(title="Customers per predicted risk band", bargap=0.35)
    fig.update_yaxes(title="Customers")
    fig.update_xaxes(title="")
    return _base_layout(fig, height=380)


def probability_histogram(probs: np.ndarray, bins: int = 40) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=probs, nbinsx=bins,
        marker=dict(color=styles.ACCENT, line=dict(color="white", width=0.5)),
        hovertemplate="Probability %{x:.2f}<br>Customers: %{y}<extra></extra>",
    ))
    fig.update_layout(title="Distribution of predicted default probability", bargap=0.02)
    fig.update_xaxes(title="Predicted probability", tickformat=".0%")
    fig.update_yaxes(title="Customers")
    return _base_layout(fig, height=380)


def default_rate_by_band_bar(df: pd.DataFrame) -> go.Figure:
    colors = [styles.RISK_COLORS.get(b, styles.ACCENT) for b in df["risk_band"]]
    fig = go.Figure(go.Bar(
        x=df["risk_band"], y=df["observed_default_rate"],
        marker=dict(color=colors),
        text=[f"{v:.1%}" for v in df["observed_default_rate"]],
        textposition="outside",
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
    fig = go.Figure(go.Bar(
        x=df["category"].astype(str), y=df["default_rate"],
        marker=dict(color=[
            styles.WARNING if lc else styles.ACCENT
            for lc in df["low_confidence"]
        ]),
        text=[f"{v:.1%}" for v in df["default_rate"]],
        textposition="outside",
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
    fig = go.Figure()
    for i, (name, data) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(
            x=data["fpr"], y=data["tpr"], mode="lines",
            name=name,
            line=dict(color=styles.CHART_SEQUENCE[i % len(styles.CHART_SEQUENCE)], width=2),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random (0.5)",
        line=dict(color=styles.LIGHT, width=1, dash="dash"),
        hoverinfo="skip",
    ))
    fig.update_layout(title="ROC curves", xaxis_title="False positive rate",
                      yaxis_title="True positive rate")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return _base_layout(fig, height=420, showlegend=True)


def pr_curve_chart(curves: dict[str, dict]) -> go.Figure:
    fig = go.Figure()
    for i, (name, data) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(
            x=data["recall"], y=data["precision"], mode="lines",
            name=name,
            line=dict(color=styles.CHART_SEQUENCE[i % len(styles.CHART_SEQUENCE)], width=2),
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(title="Precision-recall curves",
                      xaxis_title="Recall", yaxis_title="Precision")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return _base_layout(fig, height=420, showlegend=True)


def calibration_chart(curves: dict[str, dict]) -> go.Figure:
    fig = go.Figure()
    for i, (name, data) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(
            x=data["mean_predicted"], y=data["fraction_positive"],
            mode="lines+markers", name=name,
            line=dict(color=styles.CHART_SEQUENCE[i % len(styles.CHART_SEQUENCE)], width=2),
            marker=dict(size=8),
            hovertemplate="Mean predicted: %{x:.3f}<br>Fraction positive: %{y:.3f}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
        line=dict(color=styles.LIGHT, width=1, dash="dash"),
        hoverinfo="skip",
    ))
    fig.update_layout(title="Calibration curve", xaxis_title="Mean predicted probability",
                      yaxis_title="Fraction of positives")
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return _base_layout(fig, height=420, showlegend=True)


def confusion_matrix_chart(cm: list[list[int]], model_name: str) -> go.Figure:
    cm_arr = np.array(cm)
    labels = ["Non-default", "Default"]
    fig = go.Figure(go.Heatmap(
        z=cm_arr,
        x=[f"Predicted<br>{l}" for l in labels],
        y=[f"Actual<br>{l}" for l in labels],
        colorscale=[[0, styles.SURFACE_ALT], [1, styles.ACCENT]],
        showscale=False,
        text=cm_arr, texttemplate="%{text}", textfont=dict(size=18, color=styles.NAVY),
        hovertemplate="Actual %{y}<br>Predicted %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(title=f"{model_name}: confusion matrix", yaxis=dict(autorange="reversed"))
    return _base_layout(fig, height=360)


# ---------------------------------------------------------------------------
# Explainability charts
# ---------------------------------------------------------------------------

def global_importance_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    sub = df.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=sub["mean_abs_shap"], y=sub["feature"], orientation="h",
        marker=dict(color=styles.ACCENT),
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
    """
    Horizontal bar chart of SHAP contributions, sorted by |SHAP| descending.
    Positive (risk-increasing) in red, negative (risk-decreasing) in teal.
    """
    pairs = sorted(zip(features, shap_values, input_values),
                   key=lambda t: abs(t[1]))
    feats = [p[0] for p in pairs]
    svs = [p[1] for p in pairs]
    inputs = [p[2] for p in pairs]
    colors = [styles.DANGER if v > 0 else styles.ACCENT for v in svs]

    text = [f"  {v:+.4f}  (input: {i})" for v, i in zip(svs, inputs)]

    fig = go.Figure(go.Bar(
        x=svs, y=feats, orientation="h",
        marker=dict(color=colors),
        text=text, textposition="outside",
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=styles.BORDER_STRONG, width=1))
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
    sub = df[df["audit"] == audit].copy()
    sub["group"] = sub["group"].astype(str)
    ref_str = str(reference)
    colors = [styles.MUTED if g == ref_str else styles.ACCENT for g in sub["group"]]
    fig = go.Figure(go.Bar(
        x=sub["group"], y=sub[metric],
        marker=dict(color=colors),
        text=[f"{v:.3f}" if pd.notna(v) else "n/a" for v in sub[metric]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>" + metric + ": %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(title=f"{metric} by {audit} (reference = {ref_str})")
    fig.update_xaxes(title=audit)
    fig.update_yaxes(title=metric)
    return _base_layout(fig, height=380)


def fairness_difference_bar(
    diff_df: pd.DataFrame, metric: str, audit: str,
) -> go.Figure:
    sub = diff_df[diff_df["audit"] == audit].copy()
    sub["group"] = sub["group"].astype(str)
    col = f"{metric}_minus_ref"
    if col not in sub.columns:
        return None
    values = sub[col].astype(float)
    colors = [
        styles.DANGER if (pd.notna(v) and v > 0)
        else styles.ACCENT if (pd.notna(v) and v < 0)
        else styles.MUTED
        for v in values
    ]
    fig = go.Figure(go.Bar(
        x=sub["group"], y=values,
        marker=dict(color=colors),
        text=[f"{v:+.4f}" if pd.notna(v) else "n/a" for v in values],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Difference: %{y:+.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=styles.BORDER_STRONG, width=1))
    fig.update_layout(title=f"{metric}: difference from reference")
    fig.update_xaxes(title=audit)
    fig.update_yaxes(title=f"{metric} minus reference")
    return _base_layout(fig, height=380)