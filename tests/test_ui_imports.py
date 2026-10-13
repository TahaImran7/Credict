"""
Phase 7 — UI module import tests.

These do NOT run Streamlit (which requires a runtime). They only verify that
every UI module imports cleanly and that key public callables exist, so a
typo or import error is caught by `pytest` before the app is launched.
"""

import importlib

import pytest


MODULES = [
    "ui.styles",
    "ui._cache",
    "ui.components",
    "ui.charts",
    "ui.overview",
    "ui.risk_assessment",
    "ui.simulator",
    "ui.model_performance",
    "ui.explainability",
    "ui.portfolio_analytics",
    "ui.fairness_audit",
    "ui.reports",
]


@pytest.mark.parametrize("name", MODULES)
def test_ui_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_styles_exports_palette() -> None:
    from ui import styles
    for attr in ("NAVY", "ACCENT", "BORDER", "SURFACE", "RISK_COLORS",
                 "CHART_SEQUENCE", "inject"):
        assert hasattr(styles, attr), f"Missing ui.styles.{attr}"


def test_components_exports_key_functions() -> None:
    from ui import components as c
    for name in ("section_header", "page_title", "kpi_card", "kpi_row",
                 "risk_band_badge", "probability_block", "info_panel",
                 "disclaimer", "empty_state", "record_input_form"):
        assert callable(getattr(c, name)), f"Missing components.{name}"


def test_charts_exports_key_functions() -> None:
    from ui import charts
    for name in ("risk_band_bar", "probability_histogram",
                 "default_rate_by_band_bar", "roc_curve_chart",
                 "pr_curve_chart", "calibration_chart",
                 "confusion_matrix_chart", "global_importance_bar",
                 "shap_waterfall", "fairness_group_bar",
                 "fairness_difference_bar"):
        assert callable(getattr(charts, name)), f"Missing charts.{name}"


def test_each_page_has_render() -> None:
    for mod_name in ("ui.overview", "ui.risk_assessment", "ui.simulator",
                     "ui.model_performance", "ui.explainability",
                     "ui.portfolio_analytics", "ui.fairness_audit",
                     "ui.reports"):
        mod = importlib.import_module(mod_name)
        assert callable(getattr(mod, "render", None)), f"{mod_name}.render missing"


def test_no_emoji_in_source_strings() -> None:
    """
    Guard: the UI must contain no emoji characters. We scan the ui/ package
    source files for characters in the common emoji ranges.
    """
    from pathlib import Path

    def _has_emoji(text: str) -> bool:
        for ch in text:
            cp = ord(ch)
            if (0x1F300 <= cp <= 0x1FAFF or     # Misc emoji & symbols
                0x1F000 <= cp <= 0x1F2FF or     # Mahjong/dominoes/etc.
                0x2600 <= cp <= 0x27BF or       # Misc symbols & dingbats
                0x1F900 <= cp <= 0x1F9FF):      # Supplemental symbols
                return True
        return False

    root = Path(__file__).resolve().parents[1] / "ui"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if _has_emoji(text):
            offenders.append(str(path.relative_to(root.parent)))
    assert not offenders, f"Emoji found in UI files: {offenders}"