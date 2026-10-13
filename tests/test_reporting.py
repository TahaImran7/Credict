"""
Phase 6 — report generation tests.
"""

import json

import pandas as pd
import pytest

from src.reporting.report_generator import (
    comparison_report_csv,
    fairness_report_csv,
    fairness_report_json,
    model_metadata_json,
    prediction_report_csv,
    prediction_report_json,
    whatif_report_csv,
    write_to_disk,
)


class _FakePrediction:
    def __init__(self):
        self.model_name = "random_forest"
        self.probability = 0.42
        self.risk_band = "Moderate"
        self.predicted_class = 0
        self.threshold = 0.5
        self.timestamp_utc = "2026-01-01T00:00:00+00:00"


class _FakeExplanation:
    base_value = 0.5

    def top_contributors(self, n=5):
        return [
            {"feature": "PAY_0", "shap_value": -0.05, "input_value": 0,
             "direction": "decreases risk"},
            {"feature": "LIMIT_BAL", "shap_value": 0.02, "input_value": 50000,
             "direction": "increases risk"},
        ][:n]


class _FakeChange:
    def __init__(self, feature, original, new):
        self.feature = feature
        self.original = original
        self.new = new


class _FakeWhatIf:
    def __init__(self):
        self.model_name = "random_forest"
        self.original_probability = 0.37
        self.modified_probability = 0.49
        self.original_band = "Moderate"
        self.modified_band = "Moderate"
        self.changes = [_FakeChange("PAY_0", 0, 3)]
        self.difference_pp = 12.0
        self.direction = "increased"


def test_prediction_report_json_contains_expected_fields():
    payload = prediction_report_json(_FakePrediction(), _FakeExplanation())
    data = json.loads(payload.decode("utf-8"))
    assert data["report_type"] == "individual_prediction"
    assert data["prediction"]["model_name"] == "random_forest"
    assert data["explanation"]["top_contributors"][0]["feature"] == "PAY_0"
    assert "disclaimer" in data


def test_prediction_report_json_without_explanation():
    payload = prediction_report_json(_FakePrediction())
    data = json.loads(payload.decode("utf-8"))
    assert "explanation" not in data


def test_prediction_report_csv_is_parseable():
    import io
    payload = prediction_report_csv(_FakePrediction())
    df = pd.read_csv(io.BytesIO(payload))
    assert df.shape[0] == 1
    assert "probability" in df.columns
    assert df["probability"].iloc[0] == pytest.approx(0.42)


def test_whatif_report_csv_has_summary_row():
    import io
    payload = whatif_report_csv(_FakeWhatIf())
    df = pd.read_csv(io.BytesIO(payload))
    assert any(df["feature"] == "__SUMMARY__")
    assert df["feature"].iloc[0] == "PAY_0"


def test_comparison_report_csv_preserves_index():
    import io
    df = pd.DataFrame(
        {"accuracy": [0.77, 0.79], "roc_auc": [0.75, 0.77]},
        index=pd.Index(["lr", "rf"], name="model"),
    )
    payload = comparison_report_csv(df)
    out = pd.read_csv(io.BytesIO(payload))
    assert "model" in out.columns
    assert set(out["model"]) == {"lr", "rf"}


def test_fairness_report_csv_metrics_only():
    import io
    metrics = pd.DataFrame({
        "audit": ["SEX", "SEX"],
        "group": [1, 2],
        "n": [100, 100],
        "tpr": [0.5, 0.7],
    })
    payload = fairness_report_csv(metrics)
    df = pd.read_csv(io.BytesIO(payload))
    assert "generated_utc" in df.columns
    assert len(df) == 2


def test_fairness_report_csv_with_differences_includes_separator():
    metrics = pd.DataFrame({"group": ["A"], "tpr": [0.5]})
    diffs = pd.DataFrame({"group": ["A"], "tpr_minus_ref": [0.0]})
    payload = fairness_report_csv(metrics, diffs)
    text = payload.decode("utf-8")
    assert "--- differences from reference ---" in text


def test_fairness_report_json_round_trip():
    metrics = pd.DataFrame({"group": ["A"], "tpr": [0.5]})
    payload = fairness_report_json(metrics, warnings=["small group"])
    data = json.loads(payload.decode("utf-8"))
    assert data["report_type"] == "fairness_audit"
    assert data["warnings"] == ["small group"]
    assert "limitations" in data


def test_model_metadata_json_includes_seed():
    payload = model_metadata_json({"roc_auc": 0.774})
    data = json.loads(payload.decode("utf-8"))
    assert data["seed"] == 42
    assert data["report_type"] == "model_metadata"
    assert data["metrics"]["roc_auc"] == 0.774


def test_write_to_disk(tmp_path, monkeypatch):
    # Redirect PROJECT_ROOT for this test
    monkeypatch.setattr("src.reporting.report_generator.PROJECT_ROOT", tmp_path)
    payload = model_metadata_json({"roc_auc": 0.5})
    path = write_to_disk(payload, "reports/test.json")
    assert path.exists()
    assert path.read_bytes() == payload