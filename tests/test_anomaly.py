"""Tests for Isolation Forest, Rolling Z-Score, and classification metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from anomaly.detector import (
    IsolationForestDetector,
    RollingZScoreDetector,
    compute_anomaly_metrics,
    evaluate_detectors,
)
from data.generate_data import generate_dataset


def test_compute_anomaly_metrics_exact_values() -> None:
    """Verify precision, recall, and F1 computation on known binary arrays."""
    y_true = [1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 0]
    # TP=2, FP=1, FN=1 -> precision=2/3, recall=2/3, f1=2/3
    metrics = compute_anomaly_metrics(y_true, y_pred)
    assert metrics.precision == 0.6667
    assert metrics.recall == 0.6667
    assert metrics.f1 == 0.6667


def test_rolling_zscore_detects_obvious_spike() -> None:
    """Rolling Z-Score should flag an isolated 10-sigma temperature spike."""
    rng = np.random.default_rng(7)
    n = 60
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
            "machine_id": ["M1"] * n,
            "temperature": 70.0 + rng.normal(0.0, 0.5, size=n),
            "vibration": 2.0 + rng.normal(0.0, 0.05, size=n),
            "pressure": 100.0 + rng.normal(0.0, 0.4, size=n),
            "rpm": 1500.0 + rng.normal(0.0, 5.0, size=n),
        }
    )
    df.loc[45, "temperature"] = 120.0  # Extreme spike

    detector = RollingZScoreDetector(window=24, min_periods=6, z_threshold=3.2)
    labels, z_scores, flagged_sensors = detector.predict(df)

    assert labels[45] == 1
    assert z_scores[45] > 10.0
    assert "temperature" in flagged_sensors[45]


def test_isolation_forest_and_evaluation_pipeline(tmp_path: Path) -> None:
    """Verify Isolation Forest fit/predict and end-to-end detector evaluation."""
    generate_dataset(data_dir=tmp_path, docs_dir=tmp_path / "docs", seed=42)
    results_dir = tmp_path / "results"
    eval_output = evaluate_detectors(
        data_path=tmp_path / "sensor_readings.csv",
        results_dir=results_dir,
    )

    assert (results_dir / "anomaly_metrics.json").exists()
    for model_key in ("isolation_forest", "rolling_zscore"):
        model_metrics = eval_output["models"][model_key]
        assert 0.0 <= model_metrics["precision"] <= 1.0
        assert 0.0 <= model_metrics["recall"] <= 1.0
        assert 0.0 <= model_metrics["f1"] <= 1.0
        assert model_metrics["precision"] > 0.0
        assert model_metrics["recall"] > 0.0

    df = pd.read_csv(tmp_path / "sensor_readings.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    iso = IsolationForestDetector(contamination=0.025, random_state=42).fit(df.iloc[:1500])
    preds, scores = iso.predict(df.iloc[1500:1600])
    assert len(preds) == 100
    assert len(scores) == 100
    assert set(np.unique(preds)).issubset({0, 1})
