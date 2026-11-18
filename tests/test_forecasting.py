"""Tests for time-based splitting, metric calculation, and 24-hour forecasting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.generate_data import generate_dataset
from forecasting.forecaster import (
    compute_forecast_metrics,
    evaluate_and_forecast,
    time_based_split,
)


def test_time_based_split_has_no_leakage() -> None:
    """Train timestamps must strictly precede all test timestamps."""
    timestamps = pd.date_range("2026-01-01", periods=300, freq="h")
    df = pd.DataFrame({"timestamp": timestamps, "temperature": np.arange(300, dtype=float)})

    # Deliberately shuffle input to verify function enforces chronological ordering
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_df, test_df = time_based_split(shuffled, test_hours=48)

    assert len(train_df) == 252
    assert len(test_df) == 48
    assert train_df["timestamp"].max() < test_df["timestamp"].min()
    assert train_df["temperature"].max() < test_df["temperature"].min()


def test_compute_forecast_metrics_exact_values() -> None:
    """Verify MAE and RMSE calculations against analytical ground truth."""
    y_true = [10.0, 20.0, 30.0, 40.0]
    y_pred = [12.0, 18.0, 33.0, 37.0]
    # Errors: [2, 2, 3, 3] -> MAE = 2.5, MSE = (4+4+9+9)/4 = 6.5 -> RMSE = sqrt(6.5) = 2.5495
    metrics = compute_forecast_metrics(y_true, y_pred)
    assert metrics.mae == 2.5
    assert metrics.rmse == round(float(np.sqrt(6.5)), 4)


def test_evaluate_and_forecast_returns_all_models(tmp_path: Path) -> None:
    """Run all 3 forecasting models on synthetic data and check output schema."""
    generate_dataset(data_dir=tmp_path, docs_dir=tmp_path / "docs", seed=42)
    result = evaluate_and_forecast(
        machine_id="M1",
        sensor="temperature",
        horizon_hours=24,
        test_hours=72,
        data_path=tmp_path / "sensor_readings.csv",
    )

    assert result["machine_id"] == "M1"
    assert result["sensor"] == "temperature"
    assert result["horizon_hours"] == 24
    assert len(result["forecast"]) == 24
    assert set(result["model_metrics"].keys()) == {
        "naive_last_value",
        "arima",
        "gradient_boosting",
    }
    assert result["best_model"] in result["model_metrics"]
    for model_vals in result["model_metrics"].values():
        assert model_vals["mae"] >= 0.0
        assert model_vals["rmse"] >= model_vals["mae"]
