"""24-hour sensor forecasting comparing Naive, ARIMA, and Gradient Boosting models."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from data.generate_data import MACHINES, SENSORS

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
DEFAULT_RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(BASE_DIR / "results")))

LAGS = (1, 2, 3, 6, 12, 24, 48)
ROLLING_WINDOWS = (6, 12, 24)
FEATURE_COLUMNS = [
    *[f"lag_{lag}" for lag in LAGS],
    *[f"roll_mean_{w}" for w in ROLLING_WINDOWS],
    *[f"roll_std_{w}" for w in ROLLING_WINDOWS],
    "hour_sin",
    "hour_cos",
]


@dataclass(frozen=True)
class ModelMetrics:
    """Evaluation metrics for a forecasting model."""

    mae: float
    rmse: float


def load_sensor_data(
    machine_id: str,
    sensor: str,
    data_path: Path | None = None,
) -> pd.DataFrame:
    """Load and chronologically sort sensor readings for a machine.

    Args:
        machine_id: Machine identifier (M1..M5).
        sensor: Sensor column name (temperature, vibration, pressure, rpm).
        data_path: Optional override path to sensor_readings.csv.

    Returns:
        DataFrame with columns ['timestamp', sensor] sorted ascending by timestamp.

    Raises:
        ValueError: If machine_id or sensor is invalid or no rows exist.
        FileNotFoundError: If sensor_readings.csv does not exist.
    """
    if machine_id not in MACHINES:
        raise ValueError(f"Unknown machine_id '{machine_id}'. Must be one of {MACHINES}.")
    if sensor not in SENSORS:
        raise ValueError(f"Unknown sensor '{sensor}'. Must be one of {SENSORS}.")

    csv_path = data_path or (DEFAULT_DATA_DIR / "sensor_readings.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run 'python -m data.generate_data' first."
        )

    df = pd.read_csv(csv_path)
    machine_df = df[df["machine_id"] == machine_id].copy()
    if machine_df.empty:
        raise ValueError(f"No sensor readings found for machine '{machine_id}'.")

    machine_df["timestamp"] = pd.to_datetime(machine_df["timestamp"])
    machine_df = machine_df.sort_values("timestamp").reset_index(drop=True)
    return machine_df[["timestamp", sensor]]


def time_based_split(
    df: pd.DataFrame,
    test_hours: int = 168,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a strict chronological train/test split with zero leakage.

    Args:
        df: Chronologically ordered DataFrame with a 'timestamp' column.
        test_hours: Number of trailing hours reserved for testing.

    Returns:
        Tuple of (train_df, test_df) where train_df strictly precedes test_df.

    Raises:
        ValueError: If dataset is too short for the requested split.
    """
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    if test_hours <= 0 or test_hours >= len(ordered):
        raise ValueError(
            f"test_hours={test_hours} must be between 1 and {len(ordered) - 1}."
        )
    split_idx = len(ordered) - test_hours
    train_df = ordered.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = ordered.iloc[split_idx:].copy().reset_index(drop=True)

    if train_df["timestamp"].max() >= test_df["timestamp"].min():
        raise ValueError("Time-based split violation: train overlaps test.")

    return train_df, test_df


def compute_forecast_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> ModelMetrics:
    """Compute MAE and RMSE between ground truth and forecasted values.

    Args:
        y_true: Ground truth observations.
        y_pred: Forecasted values.

    Returns:
        ModelMetrics with rounded MAE and RMSE.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    return ModelMetrics(mae=round(mae, 4), rmse=round(rmse, 4))


def _build_lag_feature_frame(df: pd.DataFrame, sensor: str) -> pd.DataFrame:
    """Build strictly causal lag, rolling, and cyclical time features."""
    feat = pd.DataFrame(index=df.index)
    series = df[sensor].astype(float)

    for lag in LAGS:
        feat[f"lag_{lag}"] = series.shift(lag)

    shifted = series.shift(1)
    for window in ROLLING_WINDOWS:
        feat[f"roll_mean_{window}"] = shifted.rolling(window=window, min_periods=window).mean()
        roll_std = shifted.rolling(window=window, min_periods=window).std().fillna(0.0)
        feat[f"roll_std_{window}"] = roll_std

    hours = pd.to_datetime(df["timestamp"]).dt.hour.astype(float)
    feat["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    feat["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    feat["target"] = series
    return feat.dropna().reset_index(drop=True)


def _step_features_from_history(
    history: list[float],
    target_ts: pd.Timestamp,
) -> np.ndarray:
    """Construct a single-step feature vector from causal history only."""
    row: list[float] = []
    for lag in LAGS:
        row.append(float(history[-lag]))
    for window in ROLLING_WINDOWS:
        win_slice = np.asarray(history[-window:], dtype=float)
        row.append(float(np.mean(win_slice)))
    for window in ROLLING_WINDOWS:
        win_slice = np.asarray(history[-window:], dtype=float)
        row.append(float(np.std(win_slice, ddof=1)) if len(win_slice) > 1 else 0.0)
    hour = float(target_ts.hour)
    row.append(float(np.sin(2.0 * np.pi * hour / 24.0)))
    row.append(float(np.cos(2.0 * np.pi * hour / 24.0)))
    return np.asarray(row, dtype=float).reshape(1, -1)


def _forecast_naive(last_value: float, horizon: int) -> np.ndarray:
    """Generate a naive last-value constant forecast."""
    return np.full(shape=horizon, fill_value=float(last_value), dtype=float)


def _fit_arima_model(train_series: np.ndarray) -> Any:
    """Fit an ARIMA model on training data with safe fallback order."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in ((2, 1, 2), (1, 1, 1), (1, 1, 0)):
            try:
                model = ARIMA(train_series, order=order)
                return model.fit()
            except Exception:
                continue
        model = ARIMA(train_series, order=(0, 1, 0))
        return model.fit()


def _forecast_arima_with_context(
    fitted_arima: Any,
    context_series: np.ndarray,
    horizon: int,
    base_train_len: int,
) -> np.ndarray:
    """Forecast horizon steps using fitted ARIMA parameters and causal context."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(context_series) == base_train_len:
            preds = fitted_arima.forecast(steps=horizon)
        else:
            extra = context_series[base_train_len:]
            updated = fitted_arima.append(extra, refit=False)
            preds = updated.forecast(steps=horizon)
    return np.asarray(preds, dtype=float)


def _fit_gbr_model(train_df: pd.DataFrame, sensor: str) -> GradientBoostingRegressor:
    """Fit GradientBoostingRegressor strictly on training split features."""
    feat_df = _build_lag_feature_frame(train_df, sensor)
    x_train = feat_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = feat_df["target"].to_numpy(dtype=float)

    model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def _forecast_gbr_recursive(
    model: GradientBoostingRegressor,
    context_values: list[float],
    start_ts: pd.Timestamp,
    horizon: int,
) -> np.ndarray:
    """Recursively predict horizon future hours using only causal history."""
    history = list(context_values)
    preds: list[float] = []
    for step in range(horizon):
        ts = start_ts + pd.Timedelta(hours=step)
        x_step = _step_features_from_history(history, ts)
        y_hat = float(model.predict(x_step)[0])
        preds.append(y_hat)
        history.append(y_hat)
    return np.asarray(preds, dtype=float)


def evaluate_and_forecast(
    machine_id: str,
    sensor: str,
    horizon_hours: int = 24,
    test_hours: int = 168,
    data_path: Path | None = None,
) -> dict[str, Any]:
    """Compare Naive, ARIMA, and Gradient Boosting on a time split and forecast ahead.

    Args:
        machine_id: Machine identifier (M1..M5).
        sensor: Sensor name (temperature, vibration, pressure, rpm).
        horizon_hours: Number of future hours to forecast (default 24).
        test_hours: Trailing test window hours for walk-forward evaluation (default 168).
        data_path: Optional override path to sensor_readings.csv.

    Returns:
        Dictionary with model metrics, best model name, and 24-hour forecast points.
    """
    if horizon_hours <= 0 or horizon_hours > 168:
        raise ValueError("horizon_hours must be between 1 and 168.")

    df = load_sensor_data(machine_id=machine_id, sensor=sensor, data_path=data_path)
    train_df, test_df = time_based_split(df, test_hours=test_hours)

    train_values = train_df[sensor].to_numpy(dtype=float)
    fitted_arima = _fit_arima_model(train_values)
    gbr_model = _fit_gbr_model(train_df, sensor)

    # Walk-forward evaluation over non-overlapping horizon_hours blocks in test_df
    y_true_all: list[float] = []
    preds_naive_all: list[float] = []
    preds_arima_all: list[float] = []
    preds_gbr_all: list[float] = []

    full_values = df[sensor].to_numpy(dtype=float)
    full_timestamps = pd.to_datetime(df["timestamp"]).tolist()
    split_idx = len(train_df)

    for offset in range(0, len(test_df), horizon_hours):
        origin_idx = split_idx + offset
        block_end = min(origin_idx + horizon_hours, len(df))
        block_len = block_end - origin_idx
        if block_len < horizon_hours:
            break

        causal_context = full_values[:origin_idx]
        origin_ts = pd.Timestamp(full_timestamps[origin_idx])

        y_true_block = full_values[origin_idx:block_end]
        naive_block = _forecast_naive(float(causal_context[-1]), block_len)
        arima_block = _forecast_arima_with_context(
            fitted_arima=fitted_arima,
            context_series=causal_context,
            horizon=block_len,
            base_train_len=split_idx,
        )
        gbr_block = _forecast_gbr_recursive(
            model=gbr_model,
            context_values=causal_context.tolist(),
            start_ts=origin_ts,
            horizon=block_len,
        )

        y_true_all.extend(y_true_block.tolist())
        preds_naive_all.extend(naive_block.tolist())
        preds_arima_all.extend(arima_block.tolist())
        preds_gbr_all.extend(gbr_block.tolist())

    metrics = {
        "naive_last_value": asdict(compute_forecast_metrics(y_true_all, preds_naive_all)),
        "arima": asdict(compute_forecast_metrics(y_true_all, preds_arima_all)),
        "gradient_boosting": asdict(compute_forecast_metrics(y_true_all, preds_gbr_all)),
    }

    best_model = min(metrics, key=lambda name: metrics[name]["rmse"])

    # Fit on full observed history to forecast the next horizon_hours beyond dataset end
    last_ts = pd.Timestamp(df["timestamp"].iloc[-1])
    next_start_ts = last_ts + pd.Timedelta(hours=1)

    if best_model == "naive_last_value":
        future_preds = _forecast_naive(float(full_values[-1]), horizon_hours)
    elif best_model == "arima":
        full_arima = _fit_arima_model(full_values)
        future_preds = np.asarray(full_arima.forecast(steps=horizon_hours), dtype=float)
    else:
        full_gbr = _fit_gbr_model(df, sensor)
        future_preds = _forecast_gbr_recursive(
            model=full_gbr,
            context_values=full_values.tolist(),
            start_ts=next_start_ts,
            horizon=horizon_hours,
        )

    forecast_points = [
        {
            "timestamp": (next_start_ts + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S"),
            "predicted_value": round(float(val), 3),
        }
        for i, val in enumerate(future_preds)
    ]

    return {
        "machine_id": machine_id,
        "sensor": sensor,
        "horizon_hours": horizon_hours,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "last_observed_timestamp": last_ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "last_observed_value": round(float(full_values[-1]), 3),
        "best_model": best_model,
        "model_metrics": metrics,
        "forecast": forecast_points,
    }


def run_benchmark(
    data_path: Path | None = None,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate all 3 forecasting models across all machines and sensors.

    Args:
        data_path: Optional override path to sensor_readings.csv.
        results_dir: Output directory for forecast_metrics.json.

    Returns:
        Summary dictionary saved to results/forecast_metrics.json.
    """
    out_dir = results_dir or DEFAULT_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    per_series: list[dict[str, Any]] = []
    model_names = ("naive_last_value", "arima", "gradient_boosting")
    agg: dict[str, dict[str, list[float]]] = {
        m: {"mae": [], "rmse": []} for m in model_names
    }

    for machine_id in MACHINES:
        for sensor in SENSORS:
            res = evaluate_and_forecast(
                machine_id=machine_id,
                sensor=sensor,
                horizon_hours=24,
                test_hours=168,
                data_path=data_path,
            )
            per_series.append(
                {
                    "machine_id": machine_id,
                    "sensor": sensor,
                    "best_model": res["best_model"],
                    "model_metrics": res["model_metrics"],
                }
            )
            for model_name in model_names:
                agg[model_name]["mae"].append(res["model_metrics"][model_name]["mae"])
                agg[model_name]["rmse"].append(res["model_metrics"][model_name]["rmse"])

    aggregate_metrics = {
        model_name: {
            "mean_mae": round(float(np.mean(vals["mae"])), 4),
            "mean_rmse": round(float(np.mean(vals["rmse"])), 4),
        }
        for model_name, vals in agg.items()
    }

    output = {
        "horizon_hours": 24,
        "test_hours": 168,
        "aggregate_metrics": aggregate_metrics,
        "per_series": per_series,
    }

    out_path = out_dir / "forecast_metrics.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def main() -> None:
    """CLI entry point to run forecasting benchmark and save results."""
    results = run_benchmark()
    print(json.dumps(results["aggregate_metrics"], indent=2))


if __name__ == "__main__":
    main()
