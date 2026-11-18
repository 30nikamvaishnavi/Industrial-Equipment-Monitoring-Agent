"""Anomaly detection using Isolation Forest and Rolling Z-Score detectors."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from data.generate_data import MACHINES, SENSORS

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
DEFAULT_RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", str(BASE_DIR / "results")))


@dataclass(frozen=True)
class AnomalyEvalMetrics:
    """Binary classification metrics against ground-truth anomaly labels."""

    precision: float
    recall: float
    f1: float


def compute_anomaly_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
) -> AnomalyEvalMetrics:
    """Compute precision, recall, and F1 score for anomaly detection.

    Args:
        y_true: Ground-truth binary labels (0 normal, 1 anomaly).
        y_pred: Predicted binary labels (0 normal, 1 anomaly).

    Returns:
        AnomalyEvalMetrics instance.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    return AnomalyEvalMetrics(
        precision=round(float(precision_score(yt, yp, zero_division=0)), 4),
        recall=round(float(recall_score(yt, yp, zero_division=0)), 4),
        f1=round(float(f1_score(yt, yp, zero_division=0)), 4),
    )


def _build_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build strictly causal multi-sensor features per machine for anomaly detection."""
    ordered = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    feat = pd.DataFrame(index=ordered.index)

    for sensor in SENSORS:
        grp = ordered.groupby("machine_id")[sensor]
        feat[sensor] = ordered[sensor].astype(float)
        feat[f"{sensor}_diff1"] = grp.diff().fillna(0.0).astype(float)
        roll_mean = grp.transform(lambda s: s.rolling(window=24, min_periods=1).mean())
        roll_std = grp.transform(lambda s: s.rolling(window=6, min_periods=2).std().fillna(1.0))
        feat[f"{sensor}_dev"] = ordered[sensor].astype(float) - roll_mean.astype(float)
        feat[f"{sensor}_roll_std"] = roll_std.astype(float)

    return feat


class IsolationForestDetector:
    """Multi-sensor Isolation Forest anomaly detector with StandardScaler."""

    def __init__(self, contamination: float = 0.025, random_state: int = 42) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
        )
        self._is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> IsolationForestDetector:
        """Fit scaler and Isolation Forest strictly on training observations."""
        features = _build_anomaly_features(train_df)
        x_scaled = self.scaler.fit_transform(features.to_numpy(dtype=float))
        self.model.fit(x_scaled)
        self._is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Predict binary anomaly flags (1=anomaly, 0=normal) and anomaly scores."""
        if not self._is_fitted:
            raise RuntimeError("IsolationForestDetector must be fitted before predict().")
        features = _build_anomaly_features(df)
        x_scaled = self.scaler.transform(features.to_numpy(dtype=float))
        raw_preds = self.model.predict(x_scaled)  # -1 anomaly, 1 normal
        labels = np.where(raw_preds == -1, 1, 0).astype(int)
        # Higher positive score = more anomalous
        scores = -self.model.decision_function(x_scaled)
        return labels, scores


class RollingZScoreDetector:
    """Causal rolling z-score + flat-line fault detector across sensor channels."""

    def __init__(
        self,
        window: int = 24,
        min_periods: int = 6,
        z_threshold: float = 3.2,
        flatline_std_eps: float = 1e-4,
    ) -> None:
        self.window = window
        self.min_periods = min_periods
        self.z_threshold = z_threshold
        self.flatline_std_eps = flatline_std_eps

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
        """Detect anomalies using causal rolling z-score and flat-line variance checks.

        Args:
            df: Sensor DataFrame containing machine_id, timestamp, and sensor columns.

        Returns:
            Tuple of (binary_labels, max_z_scores, flagged_sensors_per_row).
        """
        ordered = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
        n_rows = len(ordered)
        max_z = np.zeros(n_rows, dtype=float)
        flatline_flag = np.zeros(n_rows, dtype=bool)
        row_sensors: list[list[str]] = [[] for _ in range(n_rows)]

        for sensor in SENSORS:
            grp = ordered.groupby("machine_id")[sensor]
            # Use shifted causal window so current observation does not inflate its own std
            shifted = grp.shift(1)
            roll_mean = shifted.groupby(ordered["machine_id"]).transform(
                lambda s: s.rolling(window=self.window, min_periods=self.min_periods).mean()
            )
            roll_std = shifted.groupby(ordered["machine_id"]).transform(
                lambda s: s.rolling(window=self.window, min_periods=self.min_periods).std()
            )
            short_std = grp.transform(
                lambda s: s.rolling(window=4, min_periods=4).std()
            )

            safe_std = np.where(roll_std.isna() | (roll_std < 1e-6), 1.0, roll_std)
            z_vals = np.where(
                roll_mean.isna(),
                0.0,
                np.abs(ordered[sensor].to_numpy(dtype=float) - roll_mean.to_numpy(dtype=float))
                / safe_std,
            )
            is_flat = (~short_std.isna()) & (
                short_std.to_numpy(dtype=float) < self.flatline_std_eps
            )

            for idx in range(n_rows):
                if z_vals[idx] > max_z[idx]:
                    max_z[idx] = float(z_vals[idx])
                if z_vals[idx] >= self.z_threshold or is_flat[idx]:
                    row_sensors[idx].append(sensor)
                if is_flat[idx]:
                    flatline_flag[idx] = True

        labels = ((max_z >= self.z_threshold) | flatline_flag).astype(int)
        return labels, np.round(max_z, 4), row_sensors


def load_all_sensor_data(data_path: Path | None = None) -> pd.DataFrame:
    """Load full sensor_readings.csv ordered by machine_id and timestamp."""
    csv_path = data_path or (DEFAULT_DATA_DIR / "sensor_readings.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run 'python -m data.generate_data' first."
        )
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)


def evaluate_detectors(
    data_path: Path | None = None,
    results_dir: Path | None = None,
    train_ratio: float = 0.7,
) -> dict[str, Any]:
    """Evaluate Isolation Forest and Rolling Z-Score on a chronological test split.

    Args:
        data_path: Optional override path to sensor_readings.csv.
        results_dir: Optional directory to save anomaly_metrics.json.
        train_ratio: Fraction of earliest timestamps used to fit Isolation Forest.

    Returns:
        Dictionary of evaluation metrics written to results/anomaly_metrics.json.
    """
    df = load_all_sensor_data(data_path=data_path)
    unique_ts = sorted(df["timestamp"].unique())
    split_idx = int(len(unique_ts) * train_ratio)
    split_ts = unique_ts[split_idx]

    train_df = df[df["timestamp"] < split_ts].copy().reset_index(drop=True)
    test_df = df[df["timestamp"] >= split_ts].copy().reset_index(drop=True)

    iso_detector = IsolationForestDetector(contamination=0.025, random_state=42)
    iso_detector.fit(train_df)
    iso_preds, _ = iso_detector.predict(test_df)

    z_detector = RollingZScoreDetector(window=24, min_periods=6, z_threshold=3.2)
    # Compute rolling z-score over full ordered history per machine, then slice test window
    full_z_preds, _, _ = z_detector.predict(df)
    test_mask = (df["timestamp"] >= split_ts).to_numpy()
    z_preds = full_z_preds[test_mask]

    y_true = test_df["is_anomaly"].to_numpy(dtype=int)

    iso_metrics = compute_anomaly_metrics(y_true, iso_preds)
    z_metrics = compute_anomaly_metrics(y_true, z_preds)

    output = {
        "split_timestamp": pd.Timestamp(split_ts).strftime("%Y-%m-%dT%H:%M:%S"),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "test_positive_count": int(y_true.sum()),
        "models": {
            "isolation_forest": asdict(iso_metrics),
            "rolling_zscore": asdict(z_metrics),
        },
    }

    out_dir = results_dir or DEFAULT_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "anomaly_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def detect_machine_anomalies(
    machine_id: str,
    lookback_hours: int = 168,
    method: str = "both",
    data_path: Path | None = None,
) -> dict[str, Any]:
    """Scan recent readings for a specific machine and return detected anomalies.

    Args:
        machine_id: Machine identifier (M1..M5).
        lookback_hours: Number of trailing hours to inspect (default 168).
        method: Detection method ('isolation_forest', 'rolling_zscore', or 'both').
        data_path: Optional override path to sensor_readings.csv.

    Returns:
        Dictionary summarizing detected anomalies and affected sensors.
    """
    if machine_id not in MACHINES:
        raise ValueError(f"Unknown machine_id '{machine_id}'. Must be one of {MACHINES}.")
    if method not in {"isolation_forest", "rolling_zscore", "both"}:
        raise ValueError("method must be 'isolation_forest', 'rolling_zscore', or 'both'.")
    if lookback_hours <= 0:
        raise ValueError("lookback_hours must be positive.")

    full_df = load_all_sensor_data(data_path=data_path)
    machine_df = full_df[full_df["machine_id"] == machine_id].copy().reset_index(drop=True)

    split_idx = int(len(machine_df) * 0.7)
    train_df = machine_df.iloc[:split_idx].copy().reset_index(drop=True)

    iso_detector = IsolationForestDetector(contamination=0.025, random_state=42)
    iso_detector.fit(train_df)
    iso_preds, iso_scores = iso_detector.predict(machine_df)

    z_detector = RollingZScoreDetector(window=24, min_periods=6, z_threshold=3.2)
    z_preds, z_scores, flagged_sensors = z_detector.predict(machine_df)

    tail_start = max(0, len(machine_df) - lookback_hours)
    anomalies: list[dict[str, Any]] = []

    for idx in range(tail_start, len(machine_df)):
        iso_hit = bool(iso_preds[idx] == 1)
        z_hit = bool(z_preds[idx] == 1)
        if method == "isolation_forest":
            flagged = iso_hit
        elif method == "rolling_zscore":
            flagged = z_hit
        else:
            flagged = iso_hit or z_hit

        if flagged:
            row = machine_df.iloc[idx]
            anomalies.append(
                {
                    "timestamp": pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%dT%H:%M:%S"),
                    "temperature": round(float(row["temperature"]), 3),
                    "vibration": round(float(row["vibration"]), 3),
                    "pressure": round(float(row["pressure"]), 3),
                    "rpm": round(float(row["rpm"]), 2),
                    "isolation_forest_flag": iso_hit,
                    "rolling_zscore_flag": z_hit,
                    "isolation_score": round(float(iso_scores[idx]), 4),
                    "max_z_score": round(float(z_scores[idx]), 4),
                    "affected_sensors": (
                        flagged_sensors[idx] if flagged_sensors[idx] else ["multivariate"]
                    ),
                }
            )

    return {
        "machine_id": machine_id,
        "lookback_hours": lookback_hours,
        "method": method,
        "rows_scanned": len(machine_df) - tail_start,
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
    }


def main() -> None:
    """CLI entry point to evaluate anomaly detectors and save metrics."""
    results = evaluate_detectors()
    print(json.dumps(results["models"], indent=2))


if __name__ == "__main__":
    main()
