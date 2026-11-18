"""Tests verifying deterministic synthetic data generation and database schema."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from data.generate_data import MACHINES, TOTAL_HOURS, generate_dataset


def test_data_generation_is_deterministic(tmp_path: Path) -> None:
    """Running generate_dataset twice with seed 42 yields identical outputs."""
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    sensor_a, logs_a = generate_dataset(data_dir=dir_a, docs_dir=dir_a / "docs", seed=42)
    sensor_b, logs_b = generate_dataset(data_dir=dir_b, docs_dir=dir_b / "docs", seed=42)

    assert sensor_a.equals(sensor_b)
    assert logs_a.equals(logs_b)
    assert (dir_a / "sensor_readings.csv").read_text(encoding="utf-8") == (
        dir_b / "sensor_readings.csv"
    ).read_text(encoding="utf-8")


def test_dataset_shape_anomalies_db_and_docs(tmp_path: Path) -> None:
    """Verify row count, ~2% anomaly rate, ~40 SQLite logs, and 4 manual docs."""
    docs_dir = tmp_path / "docs"
    sensor_df, logs_df = generate_dataset(data_dir=tmp_path, docs_dir=docs_dir, seed=42)

    assert len(sensor_df) == len(MACHINES) * TOTAL_HOURS
    expected_cols = [
        "timestamp",
        "machine_id",
        "temperature",
        "vibration",
        "pressure",
        "rpm",
        "is_anomaly",
    ]
    assert list(sensor_df.columns) == expected_cols

    anomaly_ratio = float(sensor_df["is_anomaly"].mean())
    assert 0.015 <= anomaly_ratio <= 0.025

    assert len(logs_df) == 40
    with sqlite3.connect(tmp_path / "maintenance.db") as conn:
        count = conn.execute("SELECT COUNT(*) FROM maintenance_logs").fetchone()[0]
    assert count == 40

    md_files = sorted(p.name for p in docs_dir.glob("*.md"))
    assert md_files == [
        "bearing_vibration.md",
        "overheating.md",
        "pressure_loss.md",
        "sensor_faults.md",
    ]
