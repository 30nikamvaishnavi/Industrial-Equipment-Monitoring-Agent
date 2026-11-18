"""Tests for scripts/update_readme.py."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.update_readme import (
    AGENT_END,
    AGENT_START,
    ANOMALY_END,
    ANOMALY_START,
    FORECAST_END,
    FORECAST_START,
    update_readme,
)

SAMPLE_README = f"""# Sample Project

## Results

### Forecasting Benchmark (`results/forecast_metrics.json`)

{FORECAST_START}
| Model | Mean MAE | Mean RMSE |
| --- | --- | --- |
| Naive Last-Value | not run yet | not run yet |
{FORECAST_END}

### Anomaly Detection Benchmark (`results/anomaly_metrics.json`)

{ANOMALY_START}
| Detector | Precision | Recall | F1 Score |
| --- | --- | --- | --- |
| Isolation Forest | not run yet | not run yet | not run yet |
{ANOMALY_END}

### Agent Report Evaluation (`results/agent_eval.json`)

{AGENT_START}
| Metric | Value |
| --- | --- |
| Total Test Cases | requires GOOGLE_API_KEY and GEMINI_MODEL |
{AGENT_END}
"""


def test_update_readme_populates_tables_from_json(tmp_path: Path) -> None:
    """Tables should be populated directly from forecast, anomaly, and eval JSON files."""
    readme_path = tmp_path / "README.md"
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(SAMPLE_README, encoding="utf-8")

    (results_dir / "forecast_metrics.json").write_text(
        json.dumps(
            {
                "aggregate_metrics": {
                    "naive_last_value": {"mean_mae": 1.25, "mean_rmse": 1.5},
                    "arima": {"mean_mae": 0.91234, "mean_rmse": 1.12345},
                    "gradient_boosting": {"mean_mae": 0.75, "mean_rmse": 0.95},
                }
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "anomaly_metrics.json").write_text(
        json.dumps(
            {
                "models": {
                    "isolation_forest": {"precision": 0.8125, "recall": 0.65, "f1": 0.7222},
                    "rolling_zscore": {"precision": 0.75, "recall": 0.6, "f1": 0.6667},
                }
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "agent_eval.json").write_text(
        json.dumps({"total_cases": 10, "passed_cases": 9, "pass_rate_pct": 90.0}),
        encoding="utf-8",
    )

    updated = update_readme(
        readme_path=readme_path,
        results_dir=results_dir,
        utc_date="2026-11-18",
    )

    assert "Results generated on 2026-11-18 from the files in results/" in updated
    assert "| Naive Last-Value | 1.2500 | 1.5000 |" in updated
    assert "| ARIMA (`statsmodels`) | 0.9123 | 1.1235 |" in updated
    assert "| Gradient Boosting (`scikit-learn`) | 0.7500 | 0.9500 |" in updated
    assert "| Isolation Forest | 0.8125 | 0.6500 | 0.7222 |" in updated
    assert "| Rolling Z-Score | 0.7500 | 0.6000 | 0.6667 |" in updated
    assert "| Total Test Cases | 10 |" in updated
    assert "| Passed Cases | 9 |" in updated
    assert "| Pass Rate (%) | 90.00% |" in updated


def test_update_readme_missing_files_placeholders(tmp_path: Path) -> None:
    """Missing results JSON files should produce explicit placeholder messages."""
    readme_path = tmp_path / "README.md"
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(SAMPLE_README, encoding="utf-8")

    updated = update_readme(
        readme_path=readme_path,
        results_dir=results_dir,
        utc_date="2026-11-18",
    )

    assert "| Naive Last-Value | not run yet | not run yet |" in updated
    assert "| Isolation Forest | not run yet | not run yet | not run yet |" in updated
    assert "| Total Test Cases | requires GOOGLE_API_KEY and GEMINI_MODEL |" in updated
    assert "| Pass Rate (%) | requires GOOGLE_API_KEY and GEMINI_MODEL |" in updated


def test_update_readme_is_idempotent(tmp_path: Path) -> None:
    """Running update_readme twice with the same files must yield byte-identical output."""
    readme_path = tmp_path / "README.md"
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(SAMPLE_README, encoding="utf-8")

    (results_dir / "forecast_metrics.json").write_text(
        json.dumps(
            {
                "aggregate_metrics": {
                    "naive_last_value": {"mean_mae": 1.0, "mean_rmse": 1.2},
                    "arima": {"mean_mae": 0.9, "mean_rmse": 1.1},
                    "gradient_boosting": {"mean_mae": 0.8, "mean_rmse": 1.0},
                }
            }
        ),
        encoding="utf-8",
    )

    first = update_readme(
        readme_path=readme_path,
        results_dir=results_dir,
        utc_date="2026-11-18",
    )
    second = update_readme(
        readme_path=readme_path,
        results_dir=results_dir,
        utc_date="2026-11-18",
    )

    assert first == second
    assert first.count("Results generated on 2026-11-18 from the files in results/") == 1
