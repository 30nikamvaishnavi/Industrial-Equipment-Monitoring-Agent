"""Idempotently update README.md Results tables from results/*.json files."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_README_PATH = BASE_DIR / "README.md"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"

FORECAST_START = "<!-- RESULTS:FORECAST:START -->"
FORECAST_END = "<!-- RESULTS:FORECAST:END -->"
ANOMALY_START = "<!-- RESULTS:ANOMALY:START -->"
ANOMALY_END = "<!-- RESULTS:ANOMALY:END -->"
AGENT_START = "<!-- RESULTS:AGENT:START -->"
AGENT_END = "<!-- RESULTS:AGENT:END -->"

_GENERATED_LINE_RE = re.compile(r"^Results generated on .* from the files in results/$", re.M)


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file if it exists, returning None otherwise."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _format_num(value: Any) -> str:
    """Format a numeric metric cleanly or return 'not run yet' if missing."""
    if value is None:
        return "not run yet"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_forecast_table(forecast_data: dict[str, Any] | None) -> str:
    """Render the Markdown table for results/forecast_metrics.json."""
    rows = [
        ("Naive Last-Value", "naive_last_value"),
        ("ARIMA (`statsmodels`)", "arima"),
        ("Gradient Boosting (`scikit-learn`)", "gradient_boosting"),
    ]
    agg = (forecast_data or {}).get("aggregate_metrics", {})
    lines = [
        "| Model | Mean MAE | Mean RMSE |",
        "| --- | --- | --- |",
    ]
    for label, key in rows:
        if forecast_data is None or key not in agg:
            mae_str = "not run yet"
            rmse_str = "not run yet"
        else:
            mae_str = _format_num(agg[key].get("mean_mae"))
            rmse_str = _format_num(agg[key].get("mean_rmse"))
        lines.append(f"| {label} | {mae_str} | {rmse_str} |")
    return "\n".join(lines)


def render_anomaly_table(anomaly_data: dict[str, Any] | None) -> str:
    """Render the Markdown table for results/anomaly_metrics.json."""
    rows = [
        ("Isolation Forest", "isolation_forest"),
        ("Rolling Z-Score", "rolling_zscore"),
    ]
    models = (anomaly_data or {}).get("models", {})
    lines = [
        "| Detector | Precision | Recall | F1 Score |",
        "| --- | --- | --- | --- |",
    ]
    for label, key in rows:
        if anomaly_data is None or key not in models:
            p_str = "not run yet"
            r_str = "not run yet"
            f_str = "not run yet"
        else:
            p_str = _format_num(models[key].get("precision"))
            r_str = _format_num(models[key].get("recall"))
            f_str = _format_num(models[key].get("f1"))
        lines.append(f"| {label} | {p_str} | {r_str} | {f_str} |")
    return "\n".join(lines)


def render_agent_table(agent_data: dict[str, Any] | None) -> str:
    """Render the Markdown table for results/agent_eval.json."""
    missing_msg = "requires GOOGLE_API_KEY and GEMINI_MODEL"
    lines = [
        "| Metric | Value |",
        "| --- | --- |",
    ]
    if agent_data is None:
        lines.append(f"| Total Test Cases | {missing_msg} |")
        lines.append(f"| Passed Cases | {missing_msg} |")
        lines.append(f"| Pass Rate (%) | {missing_msg} |")
    else:
        total = agent_data.get("total_cases", missing_msg)
        passed = agent_data.get("passed_cases", missing_msg)
        pct = agent_data.get("pass_rate_pct")
        pct_str = f"{float(pct):.2f}%" if pct is not None else missing_msg
        lines.append(f"| Total Test Cases | {total} |")
        lines.append(f"| Passed Cases | {passed} |")
        lines.append(f"| Pass Rate (%) | {pct_str} |")
    return "\n".join(lines)


def _replace_between_markers(text: str, start_marker: str, end_marker: str, body: str) -> str:
    """Replace content strictly between start_marker and end_marker."""
    pattern = re.compile(
        re.escape(start_marker) + r".*?" + re.escape(end_marker),
        re.DOTALL,
    )
    if not pattern.search(text):
        raise ValueError(f"Markers '{start_marker}' ... '{end_marker}' not found in README.")
    replacement = f"{start_marker}\n{body}\n{end_marker}"
    return pattern.sub(replacement, text, count=1)


def update_readme(
    readme_path: Path = DEFAULT_README_PATH,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    utc_date: str | None = None,
) -> str:
    """Read results/*.json and idempotently update README.md Results tables.

    Args:
        readme_path: Path to the target README.md file.
        results_dir: Directory containing forecast_metrics.json, anomaly_metrics.json,
            and agent_eval.json.
        utc_date: Optional YYYY-MM-DD UTC date string override for deterministic testing.

    Returns:
        Updated README markdown content.
    """
    content = readme_path.read_text(encoding="utf-8")
    date_str = utc_date or datetime.now(UTC).strftime("%Y-%m-%d")
    generated_line = f"Results generated on {date_str} from the files in results/"

    forecast_data = _load_json(results_dir / "forecast_metrics.json")
    anomaly_data = _load_json(results_dir / "anomaly_metrics.json")
    agent_data = _load_json(results_dir / "agent_eval.json")

    if _GENERATED_LINE_RE.search(content):
        content = _GENERATED_LINE_RE.sub(generated_line, content, count=1)
    elif FORECAST_START in content:
        content = content.replace(
            FORECAST_START,
            f"{generated_line}\n\n{FORECAST_START}",
            1,
        )

    content = _replace_between_markers(
        content,
        FORECAST_START,
        FORECAST_END,
        render_forecast_table(forecast_data),
    )
    content = _replace_between_markers(
        content,
        ANOMALY_START,
        ANOMALY_END,
        render_anomaly_table(anomaly_data),
    )
    content = _replace_between_markers(
        content,
        AGENT_START,
        AGENT_END,
        render_agent_table(agent_data),
    )

    readme_path.write_text(content, encoding="utf-8")
    return content


def main() -> None:
    """CLI entry point to refresh README.md from results/*.json."""
    update_readme()
    print("Updated README.md Results tables from results/.")


if __name__ == "__main__":
    main()
