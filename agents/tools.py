"""LangChain tools and CrewAI adapters for monitoring, SQL logs, and RAG."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from crewai.tools import BaseTool as CrewAIBaseTool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agents.rag import MaintenanceManualRAG
from anomaly.detector import detect_machine_anomalies
from forecasting.forecaster import evaluate_and_forecast

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
DEFAULT_DOCS_DIR = Path(os.environ.get("DOCS_DIR", str(BASE_DIR / "docs")))

_FORBIDDEN_SQL_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|reindex)\b",
    re.IGNORECASE,
)


class ForecastToolInput(BaseModel):
    """Input schema for forecast_tool."""

    machine_id: str = Field(..., description="Machine ID (M1, M2, M3, M4, or M5).")
    sensor: str = Field(..., description="Sensor name: temperature, vibration, pressure, or rpm.")
    horizon_hours: int = Field(default=24, ge=1, le=72, description="Forecast horizon in hours.")


class AnomalyToolInput(BaseModel):
    """Input schema for anomaly_tool."""

    machine_id: str = Field(..., description="Machine ID (M1, M2, M3, M4, or M5).")
    lookback_hours: int = Field(
        default=168,
        ge=24,
        le=2160,
        description="Trailing hours to scan for anomalies.",
    )
    method: str = Field(
        default="both",
        description="Detector to run: 'isolation_forest', 'rolling_zscore', or 'both'.",
    )


class MaintenanceLogToolInput(BaseModel):
    """Input schema for maintenance_log_tool."""

    sql_query: str = Field(
        ...,
        description=(
            "Read-only SQL SELECT query against table 'maintenance_logs' "
            "(columns: log_id, machine_id, date, issue, action_taken, downtime_hours)."
        ),
    )


class RagToolInput(BaseModel):
    """Input schema for rag_tool."""

    query: str = Field(
        ...,
        description="Diagnostic symptom or equipment fault query to search maintenance manuals.",
    )
    top_k: int = Field(default=3, ge=1, le=6, description="Number of manual sections to retrieve.")


def run_forecast_tool(
    machine_id: str,
    sensor: str,
    horizon_hours: int = 24,
    data_path: Path | None = None,
) -> str:
    """Run 24-hour sensor forecast and return JSON summary string."""
    result = evaluate_and_forecast(
        machine_id=machine_id,
        sensor=sensor,
        horizon_hours=horizon_hours,
        data_path=data_path,
    )
    values = [pt["predicted_value"] for pt in result["forecast"]]
    summary = {
        "machine_id": result["machine_id"],
        "sensor": result["sensor"],
        "horizon_hours": result["horizon_hours"],
        "last_observed_timestamp": result["last_observed_timestamp"],
        "last_observed_value": result["last_observed_value"],
        "best_model": result["best_model"],
        "model_metrics": result["model_metrics"],
        "forecast_stats": {
            "min_predicted": round(min(values), 3),
            "max_predicted": round(max(values), 3),
            "mean_predicted": round(sum(values) / len(values), 3),
            "first_hour_predicted": values[0],
            "last_hour_predicted": values[-1],
        },
        "forecast_sample": result["forecast"][:6],
    }
    return json.dumps(summary, indent=2)


def run_anomaly_tool(
    machine_id: str,
    lookback_hours: int = 168,
    method: str = "both",
    data_path: Path | None = None,
) -> str:
    """Run anomaly detection on recent machine readings and return JSON summary."""
    result = detect_machine_anomalies(
        machine_id=machine_id,
        lookback_hours=lookback_hours,
        method=method,
        data_path=data_path,
    )
    recent_anomalies = result["anomalies"][-10:]
    summary = {
        "machine_id": result["machine_id"],
        "lookback_hours": result["lookback_hours"],
        "method": result["method"],
        "rows_scanned": result["rows_scanned"],
        "anomaly_count": result["anomaly_count"],
        "recent_anomalies": recent_anomalies,
    }
    return json.dumps(summary, indent=2)


def _validate_read_only_sql(sql_query: str) -> str:
    """Validate that sql_query is a single read-only SELECT or WITH query."""
    cleaned = sql_query.strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("SQL query must not be empty.")
    if ";" in cleaned:
        raise ValueError("Multiple SQL statements are not permitted.")
    if "--" in cleaned or "/*" in cleaned:
        raise ValueError("SQL comments are not permitted.")
    lowered = cleaned.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("Only read-only SELECT queries are permitted.")
    if _FORBIDDEN_SQL_RE.search(cleaned):
        raise ValueError("Forbidden SQL keyword detected; only read-only SELECT is allowed.")
    return cleaned


def run_maintenance_log_tool(sql_query: str, db_path: Path | None = None) -> str:
    """Execute a validated read-only SQL query against data/maintenance.db."""
    cleaned_sql = _validate_read_only_sql(sql_query)
    target_db = db_path or (DEFAULT_DATA_DIR / "maintenance.db")
    if not target_db.exists():
        raise FileNotFoundError(
            f"Database not found at {target_db}. Run 'python -m data.generate_data' first."
        )

    uri = f"file:{target_db.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(cleaned_sql)
        rows = [dict(row) for row in cursor.fetchmany(50)]

    return json.dumps({"row_count": len(rows), "rows": rows}, indent=2)


def build_langchain_tools(
    data_dir: Path | None = None,
    docs_dir: Path | None = None,
) -> dict[str, StructuredTool]:
    """Create the four LangChain StructuredTool instances.

    Args:
        data_dir: Optional override directory containing sensor_readings.csv and maintenance.db.
        docs_dir: Optional override directory containing maintenance manual markdown files.

    Returns:
        Mapping from tool name to LangChain StructuredTool.
    """
    resolved_data = data_dir or DEFAULT_DATA_DIR
    resolved_docs = docs_dir or DEFAULT_DOCS_DIR
    csv_path = resolved_data / "sensor_readings.csv"
    db_path = resolved_data / "maintenance.db"
    rag_engine = MaintenanceManualRAG(docs_dir=resolved_docs)

    def _forecast(machine_id: str, sensor: str, horizon_hours: int = 24) -> str:
        return run_forecast_tool(
            machine_id=machine_id,
            sensor=sensor,
            horizon_hours=horizon_hours,
            data_path=csv_path,
        )

    def _anomalies(machine_id: str, lookback_hours: int = 168, method: str = "both") -> str:
        return run_anomaly_tool(
            machine_id=machine_id,
            lookback_hours=lookback_hours,
            method=method,
            data_path=csv_path,
        )

    def _maintenance_sql(sql_query: str) -> str:
        return run_maintenance_log_tool(sql_query=sql_query, db_path=db_path)

    def _rag_search(query: str, top_k: int = 3) -> str:
        chunks = rag_engine.search(query=query, top_k=top_k)
        payload = [
            {
                "source": chunk.source,
                "section": chunk.section,
                "score": chunk.score,
                "content": chunk.content,
            }
            for chunk in chunks
        ]
        return json.dumps({"results": payload}, indent=2)

    forecast_tool = StructuredTool.from_function(
        func=_forecast,
        name="forecast_tool",
        description=(
            "Forecast the next 24 hours for a machine (M1..M5) and sensor "
            "(temperature, vibration, pressure, rpm) and compare Naive, ARIMA, and GBR models."
        ),
        args_schema=ForecastToolInput,
    )

    anomaly_tool = StructuredTool.from_function(
        func=_anomalies,
        name="anomaly_tool",
        description=(
            "Detect recent sensor anomalies for a machine (M1..M5) using Isolation Forest "
            "and Rolling Z-Score detectors."
        ),
        args_schema=AnomalyToolInput,
    )

    maintenance_log_tool = StructuredTool.from_function(
        func=_maintenance_sql,
        name="maintenance_log_tool",
        description=(
            "Execute a read-only SQL SELECT query against table 'maintenance_logs' in "
            "maintenance.db (columns: log_id, machine_id, date, issue, action_taken, "
            "downtime_hours)."
        ),
        args_schema=MaintenanceLogToolInput,
    )

    rag_tool = StructuredTool.from_function(
        func=_rag_search,
        name="rag_tool",
        description=(
            "Search maintenance manuals in docs/ using hybrid FAISS + BM25 retrieval "
            "to find likely causes and recommended corrective actions."
        ),
        args_schema=RagToolInput,
    )

    return {
        "forecast_tool": forecast_tool,
        "anomaly_tool": anomaly_tool,
        "maintenance_log_tool": maintenance_log_tool,
        "rag_tool": rag_tool,
    }


class CrewToolAdapter(CrewAIBaseTool):
    """Adapter bridging a LangChain StructuredTool to CrewAI while capturing outputs."""

    name: str
    description: str
    args_schema: type[BaseModel]
    lc_tool: Any
    output_log: list[str]

    def _run(self, **kwargs: Any) -> str:
        """Execute underlying LangChain tool and record its output for guardrails."""
        try:
            result = str(self.lc_tool.invoke(kwargs))
        except Exception as exc:
            result = json.dumps({"error": str(exc)})
        self.output_log.append(result)
        return result


def wrap_for_crewai(
    lc_tools: dict[str, StructuredTool],
    output_log: list[str],
) -> dict[str, CrewToolAdapter]:
    """Wrap LangChain tools as CrewAI tools with shared output logging."""
    wrapped: dict[str, CrewToolAdapter] = {}
    for name, lc_tool in lc_tools.items():
        wrapped[name] = CrewToolAdapter(
            name=lc_tool.name,
            description=lc_tool.description,
            args_schema=lc_tool.args_schema,
            lc_tool=lc_tool,
            output_log=output_log,
        )
    return wrapped
