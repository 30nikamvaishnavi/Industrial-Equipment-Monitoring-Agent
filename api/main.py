"""FastAPI service with structured JSON logging, guardrails, and CrewAI timeouts."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from agents.crew import MissingAPIKeyError, MissingModelError, run_monitoring_crew
from anomaly.detector import detect_machine_anomalies
from data.generate_data import MACHINES, SENSORS
from forecasting.forecaster import evaluate_and_forecast
from guardrails import check_prompt_injection, verify_grounded_numbers

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))


class JsonLogFormatter(logging.Formatter):
    """Format log records as single-line structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize LogRecord fields and extras into JSON."""
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("request_id", "method", "path", "status_code", "duration_ms", "machine_id"):
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> logging.Logger:
    """Configure structured JSON logger for the API."""
    logger = logging.getLogger("equipment_monitor.api")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


logger = configure_logging()

app = FastAPI(
    title="Industrial Equipment Monitoring Agent",
    description=(
        "Predictive maintenance forecasting, anomaly detection, and grounded CrewAI diagnostics."
    ),
    version="1.0.0",
)


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    database_exists: bool
    dataset_exists: bool
    gemini_configured: bool
    version: str


class ForecastRequest(BaseModel):
    """Request schema for POST /forecast."""

    machine_id: Literal["M1", "M2", "M3", "M4", "M5"] = Field(..., description="Target machine ID.")
    sensor: Literal["temperature", "vibration", "pressure", "rpm"] = Field(
        ..., description="Sensor channel to forecast."
    )
    horizon_hours: int = Field(default=24, ge=1, le=72, description="Forecast horizon in hours.")


class ModelMetricSchema(BaseModel):
    """MAE and RMSE metrics for a single forecasting model."""

    mae: float
    rmse: float


class ForecastPoint(BaseModel):
    """Single timestamped forecast point."""

    timestamp: str
    predicted_value: float


class ForecastResponse(BaseModel):
    """Response schema for POST /forecast."""

    machine_id: str
    sensor: str
    horizon_hours: int
    train_rows: int
    test_rows: int
    last_observed_timestamp: str
    last_observed_value: float
    best_model: str
    model_metrics: dict[str, ModelMetricSchema]
    forecast: list[ForecastPoint]


class AnomalyRequest(BaseModel):
    """Request schema for POST /anomalies."""

    machine_id: Literal["M1", "M2", "M3", "M4", "M5"] = Field(..., description="Target machine ID.")
    lookback_hours: int = Field(default=168, ge=24, le=2160, description="Trailing hours to scan.")
    method: Literal["isolation_forest", "rolling_zscore", "both"] = Field(
        default="both", description="Anomaly detector method."
    )


class AnomalyRecord(BaseModel):
    """Detected anomalous observation details."""

    timestamp: str
    temperature: float
    vibration: float
    pressure: float
    rpm: float
    isolation_forest_flag: bool
    rolling_zscore_flag: bool
    isolation_score: float
    max_z_score: float
    affected_sensors: list[str]


class AnomalyResponse(BaseModel):
    """Response schema for POST /anomalies."""

    machine_id: str
    lookback_hours: int
    method: str
    rows_scanned: int
    anomaly_count: int
    anomalies: list[AnomalyRecord]


class ReportRequest(BaseModel):
    """Request schema for POST /report."""

    machine_id: Literal["M1", "M2", "M3", "M4", "M5"] = Field(..., description="Target machine ID.")
    sensor: Literal["temperature", "vibration", "pressure", "rpm"] = Field(
        default="vibration", description="Primary sensor channel to analyze."
    )
    notes: str = Field(default="", max_length=500, description="Optional operator context notes.")
    lookback_hours: int = Field(
        default=168, ge=24, le=2160, description="Trailing anomaly scan window."
    )
    max_steps: int = Field(default=5, ge=1, le=15, description="Max reasoning steps per agent.")
    timeout_seconds: int = Field(
        default=60, ge=5, le=180, description="Agent run timeout in seconds."
    )


class ReportResponse(BaseModel):
    """Response schema for POST /report."""

    machine_id: str
    sensor: str
    model_name: str
    elapsed_seconds: float
    guardrail_passed: bool
    retried: bool
    grounded_numbers_count: int
    tool_calls_recorded: int
    report: str


@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next: object) -> Response:
    """Log every HTTP request and response as structured JSON."""
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    response = await call_next(request)  # type: ignore[operator]
    duration_ms = round((time.perf_counter() - start) * 1000.0, 2)
    logger.info(
        "http_request_completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health, data readiness, and Gemini API configuration status."""
    csv_exists = (DEFAULT_DATA_DIR / "sensor_readings.csv").exists()
    db_exists = (DEFAULT_DATA_DIR / "maintenance.db").exists()
    gemini_configured = bool(os.environ.get("GOOGLE_API_KEY", "").strip())
    return HealthResponse(
        status="ok",
        database_exists=db_exists,
        dataset_exists=csv_exists,
        gemini_configured=gemini_configured,
        version="1.0.0",
    )


@app.post("/forecast", response_model=ForecastResponse)
async def forecast_endpoint(req: ForecastRequest) -> ForecastResponse:
    """Compare Naive, ARIMA, and GBR models and return a 24-hour sensor forecast."""
    if req.machine_id not in MACHINES or req.sensor not in SENSORS:
        raise HTTPException(status_code=400, detail="Invalid machine_id or sensor.")
    try:
        result = await asyncio.to_thread(
            evaluate_and_forecast,
            machine_id=req.machine_id,
            sensor=req.sensor,
            horizon_hours=req.horizon_hours,
        )
        return ForecastResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/anomalies", response_model=AnomalyResponse)
async def anomalies_endpoint(req: AnomalyRequest) -> AnomalyResponse:
    """Detect recent anomalies for a machine using Isolation Forest and/or Rolling Z-Score."""
    try:
        result = await asyncio.to_thread(
            detect_machine_anomalies,
            machine_id=req.machine_id,
            lookback_hours=req.lookback_hours,
            method=req.method,
        )
        return AnomalyResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _run_crew_with_timeout(
    req: ReportRequest,
    extra_instruction: str = "",
):
    """Run the CrewAI monitoring pipeline inside an async timeout boundary."""
    return await asyncio.wait_for(
        asyncio.to_thread(
            run_monitoring_crew,
            machine_id=req.machine_id,
            sensor=req.sensor,
            notes=req.notes,
            lookback_hours=req.lookback_hours,
            max_steps=req.max_steps,
            timeout_seconds=req.timeout_seconds,
            extra_instruction=extra_instruction,
        ),
        timeout=float(req.timeout_seconds),
    )


@app.post("/report", response_model=ReportResponse)
async def report_endpoint(req: ReportRequest) -> ReportResponse:
    """Run the 3-agent CrewAI maintenance diagnosis with timeout, guardrails, and 1 retry."""
    injection_check = check_prompt_injection(req.notes)
    if not injection_check.is_safe:
        logger.warning(
            "prompt_injection_blocked",
            extra={"machine_id": req.machine_id},
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "prompt_injection_detected",
                "message": injection_check.reason,
                "matched_patterns": injection_check.matched_patterns,
            },
        )

    retried = False
    try:
        crew_result = await _run_crew_with_timeout(req)
        guardrail = verify_grounded_numbers(
            report_text=crew_result.report_text,
            tool_outputs=crew_result.tool_outputs,
        )

        if not guardrail.passed:
            retried = True
            logger.warning(
                "report_guardrail_retry",
                extra={"machine_id": req.machine_id},
            )
            bad_nums = ", ".join(guardrail.ungrounded_numbers) or "none listed"
            retry_instruction = (
                "RETRY CORRECTION: Your previous report was rejected because it contained "
                f"ungrounded numbers not produced by any tool: {bad_nums}. "
                "Remove ALL of those ungrounded numbers and cite only numbers from tool outputs."
            )
            crew_result = await _run_crew_with_timeout(req, extra_instruction=retry_instruction)
            guardrail = verify_grounded_numbers(
                report_text=crew_result.report_text,
                tool_outputs=crew_result.tool_outputs,
            )
    except MissingAPIKeyError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "missing_google_api_key",
                "message": str(exc),
            },
        ) from exc
    except MissingModelError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "missing_gemini_model",
                "message": str(exc),
            },
        ) from exc
    except TimeoutError as exc:
        logger.error(
            "agent_execution_timeout",
            extra={"machine_id": req.machine_id},
        )
        raise HTTPException(
            status_code=504,
            detail={
                "error": "agent_timeout",
                "message": f"Agent execution exceeded timeout of {req.timeout_seconds} seconds.",
            },
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("agent_execution_failed", extra={"machine_id": req.machine_id})
        raise HTTPException(
            status_code=500,
            detail={"error": "agent_execution_error", "message": str(exc)},
        ) from exc

    if not guardrail.passed:
        logger.warning(
            "report_rejected_ungrounded_numbers",
            extra={"machine_id": req.machine_id},
        )
        raise HTTPException(
            status_code=422,
            detail={
                "error": "ungrounded_numbers_rejected",
                "message": guardrail.reason,
                "ungrounded_numbers": guardrail.ungrounded_numbers,
            },
        )

    return ReportResponse(
        machine_id=crew_result.machine_id,
        sensor=crew_result.sensor,
        model_name=crew_result.model_name,
        elapsed_seconds=crew_result.elapsed_seconds,
        guardrail_passed=True,
        retried=retried,
        grounded_numbers_count=len(guardrail.report_numbers),
        tool_calls_recorded=len(crew_result.tool_outputs),
        report=crew_result.report_text,
    )
