"""CrewAI 3-agent predictive maintenance pipeline powered by Gemini."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

from crewai import LLM, Agent, Crew, Process, Task

from agents.tools import build_langchain_tools, wrap_for_crewai
from data.generate_data import MACHINES, SENSORS


class MissingAPIKeyError(RuntimeError):
    """Raised when GOOGLE_API_KEY is not configured for the CrewAI Gemini agents."""


class MissingModelError(RuntimeError):
    """Raised when GEMINI_MODEL is not configured for the CrewAI Gemini agents."""


@dataclass(frozen=True)
class CrewRunResult:
    """Structured output returned by a CrewAI maintenance diagnosis run."""

    machine_id: str
    sensor: str
    report_text: str
    tool_outputs: list[str]
    elapsed_seconds: float
    model_name: str


def get_gemini_llm() -> tuple[LLM, str]:
    """Initialize CrewAI Gemini LLM from GOOGLE_API_KEY and GEMINI_MODEL env vars.

    Returns:
        Tuple of (CrewAI LLM instance, model identifier string).

    Raises:
        MissingAPIKeyError: If GOOGLE_API_KEY is unset or empty.
        MissingModelError: If GEMINI_MODEL is unset or empty.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise MissingAPIKeyError(
            "GOOGLE_API_KEY environment variable is missing. Set GOOGLE_API_KEY to run "
            "the AI monitoring agents. Forecasting (/forecast) and anomaly detection "
            "(/anomalies) remain available without an API key."
        )

    model_name = os.environ.get("GEMINI_MODEL", "").strip()
    if not model_name:
        raise MissingModelError(
            "GEMINI_MODEL environment variable is missing. Set GEMINI_MODEL (for example in "
            ".env) to a supported Gemini model name from Google AI Studio."
        )
    if not model_name.startswith("gemini/"):
        model_name = f"gemini/{model_name}"

    llm = LLM(
        model=model_name,
        api_key=api_key,
        temperature=0.1,
    )
    return llm, model_name


def run_monitoring_crew(
    machine_id: str,
    sensor: str = "vibration",
    notes: str = "",
    lookback_hours: int = 168,
    max_steps: int = 5,
    timeout_seconds: int = 60,
    extra_instruction: str = "",
    data_dir: Path | None = None,
    docs_dir: Path | None = None,
) -> CrewRunResult:
    """Execute the 3-agent CrewAI pipeline (Monitor -> Diagnosis -> Report).

    Args:
        machine_id: Target equipment ID (M1..M5).
        sensor: Primary sensor channel to forecast and inspect.
        notes: Optional operator context notes.
        lookback_hours: Trailing window in hours for anomaly scanning.
        max_steps: Maximum reasoning iterations per agent.
        timeout_seconds: Execution timeout budget in seconds.
        extra_instruction: Optional corrective instruction appended on guardrail retry.
        data_dir: Optional data directory override.
        docs_dir: Optional docs directory override.

    Returns:
        CrewRunResult with final report text and raw captured tool outputs.
    """
    if machine_id not in MACHINES:
        raise ValueError(f"Unknown machine_id '{machine_id}'. Must be one of {MACHINES}.")
    if sensor not in SENSORS:
        raise ValueError(f"Unknown sensor '{sensor}'. Must be one of {SENSORS}.")

    llm, model_name = get_gemini_llm()

    tool_outputs: list[str] = []
    lc_tools = build_langchain_tools(data_dir=data_dir, docs_dir=docs_dir)
    crew_tools = wrap_for_crewai(lc_tools, output_log=tool_outputs)

    monitor_agent = Agent(
        role="Equipment Monitor Engineer",
        goal=(
            f"Collect accurate 24-hour sensor forecasts and recent anomaly detections "
            f"for machine {machine_id} ({sensor})."
        ),
        backstory=(
            "You monitor industrial rotating equipment telemetry and summarize quantitative "
            "forecasts and anomaly flags strictly from monitoring tools."
        ),
        tools=[crew_tools["forecast_tool"], crew_tools["anomaly_tool"]],
        llm=llm,
        max_iter=max_steps,
        max_execution_time=timeout_seconds,
        allow_delegation=False,
        verbose=False,
    )

    diagnosis_agent = Agent(
        role="Maintenance Diagnosis Specialist",
        goal=(
            f"Correlate telemetry anomalies for {machine_id} with historical SQL maintenance "
            "logs and maintenance manual procedures."
        ),
        backstory=(
            "You inspect SQLite maintenance_logs using read-only SELECT queries and consult "
            "engineering manuals via RAG to identify likely root causes."
        ),
        tools=[crew_tools["maintenance_log_tool"], crew_tools["rag_tool"]],
        llm=llm,
        max_iter=max_steps,
        max_execution_time=timeout_seconds,
        allow_delegation=False,
        verbose=False,
    )

    report_agent = Agent(
        role="Reliability Maintenance Reporter",
        goal=(
            f"Write a concise, grounded maintenance recommendation report for {machine_id} "
            "citing ONLY numbers observed directly in tool outputs."
        ),
        backstory=(
            "You author zero-hallucination reliability engineering reports. You NEVER invent "
            "numbers, thresholds, or metrics. You use bullet points (-) instead of numbered lists."
        ),
        tools=[],
        llm=llm,
        max_iter=max(2, min(max_steps, 3)),
        max_execution_time=timeout_seconds,
        allow_delegation=False,
        verbose=False,
    )

    context_note = f" Operator notes: {notes.strip()}" if notes.strip() else ""

    monitor_task = Task(
        description=(
            f"1. Call forecast_tool with machine_id='{machine_id}', sensor='{sensor}', "
            f"horizon_hours=24.\n"
            f"2. Call anomaly_tool with machine_id='{machine_id}', "
            f"lookback_hours={lookback_hours}, method='both'.\n"
            f"Summarize the exact numeric outputs returned by both tools.{context_note}"
        ),
        expected_output=(
            "A factual summary containing the exact forecast metrics, predicted range, "
            "best model, anomaly count, and recent anomalous timestamps/values."
        ),
        agent=monitor_agent,
    )

    diagnosis_task = Task(
        description=(
            f"1. Query maintenance_log_tool with a read-only SQL SELECT query on "
            f"maintenance_logs WHERE machine_id = '{machine_id}' ORDER BY date DESC.\n"
            "2. Query rag_tool with a concise search string describing the observed sensor "
            f"behavior or {sensor} anomalies on {machine_id}.\n"
            "Identify likely root causes and past recurring issues."
        ),
        expected_output=(
            "A structured diagnostic summary linking historical maintenance logs and "
            "manual procedures to the observed telemetry."
        ),
        agent=diagnosis_agent,
        context=[monitor_task],
    )

    extra_block = f"\n{extra_instruction.strip()}" if extra_instruction.strip() else ""
    report_task = Task(
        description=(
            f"Write a short, actionable maintenance recommendation report for machine {machine_id} "
            f"(focus sensor: {sensor}).\n"
            "STRICT GROUNDING RULES:\n"
            "- Mention the machine ID explicitly.\n"
            "- Clearly state whether anomalies were detected and cite exact anomaly "
            "counts/values.\n"
            "- Recommend concrete inspection and corrective maintenance actions.\n"
            "- Every single number in your report MUST appear verbatim in the tool outputs from "
            "the monitor and diagnosis tasks. Do NOT invent any numbers, percentages, or counts.\n"
            "- Use '-' bullet points instead of numbered lists (1., 2., 3.) so list indices are "
            "not mistaken for ungrounded quantities."
            f"{extra_block}"
        ),
        expected_output=(
            "A concise, grounded maintenance recommendation report using bullet points and "
            "strictly grounded tool numbers."
        ),
        agent=report_agent,
        context=[monitor_task, diagnosis_task],
    )

    crew = Crew(
        agents=[monitor_agent, diagnosis_agent, report_agent],
        tasks=[monitor_task, diagnosis_task, report_task],
        process=Process.sequential,
        verbose=False,
    )

    start_time = time.perf_counter()
    result = crew.kickoff()
    elapsed = round(time.perf_counter() - start_time, 3)

    report_text = str(getattr(result, "raw", result)).strip()
    return CrewRunResult(
        machine_id=machine_id,
        sensor=sensor,
        report_text=report_text,
        tool_outputs=tool_outputs,
        elapsed_seconds=elapsed,
        model_name=model_name,
    )
