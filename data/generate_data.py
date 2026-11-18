"""Deterministic synthetic dataset generator for predictive maintenance.

Generates:
1. data/sensor_readings.csv: 5 machines (M1..M5), 90 days of hourly readings
   (10,800 rows) with slow trends, daily seasonality, Gaussian noise, and ~2%
   labelled anomalies (sudden spikes, gradual drift, flat-line sensor faults).
2. data/maintenance.db: SQLite database with maintenance_logs table (~40 rows)
   corresponding to the injected anomaly episodes.
3. docs/*.md: 4 maintenance-manual markdown files for RAG retrieval.
"""

from __future__ import annotations

import os
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
NUM_DAYS = 90
HOURS_PER_DAY = 24
TOTAL_HOURS = NUM_DAYS * HOURS_PER_DAY  # 2160 hours per machine
MACHINES = ("M1", "M2", "M3", "M4", "M5")
SENSORS = ("temperature", "vibration", "pressure", "rpm")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
DEFAULT_DOCS_DIR = Path(os.environ.get("DOCS_DIR", str(BASE_DIR / "docs")))


@dataclass(frozen=True)
class MachineBaseline:
    """Baseline operating parameters for a machine."""

    temp_base: float
    vib_base: float
    press_base: float
    rpm_base: float


MACHINE_BASELINES: dict[str, MachineBaseline] = {
    "M1": MachineBaseline(temp_base=68.0, vib_base=1.80, press_base=105.0, rpm_base=1480.0),
    "M2": MachineBaseline(temp_base=72.0, vib_base=2.05, press_base=101.0, rpm_base=1520.0),
    "M3": MachineBaseline(temp_base=65.5, vib_base=1.65, press_base=108.5, rpm_base=1440.0),
    "M4": MachineBaseline(temp_base=74.5, vib_base=2.20, press_base=98.5, rpm_base=1560.0),
    "M5": MachineBaseline(temp_base=70.0, vib_base=1.90, press_base=103.0, rpm_base=1500.0),
}

MANUAL_DOCS: dict[str, str] = {
    "overheating.md": (
        "# Overheating Diagnostics & Maintenance Manual\n\n"
        "## Overview\n"
        "Elevated operating temperature in rotating industrial machinery accelerates\n"
        "lubricant breakdown, increases thermal expansion clearances, and leads to\n"
        "premature stator or bearing failure.\n\n"
        "## Typical Symptoms\n"
        "- Sustained or sudden temperature rise above normal baseline (65-80 C).\n"
        "- Gradual upward temperature drift over 6-24 hours with elevated RPM.\n"
        "- Secondary increases in vibration due to thermal misalignment.\n\n"
        "## Likely Causes\n"
        "1. **Cooling System Degradation**: Blocked fins or failing cooling fan.\n"
        "2. **Lubrication Failure**: Low oil level or clogged oil filter.\n"
        "3. **Mechanical Overload**: Excessive torque or prolonged high-RPM load.\n"
        "4. **Electrical Winding Stress**: Unbalanced voltage causing resistive heat.\n\n"
        "## Recommended Actions\n"
        "- **Immediate**: Inspect coolant flow, radiator cleanliness, and oil level.\n"
        "- **Short-Term**: Sample lubricant for wear particles and reduce load.\n"
        "- **Corrective**: Flush cooling circuit, replace clogged oil filters,\n"
        "  replenish synthetic lubricant, and run infrared thermography.\n"
    ),
    "bearing_vibration.md": (
        "# Bearing Vibration Diagnostics & Maintenance Manual\n\n"
        "## Overview\n"
        "Excessive mechanical vibration is an early indicator of rolling-element\n"
        "bearing wear, rotor unbalance, shaft misalignment, and looseness.\n\n"
        "## Typical Symptoms\n"
        "- Gradual upward drift in RMS vibration velocity (above 3.5 mm/s).\n"
        "- Sudden transient vibration spikes synchronized with RPM fluctuations.\n"
        "- Accompanying rise in housing temperature from rolling friction.\n\n"
        "## Likely Causes\n"
        "1. **Bearing Spalling or Fatigue**: Micro-cracks on races or rollers.\n"
        "2. **Shaft Misalignment**: Angular or parallel coupling offset.\n"
        "3. **Rotor Unbalance**: Asymmetric wear or missing balance weights.\n"
        "4. **Mounting Looseness**: Loose anchor bolts or worn housing seats.\n\n"
        "## Recommended Actions\n"
        "- **Immediate**: Perform vibration spectrum check (1X, 2X RPM harmonics).\n"
        "- **Short-Term**: Schedule shutdown to inspect coupling alignment.\n"
        "- **Corrective**: Replace worn bearings, laser-align motor coupling,\n"
        "  torque foundation bolts, and dynamically balance rotor assembly.\n"
    ),
    "pressure_loss.md": (
        "# Pressure Loss Diagnostics & Maintenance Manual\n\n"
        "## Overview\n"
        "Hydraulic and pneumatic pressure stability is critical for process\n"
        "throughput, seal integrity, and actuator responsiveness.\n\n"
        "## Typical Symptoms\n"
        "- Gradual downward drift in line pressure below nominal range (95-115 psi).\n"
        "- Sudden pressure drop spikes paired with compensatory RPM surges.\n"
        "- Cavitation noise or erratic pressure oscillations at pump discharge.\n\n"
        "## Likely Causes\n"
        "1. **Seal or Gasket Leakage**: Worn shaft seals or loose flange fittings.\n"
        "2. **Pump Wear**: Internal impeller erosion reducing volumetric efficiency.\n"
        "3. **Suction Restriction**: Clogged intake strainer causing cavitation.\n"
        "4. **Relief Valve Fault**: Stuck-open bypass releasing pressure early.\n\n"
        "## Recommended Actions\n"
        "- **Immediate**: Inspect manifold flanges and mechanical seals for leaks.\n"
        "- **Short-Term**: Test relief valve cracking pressure and check-valve seat.\n"
        "- **Corrective**: Replace leaking seals and gaskets, clean intake\n"
        "  strainers, and rebuild or swap degraded pump cartridges.\n"
    ),
    "sensor_faults.md": (
        "# Sensor Faults & Instrumentation Maintenance Manual\n\n"
        "## Overview\n"
        "Reliable predictive maintenance depends on accurate field sensors.\n"
        "Instrumentation faults can mask real wear or trigger false alarms.\n\n"
        "## Typical Symptoms\n"
        "- **Flat-line (Frozen) Readings**: Output stays constant or drops to 0.0.\n"
        "- **Open Circuit Dropouts**: Instantaneous drop while RPM stays normal.\n"
        "- **Wiring Intermittent Spikes**: Single-sample extreme outliers.\n\n"
        "## Likely Causes\n"
        "1. **Loose Field Wiring**: Vibration loosening or moisture in junction box.\n"
        "2. **Transducer Failure**: Open RTD element or damaged sensor cable.\n"
        "3. **PLC Input Freeze**: Stale buffer register or ADC latch-up.\n"
        "4. **Loop Power Brownout**: Interrupted 24 VDC / 4-20 mA excitation.\n\n"
        "## Recommended Actions\n"
        "- **Immediate**: Cross-check suspect channel and verify 4-20 mA loop.\n"
        "- **Short-Term**: Inspect armored sensor cable and M12 connectors.\n"
        "- **Corrective**: Reseat or replace faulty transducer, clean oxidized\n"
        "  terminals, and recalibrate transmitter zero/span.\n"
    ),
}

EPISODE_TEMPLATES: tuple[tuple[str, int, int, str, str, str, float], ...] = (
    (
        "spike",
        140,
        2,
        "temperature",
        "Sudden thermal spike (+31 C)",
        "Flushed heat exchanger and cleared coolant fan intake",
        2.5,
    ),
    (
        "drift",
        380,
        7,
        "vibration",
        "Gradual bearing vibration drift",
        "Lubricated bearing housing and inspected shaft alignment",
        4.0,
    ),
    (
        "flatline",
        620,
        8,
        "pressure",
        "Flat-line pressure transducer fault",
        "Replaced corroded 4-20 mA pressure transmitter cable",
        1.5,
    ),
    (
        "spike",
        890,
        2,
        "vibration",
        "Transient vibration shock spike",
        "Torqued loose motor mounting bolts and checked coupling",
        2.0,
    ),
    (
        "drift",
        1180,
        7,
        "pressure",
        "Progressive hydraulic pressure loss",
        "Replaced worn mechanical shaft seal and cleaned intake strainer",
        5.0,
    ),
    (
        "flatline",
        1470,
        8,
        "temperature",
        "Frozen RTD temperature channel",
        "Reseated thermocouple terminal block and recalibrated RTD",
        1.5,
    ),
    (
        "drift",
        1740,
        7,
        "temperature",
        "Gradual overheating prior to thermal trip",
        "Replaced clogged oil filter and replenished synthetic oil",
        4.5,
    ),
    (
        "spike",
        2040,
        2,
        "rpm",
        "Sudden RPM surge and bearing load spike",
        "Recalibrated VFD speed controller and inspected rotor balance",
        3.0,
    ),
)


def ensure_docs(docs_dir: Path = DEFAULT_DOCS_DIR) -> list[Path]:
    """Create the 4 maintenance manual markdown files in docs_dir if missing.

    Args:
        docs_dir: Target directory for maintenance markdown documents.

    Returns:
        Sorted list of paths to created/verified markdown files.
    """
    docs_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, content in MANUAL_DOCS.items():
        file_path = docs_dir / filename
        if not file_path.exists():
            file_path.write_text(content, encoding="utf-8")
        paths.append(file_path)
    return sorted(paths)


def generate_dataset(
    data_dir: Path = DEFAULT_DATA_DIR,
    docs_dir: Path = DEFAULT_DOCS_DIR,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate deterministic synthetic sensor readings and maintenance logs.

    Args:
        data_dir: Directory where sensor_readings.csv and maintenance.db are saved.
        docs_dir: Directory where maintenance manual markdown files are ensured.
        seed: Fixed random seed for reproducibility (default 42).

    Returns:
        Tuple of (sensor_df, logs_df).
    """
    random.seed(seed)
    rng = np.random.default_rng(seed)

    data_dir.mkdir(parents=True, exist_ok=True)
    ensure_docs(docs_dir)

    timestamps = pd.date_range(
        start="2026-01-01T00:00:00",
        periods=TOTAL_HOURS,
        freq="h",
    )
    hours = np.arange(TOTAL_HOURS, dtype=float)
    daily_phase = 2.0 * np.pi * (hours % HOURS_PER_DAY) / HOURS_PER_DAY

    machine_frames: list[pd.DataFrame] = []
    maintenance_rows: list[dict[str, object]] = []
    log_counter = 1

    for m_idx, machine_id in enumerate(MACHINES):
        base = MACHINE_BASELINES[machine_id]
        phase_shift = m_idx * 0.45

        temp = (
            base.temp_base
            + 0.0018 * hours
            + 2.4 * np.sin(daily_phase + phase_shift)
            + rng.normal(0.0, 0.75, size=TOTAL_HOURS)
        )
        vib = (
            base.vib_base
            + 0.00018 * hours
            + 0.18 * np.sin(daily_phase + phase_shift + 0.3)
            + rng.normal(0.0, 0.08, size=TOTAL_HOURS)
        )
        press = (
            base.press_base
            - 0.0012 * hours
            + 1.6 * np.cos(daily_phase + phase_shift)
            + rng.normal(0.0, 0.65, size=TOTAL_HOURS)
        )
        rpm = (
            base.rpm_base
            + 0.012 * hours
            + 22.0 * np.sin(daily_phase + phase_shift + 0.7)
            + rng.normal(0.0, 7.5, size=TOTAL_HOURS)
        )
        is_anomaly = np.zeros(TOTAL_HOURS, dtype=int)

        machine_offset = m_idx * 17
        for kind, start_base, duration, target_sensor, issue, action, downtime in EPISODE_TEMPLATES:
            start = start_base + machine_offset
            end = min(start + duration, TOTAL_HOURS)
            is_anomaly[start:end] = 1

            if kind == "spike":
                if target_sensor == "temperature":
                    temp[start:end] += rng.uniform(26.0, 34.0, size=end - start)
                    vib[start:end] += rng.uniform(0.9, 1.5, size=end - start)
                elif target_sensor == "vibration":
                    vib[start:end] += rng.uniform(3.8, 5.2, size=end - start)
                    temp[start:end] += rng.uniform(6.0, 10.0, size=end - start)
                elif target_sensor == "rpm":
                    rpm[start:end] += rng.uniform(360.0, 460.0, size=end - start)
                    vib[start:end] += rng.uniform(2.2, 3.4, size=end - start)
                    temp[start:end] += rng.uniform(8.0, 13.0, size=end - start)

            elif kind == "drift":
                ramp = np.linspace(0.25, 1.0, end - start)
                if target_sensor == "vibration":
                    vib[start:end] += 3.8 * ramp
                    temp[start:end] += 9.5 * ramp
                elif target_sensor == "pressure":
                    press[start:end] -= 28.0 * ramp
                    rpm[start:end] += 140.0 * ramp
                elif target_sensor == "temperature":
                    temp[start:end] += 26.0 * ramp
                    vib[start:end] += 1.4 * ramp

            elif kind == "flatline":
                if target_sensor == "pressure":
                    press[start:end] = 0.0
                elif target_sensor == "temperature":
                    temp[start:end] = round(float(temp[start - 1]) - 35.0, 2)

            log_ts = timestamps[end - 1].strftime("%Y-%m-%d")
            maintenance_rows.append(
                {
                    "log_id": f"LOG-{log_counter:04d}",
                    "machine_id": machine_id,
                    "date": log_ts,
                    "issue": issue,
                    "action_taken": action,
                    "downtime_hours": float(downtime),
                }
            )
            log_counter += 1

        machine_df = pd.DataFrame(
            {
                "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%S"),
                "machine_id": machine_id,
                "temperature": np.round(temp, 3),
                "vibration": np.round(np.clip(vib, 0.0, None), 3),
                "pressure": np.round(np.clip(press, 0.0, None), 3),
                "rpm": np.round(np.clip(rpm, 0.0, None), 2),
                "is_anomaly": is_anomaly,
            }
        )
        machine_frames.append(machine_df)

    sensor_df = pd.concat(machine_frames, ignore_index=True)
    sensor_csv_path = data_dir / "sensor_readings.csv"
    sensor_df.to_csv(sensor_csv_path, index=False)

    logs_df = pd.DataFrame(maintenance_rows)
    db_path = data_dir / "maintenance.db"
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE maintenance_logs (
                log_id TEXT PRIMARY KEY,
                machine_id TEXT NOT NULL,
                date TEXT NOT NULL,
                issue TEXT NOT NULL,
                action_taken TEXT NOT NULL,
                downtime_hours REAL NOT NULL
            )
            """
        )
        logs_df.to_sql("maintenance_logs", conn, if_exists="append", index=False)
        conn.commit()

    return sensor_df, logs_df


def main() -> None:
    """CLI entry point for generating synthetic sensor and maintenance data."""
    sensor_df, logs_df = generate_dataset()
    anomaly_count = int(sensor_df["is_anomaly"].sum())
    anomaly_pct = 100.0 * anomaly_count / len(sensor_df)
    print(
        f"Generated {len(sensor_df)} sensor rows "
        f"({anomaly_count} anomalies, {anomaly_pct:.2f}%) "
        f"and {len(logs_df)} maintenance logs."
    )


if __name__ == "__main__":
    main()
