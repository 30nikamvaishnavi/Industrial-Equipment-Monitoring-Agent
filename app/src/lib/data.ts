export const SEED = 42;
export const NUM_DAYS = 90;
export const HOURS_PER_DAY = 24;
export const TOTAL_HOURS = NUM_DAYS * HOURS_PER_DAY; // 2160 hours per machine

export const MACHINES = ["M1", "M2", "M3", "M4", "M5"] as const;
export const SENSORS = ["temperature", "vibration", "pressure", "rpm"] as const;

export type MachineId = (typeof MACHINES)[number];
export type SensorName = (typeof SENSORS)[number];

export interface SensorReading {
  timestamp: string;
  hourIndex: number;
  machine_id: MachineId;
  temperature: number;
  vibration: number;
  pressure: number;
  rpm: number;
  is_anomaly: 0 | 1;
}

export interface MaintenanceLog {
  log_id: string;
  machine_id: MachineId;
  date: string;
  issue: string;
  action_taken: string;
  downtime_hours: number;
}

export interface ManualDoc {
  filename: string;
  title: string;
  content: string;
}

export interface MachineSummary {
  machine_id: MachineId;
  rows: number;
  anomalies: number;
  anomalyRatePct: number;
  logs: number;
}

export interface DatasetStats {
  seed: number;
  totalRows: number;
  machinesCount: number;
  hoursPerMachine: number;
  anomalyCount: number;
  anomalyRatePct: number;
  logsCount: number;
  perMachine: MachineSummary[];
}

export interface SyntheticDataset {
  readings: SensorReading[];
  logs: MaintenanceLog[];
  stats: DatasetStats;
}

interface MachineBaseline {
  tempBase: number;
  vibBase: number;
  pressBase: number;
  rpmBase: number;
}

const MACHINE_BASELINES: Record<MachineId, MachineBaseline> = {
  M1: { tempBase: 68.0, vibBase: 1.8, pressBase: 105.0, rpmBase: 1480.0 },
  M2: { tempBase: 72.0, vibBase: 2.05, pressBase: 101.0, rpmBase: 1520.0 },
  M3: { tempBase: 65.5, vibBase: 1.65, pressBase: 108.5, rpmBase: 1440.0 },
  M4: { tempBase: 74.5, vibBase: 2.2, pressBase: 98.5, rpmBase: 1560.0 },
  M5: { tempBase: 70.0, vibBase: 1.9, pressBase: 103.0, rpmBase: 1500.0 },
};

export const MANUAL_DOCS: ManualDoc[] = [
  {
    filename: "overheating.md",
    title: "Overheating Diagnostics & Maintenance Manual",
    content: `# Overheating Diagnostics & Maintenance Manual

## Overview
Elevated operating temperature in rotating industrial machinery accelerates lubricant breakdown, increases thermal expansion clearances, and leads to premature stator or bearing failure.

## Typical Symptoms
- Sustained or sudden temperature rise above normal operating baseline (typically 65-80 C depending on machine class).
- Gradual upward temperature drift over 6-24 hours accompanied by elevated RPM or friction load.
- Secondary increases in vibration due to thermal misalignment.

## Likely Causes
1. Cooling System Degradation: Blocked heat exchanger fins, fouled coolant lines, or failing cooling fan.
2. Lubrication Failure: Low oil level, degraded viscosity, or clogged oil filter increasing bearing friction.
3. Mechanical Overload: Excessive process torque or prolonged high-RPM operation beyond rated duty cycle.
4. Electrical Winding Stress: Unbalanced phase voltage or degraded motor insulation causing resistive heating.

## Recommended Actions
- Immediate: Inspect coolant flow rate, radiator cleanliness, and cooling fan operation; verify oil sight-glass level and oil temperature.
- Short-Term: Sample lubricant for thermal oxidation and metal wear particles; reduce operating load or RPM if temperature exceeds critical alarm threshold.
- Corrective: Flush cooling circuit, replace clogged oil filters, replenish synthetic lubricant, and perform infrared thermography on motor windings and bearing housings.`,
  },
  {
    filename: "bearing_vibration.md",
    title: "Bearing Vibration Diagnostics & Maintenance Manual",
    content: `# Bearing Vibration Diagnostics & Maintenance Manual

## Overview
Excessive mechanical vibration is a primary early indicator of rolling-element bearing wear, rotor unbalance, shaft misalignment, and structural looseness.

## Typical Symptoms
- Gradual upward drift in RMS vibration velocity (above 3.5 mm/s) prior to mechanical seizure.
- Sudden transient vibration spikes synchronized with RPM fluctuations or load shocks.
- Accompanying rise in housing temperature due to increased rolling friction.

## Likely Causes
1. Bearing Spalling or Fatigue: Subsurface micro-cracks on inner race, outer race, or rolling elements.
2. Shaft Misalignment: Angular or parallel offset between drive motor and driven equipment coupling.
3. Rotor Unbalance: Material buildup, asymmetric wear, or missing balance weights on rotating assemblies.
4. Foundation or Mounting Looseness: Loose anchor bolts or worn bearing housing seats.

## Recommended Actions
- Immediate: Perform handheld vibration spectrum check (1X, 2X RPM harmonics and bearing fault frequencies) and listen for acoustic knocking.
- Short-Term: Schedule controlled shutdown during next maintenance window to inspect coupling alignment and grease condition.
- Corrective: Replace worn rolling bearings, laser-align motor-to-shaft coupling, torque foundation bolts to specification, and dynamically balance rotor assembly.`,
  },
  {
    filename: "pressure_loss.md",
    title: "Pressure Loss Diagnostics & Maintenance Manual",
    content: `# Pressure Loss Diagnostics & Maintenance Manual

## Overview
Hydraulic and pneumatic operating pressure stability is critical for process throughput, seal integrity, and actuator responsiveness. Unexplained pressure drops indicate leaks, pump wear, or flow restrictions.

## Typical Symptoms
- Gradual downward drift in line pressure below nominal operating range (typically 95-115 psi).
- Sudden pressure drop spikes paired with compensatory RPM surges as control loops attempt to maintain flow.
- Cavitation noise or erratic pressure oscillations at pump discharge.

## Likely Causes
1. Seal or Gasket Leakage: Worn shaft mechanical seals, cracked O-rings, or loose flange fittings.
2. Pump / Compressor Wear: Internal impeller erosion, worn vane tips, or valve plate fatigue reducing volumetric efficiency.
3. Suction Line Restriction: Clogged intake strainer or partially closed isolation valve causing pump cavitation.
4. Pressure Relief Valve Fault: Stuck-open bypass or fatigued spring releasing system pressure prematurely.

## Recommended Actions
- Immediate: Visually inspect manifold flanges, hose fittings, and mechanical seals for weeping or audible air/fluid leaks; verify suction strainer differential pressure.
- Short-Term: Test pressure relief valve cracking pressure and check pump discharge check-valve seating.
- Corrective: Replace leaking mechanical seals and flange gaskets, clean or replace intake filters, and rebuild or swap degraded pump cartridges.`,
  },
  {
    filename: "sensor_faults.md",
    title: "Sensor Faults & Instrumentation Maintenance Manual",
    content: `# Sensor Faults & Instrumentation Maintenance Manual

## Overview
Reliable predictive maintenance depends on accurate field instrumentation. Sensor faults can mask real mechanical degradation or trigger false shutdown alarms if not distinguished from physical machine faults.

## Typical Symptoms
- Flat-line (Frozen) Readings: Sensor output remains completely constant with zero variance over consecutive hours, or drops abruptly to 0.0.
- Signal Dropouts / Open Circuit: Instantaneous drop to rail minimum or NaN/zero values while correlated sensors (e.g., RPM or temperature) remain normal.
- Wiring Intermittent Spikes: Single-sample extreme outliers inconsistent with physical thermal or mechanical inertia.

## Likely Causes
1. Loose Field Wiring or Terminal Corrosion: Vibration-induced terminal screw loosening or moisture ingress in junction boxes.
2. Transducer Failure: Open RTD element, damaged piezoelectric accelerometer cable, or clogged pressure impulse tube.
3. PLC / DAQ Input Module Freeze: Stale buffer register or analog-to-digital converter latch-up.
4. Power Supply Brownout: Insufficient loop excitation voltage (24 VDC / 4-20 mA loop interruption).

## Recommended Actions
- Immediate: Cross-check suspect channel against redundant sensors and inspect PLC/SCADA communication diagnostics; verify 4-20 mA loop current.
- Short-Term: Inspect armored sensor cable, M12 connectors, and junction box terminals for moisture or strain damage.
- Corrective: Reseat or replace faulty transducer, clean oxidized terminal blocks, recalibrate transmitter zero/span against reference standard, and verify signal variance restoration.`,
  },
];

interface EpisodeTemplate {
  kind: "spike" | "drift" | "flatline";
  startBase: number;
  duration: number;
  targetSensor: SensorName;
  issue: string;
  action: string;
  downtime: number;
}

const EPISODE_TEMPLATES: EpisodeTemplate[] = [
  {
    kind: "spike",
    startBase: 140,
    duration: 2,
    targetSensor: "temperature",
    issue: "Sudden thermal spike (+31 C)",
    action: "Flushed heat exchanger and cleared coolant fan intake",
    downtime: 2.5,
  },
  {
    kind: "drift",
    startBase: 380,
    duration: 7,
    targetSensor: "vibration",
    issue: "Gradual bearing vibration drift",
    action: "Lubricated bearing housing and inspected shaft alignment",
    downtime: 4.0,
  },
  {
    kind: "flatline",
    startBase: 620,
    duration: 8,
    targetSensor: "pressure",
    issue: "Flat-line pressure transducer fault",
    action: "Replaced corroded 4-20 mA pressure transmitter cable",
    downtime: 1.5,
  },
  {
    kind: "spike",
    startBase: 890,
    duration: 2,
    targetSensor: "vibration",
    issue: "Transient vibration shock spike",
    action: "Torqued loose motor mounting bolts and checked coupling",
    downtime: 2.0,
  },
  {
    kind: "drift",
    startBase: 1180,
    duration: 7,
    targetSensor: "pressure",
    issue: "Progressive hydraulic pressure loss",
    action: "Replaced worn mechanical shaft seal and cleaned intake strainer",
    downtime: 5.0,
  },
  {
    kind: "flatline",
    startBase: 1470,
    duration: 8,
    targetSensor: "temperature",
    issue: "Frozen RTD temperature channel",
    action: "Reseated thermocouple terminal block and recalibrated RTD",
    downtime: 1.5,
  },
  {
    kind: "drift",
    startBase: 1740,
    duration: 7,
    targetSensor: "temperature",
    issue: "Gradual overheating prior to thermal trip",
    action: "Replaced clogged oil filter and replenished synthetic oil",
    downtime: 4.5,
  },
  {
    kind: "spike",
    startBase: 2040,
    duration: 2,
    targetSensor: "rpm",
    issue: "Sudden RPM surge and bearing load spike",
    action: "Recalibrated VFD speed controller and inspected rotor balance",
    downtime: 3.0,
  },
];

/** Deterministic Mulberry32 PRNG. */
export class SeededRandom {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  /** Uniform [0, 1) float. */
  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform [min, max) float. */
  uniform(min: number, max: number): number {
    return min + (max - min) * this.next();
  }

  /** Normal(mean, std) via Box-Muller transform. */
  normal(mean: number, std: number): number {
    const u1 = Math.max(this.next(), 1e-12);
    const u2 = this.next();
    const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + std * z;
  }
}

function roundTo(val: number, decimals: number): number {
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}

function formatUtcTimestamp(startEpochMs: number, hourOffset: number): string {
  const dt = new Date(startEpochMs + hourOffset * 3600 * 1000);
  return dt.toISOString().slice(0, 19);
}

/** Generate deterministic synthetic 90-day hourly telemetry and ~40 maintenance logs. */
export function generateSyntheticDataset(seed: number = SEED): SyntheticDataset {
  const rng = new SeededRandom(seed);
  const startEpochMs = Date.UTC(2026, 0, 1, 0, 0, 0);

  const readings: SensorReading[] = [];
  const logs: MaintenanceLog[] = [];
  const perMachine: MachineSummary[] = [];

  let logCounter = 1;
  let totalAnomalies = 0;

  for (let mIdx = 0; mIdx < MACHINES.length; mIdx += 1) {
    const machineId = MACHINES[mIdx];
    const base = MACHINE_BASELINES[machineId];
    const phaseShift = mIdx * 0.45;

    const temp = new Array<number>(TOTAL_HOURS);
    const vib = new Array<number>(TOTAL_HOURS);
    const press = new Array<number>(TOTAL_HOURS);
    const rpm = new Array<number>(TOTAL_HOURS);
    const isAnomaly = new Array<0 | 1>(TOTAL_HOURS).fill(0);

    for (let h = 0; h < TOTAL_HOURS; h += 1) {
      const dailyPhase = (2.0 * Math.PI * (h % HOURS_PER_DAY)) / HOURS_PER_DAY;
      temp[h] =
        base.tempBase +
        0.0018 * h +
        2.4 * Math.sin(dailyPhase + phaseShift) +
        rng.normal(0.0, 0.75);
      vib[h] =
        base.vibBase +
        0.00018 * h +
        0.18 * Math.sin(dailyPhase + phaseShift + 0.3) +
        rng.normal(0.0, 0.08);
      press[h] =
        base.pressBase -
        0.0012 * h +
        1.6 * Math.cos(dailyPhase + phaseShift) +
        rng.normal(0.0, 0.65);
      rpm[h] =
        base.rpmBase +
        0.012 * h +
        22.0 * Math.sin(dailyPhase + phaseShift + 0.7) +
        rng.normal(0.0, 7.5);
    }

    const machineOffset = mIdx * 17;
    let machineLogs = 0;

    for (const ep of EPISODE_TEMPLATES) {
      const start = ep.startBase + machineOffset;
      const end = Math.min(start + ep.duration, TOTAL_HOURS);
      const len = end - start;

      for (let i = start; i < end; i += 1) {
        isAnomaly[i] = 1;
      }

      if (ep.kind === "spike") {
        if (ep.targetSensor === "temperature") {
          for (let i = start; i < end; i += 1) {
            temp[i] += rng.uniform(26.0, 34.0);
            vib[i] += rng.uniform(0.9, 1.5);
          }
        } else if (ep.targetSensor === "vibration") {
          for (let i = start; i < end; i += 1) {
            vib[i] += rng.uniform(3.8, 5.2);
            temp[i] += rng.uniform(6.0, 10.0);
          }
        } else if (ep.targetSensor === "rpm") {
          for (let i = start; i < end; i += 1) {
            rpm[i] += rng.uniform(360.0, 460.0);
            vib[i] += rng.uniform(2.2, 3.4);
            temp[i] += rng.uniform(8.0, 13.0);
          }
        }
      } else if (ep.kind === "drift") {
        for (let k = 0; k < len; k += 1) {
          const i = start + k;
          const ramp = len > 1 ? 0.25 + (0.75 * k) / (len - 1) : 1.0;
          if (ep.targetSensor === "vibration") {
            vib[i] += 3.8 * ramp;
            temp[i] += 9.5 * ramp;
          } else if (ep.targetSensor === "pressure") {
            press[i] -= 28.0 * ramp;
            rpm[i] += 140.0 * ramp;
          } else if (ep.targetSensor === "temperature") {
            temp[i] += 26.0 * ramp;
            vib[i] += 1.4 * ramp;
          }
        }
      } else if (ep.kind === "flatline") {
        if (ep.targetSensor === "pressure") {
          for (let i = start; i < end; i += 1) {
            press[i] = 0.0;
          }
        } else if (ep.targetSensor === "temperature") {
          const frozenTemp = roundTo(temp[Math.max(0, start - 1)] - 35.0, 2);
          for (let i = start; i < end; i += 1) {
            temp[i] = frozenTemp;
          }
        }
      }

      const logDate = formatUtcTimestamp(startEpochMs, end - 1).slice(0, 10);
      logs.push({
        log_id: `LOG-${String(logCounter).padStart(4, "0")}`,
        machine_id: machineId,
        date: logDate,
        issue: ep.issue,
        action_taken: ep.action,
        downtime_hours: ep.downtime,
      });
      logCounter += 1;
      machineLogs += 1;
    }

    let machineAnomalies = 0;
    for (let h = 0; h < TOTAL_HOURS; h += 1) {
      if (isAnomaly[h] === 1) {
        machineAnomalies += 1;
        totalAnomalies += 1;
      }
      readings.push({
        timestamp: formatUtcTimestamp(startEpochMs, h),
        hourIndex: h,
        machine_id: machineId,
        temperature: roundTo(temp[h], 3),
        vibration: roundTo(Math.max(0, vib[h]), 3),
        pressure: roundTo(Math.max(0, press[h]), 3),
        rpm: roundTo(Math.max(0, rpm[h]), 2),
        is_anomaly: isAnomaly[h],
      });
    }

    perMachine.push({
      machine_id: machineId,
      rows: TOTAL_HOURS,
      anomalies: machineAnomalies,
      anomalyRatePct: roundTo((100.0 * machineAnomalies) / TOTAL_HOURS, 2),
      logs: machineLogs,
    });
  }

  return {
    readings,
    logs,
    stats: {
      seed,
      totalRows: readings.length,
      machinesCount: MACHINES.length,
      hoursPerMachine: TOTAL_HOURS,
      anomalyCount: totalAnomalies,
      anomalyRatePct: roundTo((100.0 * totalAnomalies) / readings.length, 2),
      logsCount: logs.length,
      perMachine,
    },
  };
}
