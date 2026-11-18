import {
  SEED,
  SENSORS,
  SeededRandom,
  type MachineId,
  type SensorName,
  type SensorReading,
} from "./data";

export type DetectorKey = "rolling_zscore" | "isolation_forest";

export interface DetectorMetric {
  detector: DetectorKey;
  label: string;
  precision: number;
  recall: number;
  f1: number;
  tp: number;
  fp: number;
  fn: number;
  flaggedCount: number;
}

export interface AnomalyPoint {
  timestamp: string;
  hourIndex: number;
  temperature: number;
  vibration: number;
  pressure: number;
  rpm: number;
  isTrueAnomaly: boolean;
  rollingZScoreFlag: boolean;
  isolationForestFlag: boolean;
  maxZScore: number;
  isolationScore: number;
  affectedSensors: SensorName[];
  reason: string;
}

export interface AnomalyDetectionResult {
  machineId: MachineId;
  rowsScanned: number;
  trueAnomalies: number;
  metrics: DetectorMetric[];
  points: AnomalyPoint[];
  latestDetections: AnomalyPoint[];
}

function roundTo(val: number, decimals: number): number {
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}

export function computeClassificationMetrics(
  truth: boolean[],
  predicted: boolean[],
): { precision: number; recall: number; f1: number; tp: number; fp: number; fn: number } {
  let tp = 0;
  let fp = 0;
  let fn = 0;

  for (let i = 0; i < truth.length; i += 1) {
    if (predicted[i] && truth[i]) {
      tp += 1;
    } else if (predicted[i] && !truth[i]) {
      fp += 1;
    } else if (!predicted[i] && truth[i]) {
      fn += 1;
    }
  }

  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  return {
    precision: roundTo(precision, 4),
    recall: roundTo(recall, 4),
    f1: roundTo(f1, 4),
    tp,
    fp,
    fn,
  };
}

interface IsolationNode {
  splitFeature: number;
  splitValue: number;
  left: IsolationNode | null;
  right: IsolationNode | null;
  size: number;
}

function cFactor(n: number): number {
  if (n <= 1) return 0;
  if (n === 2) return 1;
  return 2 * (Math.log(n - 1) + 0.5772156649) - (2 * (n - 1)) / n;
}

function buildIsolationTree(
  data: number[][],
  indices: number[],
  depth: number,
  maxDepth: number,
  rng: SeededRandom,
): IsolationNode {
  if (depth >= maxDepth || indices.length <= 1) {
    return { splitFeature: -1, splitValue: 0, left: null, right: null, size: indices.length };
  }

  const numFeatures = data[0].length;
  const featureOrder = Array.from({ length: numFeatures }, (_, i) => i);
  for (let i = featureOrder.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng.next() * (i + 1));
    const tmp = featureOrder[i];
    featureOrder[i] = featureOrder[j];
    featureOrder[j] = tmp;
  }

  for (const feat of featureOrder) {
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (const idx of indices) {
      const v = data[idx][feat];
      if (v < minVal) minVal = v;
      if (v > maxVal) maxVal = v;
    }
    if (maxVal - minVal > 1e-12) {
      const splitValue = rng.uniform(minVal, maxVal);
      const leftIndices: number[] = [];
      const rightIndices: number[] = [];
      for (const idx of indices) {
        if (data[idx][feat] < splitValue) {
          leftIndices.push(idx);
        } else {
          rightIndices.push(idx);
        }
      }
      if (leftIndices.length > 0 && rightIndices.length > 0) {
        return {
          splitFeature: feat,
          splitValue,
          left: buildIsolationTree(data, leftIndices, depth + 1, maxDepth, rng),
          right: buildIsolationTree(data, rightIndices, depth + 1, maxDepth, rng),
          size: indices.length,
        };
      }
    }
  }

  return { splitFeature: -1, splitValue: 0, left: null, right: null, size: indices.length };
}

function pathLength(point: number[], node: IsolationNode, depth: number): number {
  if (node.left === null || node.right === null || node.splitFeature === -1) {
    return depth + cFactor(node.size);
  }
  if (point[node.splitFeature] < node.splitValue) {
    return pathLength(point, node.left, depth + 1);
  }
  return pathLength(point, node.right, depth + 1);
}

/**
 * Evaluate Rolling Z-Score (window=24, threshold=3.2 + flatline check) and
 * seeded Isolation-Forest-style detector on a machine's full telemetry.
 */
export function detectAnomalies(
  readings: SensorReading[],
  machineId: MachineId,
  windowSize: number = 24,
  zThreshold: number = 3.2,
  seed: number = SEED,
): AnomalyDetectionResult {
  const rows = readings
    .filter((r) => r.machine_id === machineId)
    .sort((a, b) => a.hourIndex - b.hourIndex);

  const n = rows.length;
  const truth = rows.map((r) => r.is_anomaly === 1);

  // 1. Causal Rolling Z-Score + Flat-Line Detector
  const zFlags = new Array<boolean>(n).fill(false);
  const maxZScores = new Array<number>(n).fill(0);
  const affectedList: SensorName[][] = Array.from({ length: n }, () => []);
  const reasons: string[] = new Array<string>(n).fill("");

  for (let i = 0; i < n; i += 1) {
    const affected = new Set<SensorName>();
    const rowReasons: string[] = [];
    let rowMaxZ = 0;

    for (const sensor of SENSORS) {
      const val = rows[i][sensor];

      // Flat-line check: 4 consecutive identical values or 0.0 pressure dropout
      if (sensor === "pressure" && val === 0.0) {
        affected.add(sensor);
        rowReasons.push(`${sensor} zero dropout`);
      } else if (i >= 3) {
        const isFlat =
          Math.abs(rows[i][sensor] - rows[i - 1][sensor]) < 1e-9 &&
          Math.abs(rows[i - 1][sensor] - rows[i - 2][sensor]) < 1e-9 &&
          Math.abs(rows[i - 2][sensor] - rows[i - 3][sensor]) < 1e-9;
        if (isFlat) {
          affected.add(sensor);
          rowReasons.push(`${sensor} flat-line`);
        }
      }

      // Causal trailing window [max(0, i - windowSize), i)
      const wStart = Math.max(0, i - windowSize);
      const count = i - wStart;
      if (count >= 12) {
        let sum = 0;
        for (let j = wStart; j < i; j += 1) {
          sum += rows[j][sensor];
        }
        const mean = sum / count;
        let varSum = 0;
        for (let j = wStart; j < i; j += 1) {
          const d = rows[j][sensor] - mean;
          varSum += d * d;
        }
        const std = Math.sqrt(varSum / count);
        if (std > 1e-6) {
          const z = Math.abs(val - mean) / std;
          if (z > rowMaxZ) {
            rowMaxZ = z;
          }
          if (z > zThreshold) {
            affected.add(sensor);
            rowReasons.push(`${sensor} z=${z.toFixed(2)}`);
          }
        }
      }
    }

    maxZScores[i] = roundTo(rowMaxZ, 3);
    if (affected.size > 0) {
      zFlags[i] = true;
      affectedList[i] = Array.from(affected);
      reasons[i] = rowReasons.join("; ");
    }
  }

  // 2. Seeded Isolation-Forest-Style Detector on multi-sensor + delta + local-std features
  const rawFeatures: number[][] = [];
  for (let i = 0; i < n; i += 1) {
    const prev = rows[Math.max(0, i - 1)];
    const wStart = Math.max(0, i - 5);
    let tMin = Infinity;
    let tMax = -Infinity;
    let pMin = Infinity;
    let pMax = -Infinity;
    for (let j = wStart; j <= i; j += 1) {
      if (rows[j].temperature < tMin) tMin = rows[j].temperature;
      if (rows[j].temperature > tMax) tMax = rows[j].temperature;
      if (rows[j].pressure < pMin) pMin = rows[j].pressure;
      if (rows[j].pressure > pMax) pMax = rows[j].pressure;
    }

    rawFeatures.push([
      rows[i].temperature,
      rows[i].vibration,
      rows[i].pressure,
      rows[i].rpm,
      rows[i].temperature - prev.temperature,
      rows[i].vibration - prev.vibration,
      rows[i].pressure - prev.pressure,
      tMax - tMin,
      pMax - pMin,
    ]);
  }

  // Standardize features
  const numDims = rawFeatures[0]?.length ?? 0;
  const featMeans = new Array<number>(numDims).fill(0);
  const featStds = new Array<number>(numDims).fill(1);
  for (let d = 0; d < numDims; d += 1) {
    let s = 0;
    for (let i = 0; i < n; i += 1) s += rawFeatures[i][d];
    featMeans[d] = s / n;
    let vs = 0;
    for (let i = 0; i < n; i += 1) {
      const diff = rawFeatures[i][d] - featMeans[d];
      vs += diff * diff;
    }
    featStds[d] = Math.max(Math.sqrt(vs / n), 1e-6);
  }
  const scaled = rawFeatures.map((row) =>
    row.map((val, d) => (val - featMeans[d]) / featStds[d]),
  );

  const rng = new SeededRandom(seed);
  const numTrees = 64;
  const sampleSize = Math.min(256, n);
  const maxDepth = Math.ceil(Math.log2(sampleSize));
  const trees: IsolationNode[] = [];

  for (let t = 0; t < numTrees; t += 1) {
    const pool = Array.from({ length: n }, (_, i) => i);
    for (let i = 0; i < sampleSize; i += 1) {
      const j = i + Math.floor(rng.next() * (n - i));
      const tmp = pool[i];
      pool[i] = pool[j];
      pool[j] = tmp;
    }
    const sub = pool.slice(0, sampleSize);
    trees.push(buildIsolationTree(scaled, sub, 0, maxDepth, rng));
  }

  const cNorm = cFactor(sampleSize);
  const isoScores = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i += 1) {
    let depthSum = 0;
    for (let t = 0; t < numTrees; t += 1) {
      depthSum += pathLength(scaled[i], trees[t], 0);
    }
    const avgPath = depthSum / numTrees;
    isoScores[i] = roundTo(Math.pow(2, -avgPath / Math.max(cNorm, 1e-6)), 4);
  }

  // Flag top ~2.5% anomalous scores (matching expected contamination rate)
  const sortedScores = [...isoScores].sort((a, b) => a - b);
  const cutoffIdx = Math.min(n - 1, Math.floor(n * 0.975));
  const scoreCutoff = sortedScores[cutoffIdx];
  const ifFlags = isoScores.map((s) => s >= scoreCutoff);

  const zMetrics = computeClassificationMetrics(truth, zFlags);
  const ifMetrics = computeClassificationMetrics(truth, ifFlags);

  const metrics: DetectorMetric[] = [
    {
      detector: "rolling_zscore",
      label: "Rolling Z-Score (w=24, z>3.2 + Flatline)",
      ...zMetrics,
      flaggedCount: zFlags.filter(Boolean).length,
    },
    {
      detector: "isolation_forest",
      label: "Isolation Forest (64 Seeded Trees)",
      ...ifMetrics,
      flaggedCount: ifFlags.filter(Boolean).length,
    },
  ];

  const points: AnomalyPoint[] = rows.map((r, i) => ({
    timestamp: r.timestamp,
    hourIndex: r.hourIndex,
    temperature: r.temperature,
    vibration: r.vibration,
    pressure: r.pressure,
    rpm: r.rpm,
    isTrueAnomaly: r.is_anomaly === 1,
    rollingZScoreFlag: zFlags[i],
    isolationForestFlag: ifFlags[i],
    maxZScore: maxZScores[i],
    isolationScore: isoScores[i],
    affectedSensors: affectedList[i],
    reason: reasons[i] || (ifFlags[i] ? `Isolation score ${isoScores[i]}` : ""),
  }));

  const latestDetections = points
    .filter((p) => p.rollingZScoreFlag || p.isolationForestFlag)
    .slice(-20)
    .reverse();

  return {
    machineId,
    rowsScanned: n,
    trueAnomalies: truth.filter(Boolean).length,
    metrics,
    points,
    latestDetections,
  };
}
