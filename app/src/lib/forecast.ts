import { HOURS_PER_DAY, type MachineId, type SensorName, type SensorReading } from "./data";

export type ForecastModelKey = "naive_last_value" | "seasonal_naive" | "linear_ar";

export interface ForecastModelMetric {
  model: ForecastModelKey;
  label: string;
  mae: number;
  rmse: number;
  isBest: boolean;
}

export interface WalkForwardPoint {
  timestamp: string;
  actual: number;
  naive_last_value: number;
  seasonal_naive: number;
  linear_ar: number;
}

export interface FutureForecastPoint {
  timestamp: string;
  horizonStep: number;
  naive_last_value: number;
  seasonal_naive: number;
  linear_ar: number;
}

export interface ForecastResult {
  machineId: MachineId;
  sensor: SensorName;
  horizonHours: number;
  trainRows: number;
  testRows: number;
  splitTimestamp: string;
  lastObservedTimestamp: string;
  lastObservedValue: number;
  bestModel: ForecastModelKey;
  metrics: ForecastModelMetric[];
  walkForwardPoints: WalkForwardPoint[];
  futureForecast: FutureForecastPoint[];
}

export const TEST_WINDOW_HOURS = 168;

function roundTo(val: number, decimals: number): number {
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}

export function computeErrorMetrics(
  actuals: number[],
  predictions: number[],
): { mae: number; rmse: number } {
  if (actuals.length === 0 || actuals.length !== predictions.length) {
    throw new Error("Actual and prediction arrays must be non-empty and equal length.");
  }
  let absSum = 0;
  let sqSum = 0;
  for (let i = 0; i < actuals.length; i += 1) {
    const err = actuals[i] - predictions[i];
    absSum += Math.abs(err);
    sqSum += err * err;
  }
  return {
    mae: roundTo(absSum / actuals.length, 4),
    rmse: roundTo(Math.sqrt(sqSum / actuals.length), 4),
  };
}

/**
 * Solve (X^T X + lambda * I) w = X^T y via Gaussian elimination with partial pivoting.
 * Bias term at index 0 is left unpenalized.
 */
export function solveLinearLeastSquares(
  X: number[][],
  y: number[],
  ridgeLambda: number = 1e-4,
): number[] {
  const n = X.length;
  const p = X[0]?.length ?? 0;
  if (n === 0 || p === 0) {
    return [];
  }

  const xtx: number[][] = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  const xty: number[] = new Array<number>(p).fill(0);

  for (let i = 0; i < n; i += 1) {
    const row = X[i];
    const yi = y[i];
    for (let j = 0; j < p; j += 1) {
      xty[j] += row[j] * yi;
      for (let k = j; k < p; k += 1) {
        xtx[j][k] += row[j] * row[k];
      }
    }
  }

  for (let j = 0; j < p; j += 1) {
    for (let k = 0; k < j; k += 1) {
      xtx[j][k] = xtx[k][j];
    }
    if (j > 0) {
      xtx[j][j] += ridgeLambda;
    }
  }

  // Augmented matrix [A | b]
  const aug = xtx.map((row, idx) => [...row, xty[idx]]);

  for (let col = 0; col < p; col += 1) {
    let pivotRow = col;
    let maxVal = Math.abs(aug[col][col]);
    for (let r = col + 1; r < p; r += 1) {
      const candidate = Math.abs(aug[r][col]);
      if (candidate > maxVal) {
        maxVal = candidate;
        pivotRow = r;
      }
    }
    if (pivotRow !== col) {
      const tmp = aug[col];
      aug[col] = aug[pivotRow];
      aug[pivotRow] = tmp;
    }

    const pivot = aug[col][col];
    if (Math.abs(pivot) < 1e-12) {
      aug[col][col] = 1e-12;
    }

    for (let r = col + 1; r < p; r += 1) {
      const factor = aug[r][col] / aug[col][col];
      for (let c = col; c <= p; c += 1) {
        aug[r][c] -= factor * aug[col][c];
      }
    }
  }

  const weights = new Array<number>(p).fill(0);
  for (let r = p - 1; r >= 0; r -= 1) {
    let sum = aug[r][p];
    for (let c = r + 1; c < p; c += 1) {
      sum -= aug[r][c] * weights[c];
    }
    weights[r] = sum / aug[r][r];
  }

  return weights;
}

function buildArFeatures(history: number[], targetIdx: number): number[] {
  const lag1 = history[targetIdx - 1];
  const lag2 = history[targetIdx - 2];
  const lag3 = history[targetIdx - 3];
  const lag24 = history[targetIdx - 24];

  let sum24 = 0;
  for (let j = targetIdx - 24; j < targetIdx; j += 1) {
    sum24 += history[j];
  }
  const rollMean24 = sum24 / 24.0;
  const hourAngle = (2.0 * Math.PI * (targetIdx % HOURS_PER_DAY)) / HOURS_PER_DAY;

  return [
    1.0,
    lag1,
    lag2,
    lag3,
    lag24,
    rollMean24,
    Math.sin(hourAngle),
    Math.cos(hourAngle),
  ];
}

export function fitLinearAR(trainSeries: number[]): number[] {
  const startIdx = 24;
  if (trainSeries.length <= startIdx + 10) {
    return [trainSeries[trainSeries.length - 1] ?? 0, 0, 0, 0, 0, 0, 0, 0];
  }

  const X: number[][] = [];
  const y: number[] = [];
  for (let t = startIdx; t < trainSeries.length; t += 1) {
    X.push(buildArFeatures(trainSeries, t));
    y.push(trainSeries[t]);
  }
  return solveLinearLeastSquares(X, y, 1e-3);
}

function recursiveArForecast(
  observedHistory: number[],
  weights: number[],
  steps: number,
): number[] {
  const buffer = [...observedHistory];
  const preds: number[] = [];

  for (let step = 0; step < steps; step += 1) {
    const idx = buffer.length;
    const feats = buildArFeatures(buffer, idx);
    let yHat = 0;
    for (let j = 0; j < weights.length; j += 1) {
      yHat += weights[j] * feats[j];
    }
    buffer.push(yHat);
    preds.push(yHat);
  }
  return preds;
}

/**
 * Strict chronological train/test split (last 168h held out) with walk-forward
 * multi-step evaluation and next-horizon future forecast.
 */
export function evaluateAndForecast(
  readings: SensorReading[],
  machineId: MachineId,
  sensor: SensorName,
  horizonHours: number = 24,
): ForecastResult {
  const machineReadings = readings
    .filter((r) => r.machine_id === machineId)
    .sort((a, b) => a.hourIndex - b.hourIndex);

  if (machineReadings.length <= TEST_WINDOW_HOURS + 48) {
    throw new Error(`Insufficient history for ${machineId} to split ${TEST_WINDOW_HOURS} test hours.`);
  }

  const series = machineReadings.map((r) => r[sensor]);
  const timestamps = machineReadings.map((r) => r.timestamp);
  const splitIdx = series.length - TEST_WINDOW_HOURS;

  // Train linear AR strictly on train slice [0, splitIdx) -> zero leakage
  const trainSeries = series.slice(0, splitIdx);
  const arWeights = fitLinearAR(trainSeries);

  const actuals: number[] = [];
  const naivePreds: number[] = [];
  const seasonalPreds: number[] = [];
  const arPreds: number[] = [];
  const walkForwardPoints: WalkForwardPoint[] = [];

  // Walk-forward across non-overlapping horizon blocks in the 168h test window
  for (let origin = splitIdx; origin < series.length; origin += horizonHours) {
    const blockSteps = Math.min(horizonHours, series.length - origin);
    const observedUpToOrigin = series.slice(0, origin);
    const lastObs = observedUpToOrigin[observedUpToOrigin.length - 1];
    const arBlock = recursiveArForecast(observedUpToOrigin, arWeights, blockSteps);

    for (let k = 0; k < blockSteps; k += 1) {
      const targetIdx = origin + k;
      const naiveVal = lastObs;
      const seasonalIdx = origin - HOURS_PER_DAY + (k % HOURS_PER_DAY);
      const seasonalVal = observedUpToOrigin[seasonalIdx];
      const arVal = arBlock[k];
      const actualVal = series[targetIdx];

      actuals.push(actualVal);
      naivePreds.push(naiveVal);
      seasonalPreds.push(seasonalVal);
      arPreds.push(arVal);

      walkForwardPoints.push({
        timestamp: timestamps[targetIdx],
        actual: roundTo(actualVal, 3),
        naive_last_value: roundTo(naiveVal, 3),
        seasonal_naive: roundTo(seasonalVal, 3),
        linear_ar: roundTo(arVal, 3),
      });
    }
  }

  const naiveM = computeErrorMetrics(actuals, naivePreds);
  const seasonalM = computeErrorMetrics(actuals, seasonalPreds);
  const arM = computeErrorMetrics(actuals, arPreds);

  const rawMetrics: Array<{ model: ForecastModelKey; label: string; mae: number; rmse: number }> = [
    { model: "naive_last_value", label: "Naive Last-Value", ...naiveM },
    { model: "seasonal_naive", label: "Seasonal Naive (24h Lag)", ...seasonalM },
    { model: "linear_ar", label: "Linear Autoregressive (OLS)", ...arM },
  ];

  const bestModel = rawMetrics.reduce((best, cur) => (cur.rmse < best.rmse ? cur : best)).model;
  const metrics: ForecastModelMetric[] = rawMetrics.map((m) => ({
    ...m,
    isBest: m.model === bestModel,
  }));

  // Next horizonHours future forecast from end of full dataset
  const fullArWeights = fitLinearAR(series);
  const futureAr = recursiveArForecast(series, fullArWeights, horizonHours);
  const lastTimestamp = timestamps[timestamps.length - 1];
  const lastEpochMs = Date.parse(`${lastTimestamp}Z`);
  const lastValue = series[series.length - 1];

  const futureForecast: FutureForecastPoint[] = [];
  for (let k = 0; k < horizonHours; k += 1) {
    const futureTs = new Date(lastEpochMs + (k + 1) * 3600 * 1000).toISOString().slice(0, 19);
    const seasonalVal = series[series.length - HOURS_PER_DAY + (k % HOURS_PER_DAY)];
    futureForecast.push({
      timestamp: futureTs,
      horizonStep: k + 1,
      naive_last_value: roundTo(lastValue, 3),
      seasonal_naive: roundTo(seasonalVal, 3),
      linear_ar: roundTo(futureAr[k], 3),
    });
  }

  return {
    machineId,
    sensor,
    horizonHours,
    trainRows: splitIdx,
    testRows: TEST_WINDOW_HOURS,
    splitTimestamp: timestamps[splitIdx],
    lastObservedTimestamp: lastTimestamp,
    lastObservedValue: roundTo(lastValue, 3),
    bestModel,
    metrics,
    walkForwardPoints,
    futureForecast,
  };
}
