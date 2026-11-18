import { useMemo, useState } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { detectAnomalies } from "../lib/anomaly";
import {
  MACHINES,
  SENSORS,
  type MachineId,
  type SensorName,
  type SensorReading,
} from "../lib/data";

interface AnomaliesTabProps {
  readings: SensorReading[];
  isDark: boolean;
}

export function AnomaliesTab({ readings, isDark }: AnomaliesTabProps) {
  const [machineId, setMachineId] = useState<MachineId>("M1");
  const [sensor, setSensor] = useState<SensorName>("vibration");
  const [windowHours, setWindowHours] = useState<number>(2160);

  const result = useMemo(() => detectAnomalies(readings, machineId), [readings, machineId]);

  const chartData = useMemo(() => {
    const slice = result.points.slice(-windowHours);
    return slice.map((p) => {
      const val = p[sensor];
      return {
        timestamp: p.timestamp,
        value: val,
        zscoreMarker: p.rollingZScoreFlag ? val : null,
        iforestMarker: p.isolationForestFlag ? val : null,
        truthMarker: p.isTrueAnomaly ? val : null,
      };
    });
  }, [result.points, sensor, windowHours]);

  const gridStroke = isDark ? "#1e293b" : "#e2e8f0";
  const axisTick = isDark ? "#94a3b8" : "#64748b";
  const tooltipStyle = {
    backgroundColor: isDark ? "#0f172a" : "#ffffff",
    borderColor: isDark ? "#334155" : "#cbd5e1",
    color: isDark ? "#f1f5f9" : "#0f172a",
    fontSize: "12px",
  };

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
              Unsupervised Anomaly Detection Benchmark
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Compares causal Rolling Z-Score (window 24, threshold 3.2 + flat-line check) and
              64-tree seeded Isolation Forest across {result.rowsScanned.toLocaleString()} hours (
              {result.trueAnomalies} labelled faults).
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <label className="flex flex-col gap-1 text-xs text-slate-600 dark:text-slate-300">
              <span>Machine</span>
              <select
                value={machineId}
                onChange={(e) => setMachineId(e.target.value as MachineId)}
                className="rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-950 px-3 py-1.5 text-sm text-slate-900 dark:text-slate-100"
              >
                {MACHINES.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </label>

            <label className="flex flex-col gap-1 text-xs text-slate-600 dark:text-slate-300">
              <span>Chart Sensor</span>
              <select
                value={sensor}
                onChange={(e) => setSensor(e.target.value as SensorName)}
                className="rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-950 px-3 py-1.5 text-sm text-slate-900 dark:text-slate-100"
              >
                {SENSORS.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </label>

            <label className="flex flex-col gap-1 text-xs text-slate-600 dark:text-slate-300">
              <span>Chart Span</span>
              <select
                value={windowHours}
                onChange={(e) => setWindowHours(Number(e.target.value))}
                className="rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-950 px-3 py-1.5 text-sm text-slate-900 dark:text-slate-100"
              >
                <option value={500}>Last 500 hours</option>
                <option value={1000}>Last 1,000 hours</option>
                <option value={2160}>Full 90 days (2,160h)</option>
              </select>
            </label>
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200 dark:border-slate-800">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Detector Evaluation Against Ground-Truth Labels ({machineId})
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="bg-slate-50 dark:bg-slate-800/60 text-xs text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
              <tr>
                <th className="py-2.5 px-4 font-medium">Detector</th>
                <th className="py-2.5 px-4 font-medium text-right">Precision</th>
                <th className="py-2.5 px-4 font-medium text-right">Recall</th>
                <th className="py-2.5 px-4 font-medium text-right">F1 Score</th>
                <th className="py-2.5 px-4 font-medium text-right">TP / FP / FN</th>
                <th className="py-2.5 px-4 font-medium text-right">Flagged</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
              {result.metrics.map((m) => (
                <tr
                  key={m.detector}
                  className="hover:bg-slate-50 dark:hover:bg-slate-800/40"
                >
                  <td className="py-3 px-4 font-medium text-slate-900 dark:text-slate-100">
                    {m.label}
                  </td>
                  <td className="py-3 px-4 text-right font-mono tabular-nums text-slate-700 dark:text-slate-300">
                    {m.precision.toFixed(4)}
                  </td>
                  <td className="py-3 px-4 text-right font-mono tabular-nums text-slate-700 dark:text-slate-300">
                    {m.recall.toFixed(4)}
                  </td>
                  <td className="py-3 px-4 text-right font-mono tabular-nums font-semibold text-slate-900 dark:text-slate-100">
                    {m.f1.toFixed(4)}
                  </td>
                  <td className="py-3 px-4 text-right font-mono tabular-nums text-slate-600 dark:text-slate-400">
                    {m.tp} / {m.fp} / {m.fn}
                  </td>
                  <td className="py-3 px-4 text-right font-mono tabular-nums text-slate-700 dark:text-slate-300">
                    {m.flaggedCount}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
          {machineId} · {sensor} Telemetry with Detected Anomalies Highlighted
        </h3>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 mb-4">
          Crimson points show Rolling Z-Score / flat-line flags; amber points show Isolation Forest
          flags.
        </p>
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
              <XAxis
                dataKey="timestamp"
                tick={{ fill: axisTick, fontSize: 11 }}
                tickFormatter={(v: string) => v.slice(5, 10)}
                minTickGap={36}
              />
              <YAxis
                tick={{ fill: axisTick, fontSize: 11 }}
                domain={["auto", "auto"]}
                width={52}
              />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              <Line
                type="monotone"
                dataKey="value"
                name={`${sensor} reading`}
                stroke="#0284c7"
                strokeWidth={1.5}
                dot={false}
              />
              <Scatter
                dataKey="iforestMarker"
                name="Isolation Forest Flag"
                fill="#d97706"
              />
              <Scatter
                dataKey="zscoreMarker"
                name="Rolling Z-Score / Flatline Flag"
                fill="#dc2626"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Latest Detections ({result.latestDetections.length})
          </h3>
          <span className="text-xs text-slate-500 dark:text-slate-400">
            Most recent 20 flagged timestamps
          </span>
        </div>
        <div className="overflow-x-auto max-h-96">
          <table className="w-full text-xs text-left">
            <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800/95 text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
              <tr>
                <th className="py-2.5 px-4 font-medium">Timestamp</th>
                <th className="py-2.5 px-4 font-medium text-right">Temp</th>
                <th className="py-2.5 px-4 font-medium text-right">Vib</th>
                <th className="py-2.5 px-4 font-medium text-right">Press</th>
                <th className="py-2.5 px-4 font-medium text-right">Max Z</th>
                <th className="py-2.5 px-4 font-medium text-right">IF Score</th>
                <th className="py-2.5 px-4 font-medium">Reason</th>
                <th className="py-2.5 px-4 font-medium text-right">Ground Truth</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800 font-mono tabular-nums">
              {result.latestDetections.map((det) => (
                <tr
                  key={det.timestamp}
                  className="hover:bg-slate-50 dark:hover:bg-slate-800/40"
                >
                  <td className="py-2 px-4 text-slate-800 dark:text-slate-200 whitespace-nowrap">
                    {det.timestamp}
                  </td>
                  <td className="py-2 px-4 text-right text-slate-700 dark:text-slate-300">
                    {det.temperature.toFixed(1)}
                  </td>
                  <td className="py-2 px-4 text-right text-slate-700 dark:text-slate-300">
                    {det.vibration.toFixed(2)}
                  </td>
                  <td className="py-2 px-4 text-right text-slate-700 dark:text-slate-300">
                    {det.pressure.toFixed(1)}
                  </td>
                  <td className="py-2 px-4 text-right text-slate-700 dark:text-slate-300">
                    {det.maxZScore.toFixed(2)}
                  </td>
                  <td className="py-2 px-4 text-right text-slate-700 dark:text-slate-300">
                    {det.isolationScore.toFixed(3)}
                  </td>
                  <td className="py-2 px-4 font-sans text-slate-700 dark:text-slate-300">
                    {det.reason}
                  </td>
                  <td
                    className={`py-2 px-4 text-right font-semibold whitespace-nowrap ${
                      det.isTrueAnomaly
                        ? "text-emerald-600 dark:text-emerald-400"
                        : "text-amber-600 dark:text-amber-400"
                    }`}
                  >
                    {det.isTrueAnomaly ? "TRUE POSITIVE" : "FALSE ALARM"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
