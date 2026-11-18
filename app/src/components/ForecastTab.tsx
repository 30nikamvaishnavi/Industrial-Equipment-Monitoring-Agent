import { useMemo, useState } from "react";
import { CheckCircle2 } from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  MACHINES,
  SENSORS,
  type MachineId,
  type SensorName,
  type SensorReading,
} from "../lib/data";
import { evaluateAndForecast } from "../lib/forecast";

interface ForecastTabProps {
  readings: SensorReading[];
  isDark: boolean;
}

const SENSOR_UNITS: Record<SensorName, string> = {
  temperature: "C",
  vibration: "mm/s",
  pressure: "psi",
  rpm: "RPM",
};

export function ForecastTab({ readings, isDark }: ForecastTabProps) {
  const [machineId, setMachineId] = useState<MachineId>("M1");
  const [sensor, setSensor] = useState<SensorName>("temperature");
  const [horizonHours, setHorizonHours] = useState<number>(24);

  const result = useMemo(
    () => evaluateAndForecast(readings, machineId, sensor, horizonHours),
    [readings, machineId, sensor, horizonHours],
  );

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
              Walk-Forward Sensor Forecasting
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Strict chronological split: {result.trainRows.toLocaleString()} training hours followed
              by {result.testRows} test hours (split at {result.splitTimestamp}). Zero future leakage.
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
              <span>Sensor</span>
              <select
                value={sensor}
                onChange={(e) => setSensor(e.target.value as SensorName)}
                className="rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-950 px-3 py-1.5 text-sm text-slate-900 dark:text-slate-100"
              >
                {SENSORS.map((s) => (
                  <option key={s} value={s}>
                    {s} ({SENSOR_UNITS[s]})
                  </option>
                ))}
              </select>
            </label>

            <label className="flex flex-col gap-1 text-xs text-slate-600 dark:text-slate-300">
              <span>Horizon</span>
              <select
                value={horizonHours}
                onChange={(e) => setHorizonHours(Number(e.target.value))}
                className="rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-950 px-3 py-1.5 text-sm text-slate-900 dark:text-slate-100"
              >
                <option value={12}>12 hours</option>
                <option value={24}>24 hours</option>
                <option value={48}>48 hours</option>
              </select>
            </label>
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Test Set Model Comparison ({result.testRows}-Hour Walk-Forward Evaluation)
          </h3>
          <span className="text-xs font-mono text-slate-500 dark:text-slate-400">
            Last observed: {result.lastObservedValue} {SENSOR_UNITS[sensor]}
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="bg-slate-50 dark:bg-slate-800/60 text-xs text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
              <tr>
                <th className="py-2.5 px-5 font-medium">Model</th>
                <th className="py-2.5 px-5 font-medium text-right">MAE</th>
                <th className="py-2.5 px-5 font-medium text-right">RMSE</th>
                <th className="py-2.5 px-5 font-medium text-right">Selection</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
              {result.metrics.map((m) => (
                <tr
                  key={m.model}
                  className={
                    m.isBest
                      ? "bg-emerald-50/60 dark:bg-emerald-950/20"
                      : "hover:bg-slate-50 dark:hover:bg-slate-800/40"
                  }
                >
                  <td className="py-3 px-5 font-medium text-slate-900 dark:text-slate-100">
                    {m.label}
                  </td>
                  <td className="py-3 px-5 text-right font-mono tabular-nums text-slate-700 dark:text-slate-300">
                    {m.mae.toFixed(4)}
                  </td>
                  <td className="py-3 px-5 text-right font-mono tabular-nums text-slate-700 dark:text-slate-300">
                    {m.rmse.toFixed(4)}
                  </td>
                  <td className="py-3 px-5 text-right">
                    {m.isBest ? (
                      <span className="inline-flex items-center gap-1.5 text-xs font-semibold text-emerald-700 dark:text-emerald-400">
                        <CheckCircle2 className="w-4 h-4" />
                        Best Model (Lowest RMSE)
                      </span>
                    ) : (
                      <span className="text-xs text-slate-500 dark:text-slate-400">Baseline</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Holdout Walk-Forward Forecast vs. Actual (Last {result.testRows}h)
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 mb-4">
            {machineId} · {sensor} ({SENSOR_UNITS[sensor]}) · {horizonHours}h recursive blocks
          </p>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={result.walkForwardPoints}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                <XAxis
                  dataKey="timestamp"
                  tick={{ fill: axisTick, fontSize: 11 }}
                  tickFormatter={(v: string) => v.slice(5, 13)}
                  minTickGap={28}
                />
                <YAxis
                  tick={{ fill: axisTick, fontSize: 11 }}
                  domain={["auto", "auto"]}
                  width={54}
                />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Line
                  type="monotone"
                  dataKey="actual"
                  name="Actual"
                  stroke={isDark ? "#f8fafc" : "#0f172a"}
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="naive_last_value"
                  name="Naive Last-Value"
                  stroke="#94a3b8"
                  strokeDasharray="4 4"
                  strokeWidth={1.5}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="seasonal_naive"
                  name="Seasonal Naive (24h)"
                  stroke="#d97706"
                  strokeWidth={1.5}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="linear_ar"
                  name="Linear AR (OLS)"
                  stroke="#0284c7"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Next {horizonHours}-Hour Future Forecast
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 mb-4">
            Forward predictions after {result.lastObservedTimestamp}
          </p>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={result.futureForecast}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                <XAxis
                  dataKey="timestamp"
                  tick={{ fill: axisTick, fontSize: 11 }}
                  tickFormatter={(v: string) => v.slice(11, 16)}
                  minTickGap={20}
                />
                <YAxis
                  tick={{ fill: axisTick, fontSize: 11 }}
                  domain={["auto", "auto"]}
                  width={54}
                />
                <Tooltip contentStyle={tooltipStyle} />
                <Legend wrapperStyle={{ fontSize: "12px" }} />
                <Line
                  type="monotone"
                  dataKey="naive_last_value"
                  name="Naive Last-Value"
                  stroke="#94a3b8"
                  strokeDasharray="4 4"
                  strokeWidth={1.5}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="seasonal_naive"
                  name="Seasonal Naive (24h)"
                  stroke="#d97706"
                  strokeWidth={1.75}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="linear_ar"
                  name="Linear AR (OLS)"
                  stroke="#0284c7"
                  strokeWidth={2.25}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
