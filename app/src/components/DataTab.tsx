import { useMemo, useState } from "react";
import { AlertTriangle, Database, RefreshCw, Table2 } from "lucide-react";

import {
  MACHINES,
  type MachineId,
  type SyntheticDataset,
} from "../lib/data";

interface DataTabProps {
  dataset: SyntheticDataset;
  onRegenerate: () => void;
}

export function DataTab({ dataset, onRegenerate }: DataTabProps) {
  const [machineFilter, setMachineFilter] = useState<"ALL" | MachineId>("ALL");
  const [anomaliesOnly, setAnomaliesOnly] = useState(false);
  const [viewMode, setViewMode] = useState<"readings" | "logs">("readings");

  const filteredReadings = useMemo(() => {
    const base =
      machineFilter === "ALL"
        ? dataset.readings
        : dataset.readings.filter((r) => r.machine_id === machineFilter);
    const filtered = anomaliesOnly ? base.filter((r) => r.is_anomaly === 1) : base;
    return filtered.slice(0, 120);
  }, [dataset.readings, machineFilter, anomaliesOnly]);

  const filteredLogs = useMemo(() => {
    const base =
      machineFilter === "ALL"
        ? dataset.logs
        : dataset.logs.filter((l) => l.machine_id === machineFilter);
    return [...base].sort((a, b) => b.date.localeCompare(a.date));
  }, [dataset.logs, machineFilter]);

  const { stats } = dataset;

  return (
    <div className="space-y-6">
      <div
        role="note"
        className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 rounded-xl border border-amber-300 dark:border-amber-800/80 bg-amber-50/70 dark:bg-amber-950/30 px-5 py-4"
      >
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-amber-950 dark:text-amber-200">
              SYNTHETIC DATA — Deterministic Browser Simulation (Seed {stats.seed})
            </p>
            <p className="text-sm text-amber-900/85 dark:text-amber-300/85 mt-0.5">
              Generated live in TypeScript across {stats.machinesCount} machines (M1–M5),{" "}
              {stats.hoursPerMachine.toLocaleString()} hourly readings each (90 days), with
              injected spikes, gradual drifts, and flat-line sensor faults.
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={onRegenerate}
          className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg border border-amber-400/80 dark:border-amber-700 bg-white dark:bg-slate-900 text-amber-950 dark:text-amber-200 hover:bg-amber-100/60 dark:hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Regenerate Dataset
        </button>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
          <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
            Total Sensor Rows
          </p>
          <p className="mt-1 text-2xl font-semibold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {stats.totalRows.toLocaleString()}
          </p>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            {stats.machinesCount} machines · 90 days hourly
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
          <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
            Labelled Anomalies
          </p>
          <p className="mt-1 text-2xl font-semibold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {stats.anomalyCount.toLocaleString()}
          </p>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            Spikes · Drifts · Flat-lines
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
          <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
            Overall Anomaly Rate
          </p>
          <p className="mt-1 text-2xl font-semibold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {stats.anomalyRatePct.toFixed(2)}%
          </p>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            Target ~2.0% injected faults
          </p>
        </div>

        <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
          <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
            Maintenance Logs
          </p>
          <p className="mt-1 text-2xl font-semibold font-mono tabular-nums text-slate-900 dark:text-slate-100">
            {stats.logsCount}
          </p>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            8 episodes per machine
          </p>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            Per-Machine Synthetic Breakdown
          </h2>
          <span className="text-xs text-slate-500 dark:text-slate-400">
            4 channels: temperature · vibration · pressure · rpm
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="bg-slate-50 dark:bg-slate-800/60 text-xs text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
              <tr>
                <th className="py-2.5 px-4 font-medium">Machine</th>
                <th className="py-2.5 px-4 font-medium text-right">Hourly Rows</th>
                <th className="py-2.5 px-4 font-medium text-right">Anomalies</th>
                <th className="py-2.5 px-4 font-medium text-right">Anomaly Rate</th>
                <th className="py-2.5 px-4 font-medium text-right">Maintenance Logs</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800 font-mono tabular-nums">
              {stats.perMachine.map((m) => (
                <tr
                  key={m.machine_id}
                  className="hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors"
                >
                  <td className="py-2.5 px-4 font-semibold text-slate-900 dark:text-slate-100">
                    {m.machine_id}
                  </td>
                  <td className="py-2.5 px-4 text-right text-slate-700 dark:text-slate-300">
                    {m.rows.toLocaleString()}
                  </td>
                  <td className="py-2.5 px-4 text-right text-slate-700 dark:text-slate-300">
                    {m.anomalies}
                  </td>
                  <td className="py-2.5 px-4 text-right text-slate-700 dark:text-slate-300">
                    {m.anomalyRatePct.toFixed(2)}%
                  </td>
                  <td className="py-2.5 px-4 text-right text-slate-700 dark:text-slate-300">
                    {m.logs}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-1 p-1 bg-slate-100 dark:bg-slate-800 rounded-lg">
            <button
              type="button"
              onClick={() => setViewMode("readings")}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                viewMode === "readings"
                  ? "bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 shadow-sm"
                  : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100"
              }`}
            >
              <Table2 className="w-3.5 h-3.5" />
              Sensor Readings Preview
            </button>
            <button
              type="button"
              onClick={() => setViewMode("logs")}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                viewMode === "logs"
                  ? "bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 shadow-sm"
                  : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100"
              }`}
            >
              <Database className="w-3.5 h-3.5" />
              Maintenance Logs ({filteredLogs.length})
            </button>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <label className="inline-flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300">
              <span>Machine</span>
              <select
                value={machineFilter}
                onChange={(e) => setMachineFilter(e.target.value as "ALL" | MachineId)}
                className="rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-950 px-2.5 py-1.5 text-xs text-slate-900 dark:text-slate-100"
              >
                <option value="ALL">All (M1–M5)</option>
                {MACHINES.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </label>

            {viewMode === "readings" && (
              <label className="inline-flex items-center gap-2 text-xs text-slate-600 dark:text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={anomaliesOnly}
                  onChange={(e) => setAnomaliesOnly(e.target.checked)}
                  className="rounded border-slate-300 dark:border-slate-700 accent-sky-600"
                />
                <span>Anomalies only</span>
              </label>
            )}
          </div>
        </div>

        {viewMode === "readings" ? (
          <div className="overflow-x-auto max-h-[420px]">
            <table className="w-full text-xs text-left">
              <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800/95 text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
                <tr>
                  <th className="py-2.5 px-4 font-medium">Timestamp</th>
                  <th className="py-2.5 px-4 font-medium">Machine</th>
                  <th className="py-2.5 px-4 font-medium text-right">Temp (C)</th>
                  <th className="py-2.5 px-4 font-medium text-right">Vibration (mm/s)</th>
                  <th className="py-2.5 px-4 font-medium text-right">Pressure (psi)</th>
                  <th className="py-2.5 px-4 font-medium text-right">RPM</th>
                  <th className="py-2.5 px-4 font-medium text-right">Label</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800 font-mono tabular-nums">
                {filteredReadings.map((row) => (
                  <tr
                    key={`${row.machine_id}-${row.hourIndex}`}
                    className={
                      row.is_anomaly === 1
                        ? "bg-rose-50/70 dark:bg-rose-950/25"
                        : "hover:bg-slate-50 dark:hover:bg-slate-800/40"
                    }
                  >
                    <td className="py-2 px-4 text-slate-700 dark:text-slate-300">{row.timestamp}</td>
                    <td className="py-2 px-4 font-semibold text-slate-900 dark:text-slate-100">
                      {row.machine_id}
                    </td>
                    <td className="py-2 px-4 text-right text-slate-700 dark:text-slate-300">
                      {row.temperature.toFixed(2)}
                    </td>
                    <td className="py-2 px-4 text-right text-slate-700 dark:text-slate-300">
                      {row.vibration.toFixed(3)}
                    </td>
                    <td className="py-2 px-4 text-right text-slate-700 dark:text-slate-300">
                      {row.pressure.toFixed(2)}
                    </td>
                    <td className="py-2 px-4 text-right text-slate-700 dark:text-slate-300">
                      {row.rpm.toFixed(1)}
                    </td>
                    <td
                      className={`py-2 px-4 text-right font-semibold ${
                        row.is_anomaly === 1
                          ? "text-rose-600 dark:text-rose-400"
                          : "text-slate-500 dark:text-slate-400"
                      }`}
                    >
                      {row.is_anomaly === 1 ? "ANOMALY (1)" : "NORMAL (0)"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="overflow-x-auto max-h-[420px]">
            <table className="w-full text-xs text-left">
              <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800/95 text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
                <tr>
                  <th className="py-2.5 px-4 font-medium">Log ID</th>
                  <th className="py-2.5 px-4 font-medium">Machine</th>
                  <th className="py-2.5 px-4 font-medium">Date</th>
                  <th className="py-2.5 px-4 font-medium">Issue</th>
                  <th className="py-2.5 px-4 font-medium">Action Taken</th>
                  <th className="py-2.5 px-4 font-medium text-right">Downtime (h)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                {filteredLogs.map((log) => (
                  <tr
                    key={log.log_id}
                    className="hover:bg-slate-50 dark:hover:bg-slate-800/40"
                  >
                    <td className="py-2.5 px-4 font-mono font-semibold text-slate-900 dark:text-slate-100 whitespace-nowrap">
                      {log.log_id}
                    </td>
                    <td className="py-2.5 px-4 font-mono text-slate-700 dark:text-slate-300">
                      {log.machine_id}
                    </td>
                    <td className="py-2.5 px-4 font-mono text-slate-600 dark:text-slate-400 whitespace-nowrap">
                      {log.date}
                    </td>
                    <td className="py-2.5 px-4 text-slate-900 dark:text-slate-100">{log.issue}</td>
                    <td className="py-2.5 px-4 text-slate-600 dark:text-slate-300">
                      {log.action_taken}
                    </td>
                    <td className="py-2.5 px-4 text-right font-mono tabular-nums text-slate-700 dark:text-slate-300">
                      {log.downtime_hours.toFixed(1)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
