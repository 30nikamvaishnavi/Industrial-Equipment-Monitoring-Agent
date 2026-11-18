import { useMemo } from "react";
import { Download } from "lucide-react";

import { detectAnomalies, type DetectorKey } from "../lib/anomaly";
import { MACHINES, SENSORS, type SensorReading } from "../lib/data";
import { evaluateAndForecast, type ForecastModelKey } from "../lib/forecast";
import { type BrowserTestSuiteResult } from "../lib/tests";

interface ResultsTabProps {
  readings: SensorReading[];
  testResults: BrowserTestSuiteResult | null;
  onRunTests: () => void;
}

export function ResultsTab({ readings, testResults, onRunTests }: ResultsTabProps) {
  const summary = useMemo(() => {
    const forecastAccumulator: Record<
      ForecastModelKey,
      { label: string; maes: number[]; rmses: number[]; wins: number }
    > = {
      naive_last_value: { label: "Naive Last-Value", maes: [], rmses: [], wins: 0 },
      seasonal_naive: { label: "Seasonal Naive (24h Lag)", maes: [], rmses: [], wins: 0 },
      linear_ar: { label: "Linear Autoregressive (OLS)", maes: [], rmses: [], wins: 0 },
    };

    for (const m of MACHINES) {
      for (const s of SENSORS) {
        const res = evaluateAndForecast(readings, m, s, 24);
        for (const metric of res.metrics) {
          forecastAccumulator[metric.model].maes.push(metric.mae);
          forecastAccumulator[metric.model].rmses.push(metric.rmse);
          if (metric.isBest) {
            forecastAccumulator[metric.model].wins += 1;
          }
        }
      }
    }

    const forecastAggregate = (Object.keys(forecastAccumulator) as ForecastModelKey[]).map(
      (key) => {
        const entry = forecastAccumulator[key];
        const meanMae = entry.maes.reduce((a, b) => a + b, 0) / entry.maes.length;
        const meanRmse = entry.rmses.reduce((a, b) => a + b, 0) / entry.rmses.length;
        return {
          model: key,
          label: entry.label,
          seriesCount: entry.maes.length,
          meanMae: Math.round(meanMae * 10000) / 10000,
          meanRmse: Math.round(meanRmse * 10000) / 10000,
          wins: entry.wins,
        };
      },
    );

    const anomalyAccumulator: Record<
      DetectorKey,
      { label: string; p: number[]; r: number[]; f1: number[] }
    > = {
      rolling_zscore: { label: "Rolling Z-Score (w=24, z>3.2 + Flatline)", p: [], r: [], f1: [] },
      isolation_forest: { label: "Isolation Forest (64 Seeded Trees)", p: [], r: [], f1: [] },
    };

    for (const m of MACHINES) {
      const res = detectAnomalies(readings, m);
      for (const metric of res.metrics) {
        anomalyAccumulator[metric.detector].p.push(metric.precision);
        anomalyAccumulator[metric.detector].r.push(metric.recall);
        anomalyAccumulator[metric.detector].f1.push(metric.f1);
      }
    }

    const anomalyAggregate = (Object.keys(anomalyAccumulator) as DetectorKey[]).map((key) => {
      const entry = anomalyAccumulator[key];
      const avg = (arr: number[]) =>
        Math.round((arr.reduce((a, b) => a + b, 0) / arr.length) * 10000) / 10000;
      return {
        detector: key,
        label: entry.label,
        machinesEvaluated: entry.f1.length,
        meanPrecision: avg(entry.p),
        meanRecall: avg(entry.r),
        meanF1: avg(entry.f1),
      };
    });

    return {
      generatedAt: new Date().toISOString(),
      forecastBenchmark: forecastAggregate,
      anomalyBenchmark: anomalyAggregate,
    };
  }, [readings]);

  const handleDownload = () => {
    const payload = {
      ...summary,
      inBrowserTests: testResults ?? "not_run_yet",
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "results.json";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
            Live Benchmark & Test Results Summary
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Aggregated live across all 5 machines × 4 sensors (20 walk-forward forecasting series
            and 5 machine anomaly scans). Nothing is hardcoded.
          </p>
        </div>

        <button
          type="button"
          onClick={handleDownload}
          className="inline-flex items-center gap-2 px-4 py-2 text-sm font-semibold rounded-lg bg-sky-600 hover:bg-sky-700 text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <Download className="w-4 h-4" />
          Download results.json
        </button>
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200 dark:border-slate-800">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            1. Forecasting Aggregate Benchmark (20 Series · 24h Horizon)
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="bg-slate-50 dark:bg-slate-800/60 text-xs text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
              <tr>
                <th className="py-2.5 px-5 font-medium">Model</th>
                <th className="py-2.5 px-5 font-medium text-right">Mean MAE</th>
                <th className="py-2.5 px-5 font-medium text-right">Mean RMSE</th>
                <th className="py-2.5 px-5 font-medium text-right">Series Won</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
              {summary.forecastBenchmark.map((row) => (
                <tr key={row.model} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                  <td className="py-3 px-5 font-medium text-slate-900 dark:text-slate-100">
                    {row.label}
                  </td>
                  <td className="py-3 px-5 text-right font-mono tabular-nums text-slate-700 dark:text-slate-300">
                    {row.meanMae.toFixed(4)}
                  </td>
                  <td className="py-3 px-5 text-right font-mono tabular-nums text-slate-700 dark:text-slate-300">
                    {row.meanRmse.toFixed(4)}
                  </td>
                  <td className="py-3 px-5 text-right font-mono tabular-nums text-slate-700 dark:text-slate-300">
                    {row.wins} / {row.seriesCount}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 overflow-hidden">
        <div className="px-5 py-3.5 border-b border-slate-200 dark:border-slate-800">
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            2. Anomaly Detection Aggregate Benchmark (5 Machines · 10,800 Rows)
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="bg-slate-50 dark:bg-slate-800/60 text-xs text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
              <tr>
                <th className="py-2.5 px-5 font-medium">Detector</th>
                <th className="py-2.5 px-5 font-medium text-right">Mean Precision</th>
                <th className="py-2.5 px-5 font-medium text-right">Mean Recall</th>
                <th className="py-2.5 px-5 font-medium text-right">Mean F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
              {summary.anomalyBenchmark.map((row) => (
                <tr key={row.detector} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                  <td className="py-3 px-5 font-medium text-slate-900 dark:text-slate-100">
                    {row.label}
                  </td>
                  <td className="py-3 px-5 text-right font-mono tabular-nums text-slate-700 dark:text-slate-300">
                    {row.meanPrecision.toFixed(4)}
                  </td>
                  <td className="py-3 px-5 text-right font-mono tabular-nums text-slate-700 dark:text-slate-300">
                    {row.meanRecall.toFixed(4)}
                  </td>
                  <td className="py-3 px-5 text-right font-mono tabular-nums font-semibold text-slate-900 dark:text-slate-100">
                    {row.meanF1.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100">
            3. In-Browser Unit Tests Status
          </h3>
          {testResults ? (
            <p className="text-xs text-slate-600 dark:text-slate-300 mt-1 font-mono tabular-nums">
              {testResults.passedCount} passed · {testResults.failedCount} failed · run at{" "}
              {testResults.runAt}
            </p>
          ) : (
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              Test suite has not been executed yet in this session.
            </p>
          )}
        </div>
        {!testResults && (
          <button
            type="button"
            onClick={onRunTests}
            className="px-4 py-2 text-xs font-semibold rounded-lg border border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer"
          >
            Run Tests Now
          </button>
        )}
      </div>
    </div>
  );
}
