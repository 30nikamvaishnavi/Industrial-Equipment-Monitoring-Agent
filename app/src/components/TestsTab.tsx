import { CheckCircle2, Play, XCircle } from "lucide-react";

import { type BrowserTestSuiteResult } from "../lib/tests";

interface TestsTabProps {
  testResults: BrowserTestSuiteResult | null;
  onRunTests: () => void;
}

export function TestsTab({ testResults, onRunTests }: TestsTabProps) {
  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
            In-Browser Verification Suite
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Tests deterministic seed generation, zero-leakage chronological splitting, exact error &
            classification metrics, numeric grounding rejection, and prompt-injection blocking.
          </p>
        </div>

        <button
          type="button"
          onClick={onRunTests}
          className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold rounded-lg bg-sky-600 hover:bg-sky-700 text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer"
        >
          <Play className="w-4 h-4" />
          Run all tests
        </button>
      </div>

      {!testResults ? (
        <div className="rounded-xl border border-dashed border-slate-300 dark:border-slate-700 bg-white/60 dark:bg-slate-900/40 p-8 text-center">
          <p className="text-sm text-slate-600 dark:text-slate-300">
            Click <span className="font-semibold">Run all tests</span> to execute all 5 unit tests
            live in your browser.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 px-1 text-sm">
            <span className="font-semibold text-slate-900 dark:text-slate-100 font-mono tabular-nums">
              {testResults.passedCount} / {testResults.total} tests passed
            </span>
            <span className="text-xs font-mono text-slate-500 dark:text-slate-400">
              Executed at {testResults.runAt}
            </span>
          </div>

          <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 divide-y divide-slate-200 dark:divide-slate-800 overflow-hidden">
            {testResults.cases.map((tc) => (
              <div key={tc.id} className="p-4 flex items-start justify-between gap-4">
                <div className="flex items-start gap-3">
                  {tc.passed ? (
                    <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400 shrink-0 mt-0.5" />
                  ) : (
                    <XCircle className="w-5 h-5 text-rose-600 dark:text-rose-400 shrink-0 mt-0.5" />
                  )}
                  <div>
                    <p className="text-sm font-semibold text-slate-900 dark:text-slate-100">
                      {tc.name}
                    </p>
                    <p className="text-xs text-slate-600 dark:text-slate-300 mt-1">{tc.details}</p>
                  </div>
                </div>
                <div className="text-right shrink-0">
                  <span
                    className={`text-xs font-mono font-semibold ${
                      tc.passed
                        ? "text-emerald-700 dark:text-emerald-400"
                        : "text-rose-700 dark:text-rose-400"
                    }`}
                  >
                    {tc.passed ? "PASS" : "FAIL"}
                  </span>
                  <p className="text-xs font-mono tabular-nums text-slate-500 dark:text-slate-400 mt-0.5">
                    {tc.durationMs.toFixed(2)} ms
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
