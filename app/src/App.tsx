import { useEffect, useState } from "react";
import {
  Activity,
  BarChart3,
  Bot,
  Database,
  FileCheck2,
  FlaskConical,
  GitBranch,
  Moon,
  Sun,
  TrendingUp,
} from "lucide-react";

import { AgentReportTab } from "./components/AgentReportTab";
import { AnomaliesTab } from "./components/AnomaliesTab";
import { DataTab } from "./components/DataTab";
import { ForecastTab } from "./components/ForecastTab";
import { HowItWorksTab } from "./components/HowItWorksTab";
import { ResultsTab } from "./components/ResultsTab";
import { TestsTab } from "./components/TestsTab";
import { generateSyntheticDataset, SEED, type SyntheticDataset } from "./lib/data";
import { runAllBrowserTests, type BrowserTestSuiteResult } from "./lib/tests";

type TabKey = "data" | "forecast" | "anomalies" | "agent" | "tests" | "results" | "how";

const TABS: Array<{ key: TabKey; label: string; icon: typeof Database }> = [
  { key: "data", label: "Data", icon: Database },
  { key: "forecast", label: "Forecast", icon: TrendingUp },
  { key: "anomalies", label: "Anomalies", icon: Activity },
  { key: "agent", label: "Agent report", icon: Bot },
  { key: "tests", label: "Tests", icon: FlaskConical },
  { key: "results", label: "Results", icon: BarChart3 },
  { key: "how", label: "How it works", icon: GitBranch },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<TabKey>("data");
  const [isDark, setIsDark] = useState<boolean>(false);
  const [dataset, setDataset] = useState<SyntheticDataset>(() => generateSyntheticDataset(SEED));
  const [testResults, setTestResults] = useState<BrowserTestSuiteResult | null>(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", isDark);
  }, [isDark]);

  const handleRegenerate = () => {
    setDataset(generateSyntheticDataset(SEED));
  };

  const handleRunTests = () => {
    setTestResults(runAllBrowserTests());
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors">
      <header className="border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-8">
          <span className="font-display text-lg font-bold tracking-tight text-slate-900 dark:text-slate-100 whitespace-nowrap shrink-0">
            Equipment Monitor
          </span>

          <nav
            aria-label="Dashboard sections"
            className="hidden lg:flex items-center gap-6 text-sm font-medium"
          >
            {TABS.slice(0, 5).map((tab) => (
              <button
                key={tab.key}
                type="button"
                onClick={() => setActiveTab(tab.key)}
                className={`whitespace-nowrap shrink-0 py-1 transition-colors cursor-pointer ${
                  activeTab === tab.key
                    ? "text-sky-600 dark:text-sky-400 font-semibold"
                    : "text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>

          <button
            type="button"
            onClick={() => setIsDark((prev) => !prev)}
            aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
            className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            <span>{isDark ? "Light mode" : "Dark mode"}</span>
          </button>
        </div>
      </header>

      <div className="border-b border-slate-200 dark:border-slate-800 bg-white/70 dark:bg-slate-900/70">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-2.5 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-1 overflow-x-auto pb-0.5">
            {TABS.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.key;
              return (
                <button
                  key={tab.key}
                  type="button"
                  onClick={() => setActiveTab(tab.key)}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                    isActive
                      ? "bg-slate-900 text-white dark:bg-sky-600 dark:text-white"
                      : "text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:text-slate-900 dark:hover:text-slate-100"
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {tab.label}
                </button>
              );
            })}
          </div>

          <div className="hidden sm:flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400 font-mono tabular-nums">
            <FileCheck2 className="w-3.5 h-3.5 text-slate-400" />
            <span>{dataset.stats.totalRows.toLocaleString()} readings</span>
            <span aria-hidden="true">·</span>
            <span>{dataset.stats.logsCount} logs</span>
          </div>
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === "data" && (
          <DataTab dataset={dataset} onRegenerate={handleRegenerate} />
        )}
        {activeTab === "forecast" && (
          <ForecastTab readings={dataset.readings} isDark={isDark} />
        )}
        {activeTab === "anomalies" && (
          <AnomaliesTab readings={dataset.readings} isDark={isDark} />
        )}
        {activeTab === "agent" && (
          <AgentReportTab readings={dataset.readings} logs={dataset.logs} />
        )}
        {activeTab === "tests" && (
          <TestsTab testResults={testResults} onRunTests={handleRunTests} />
        )}
        {activeTab === "results" && (
          <ResultsTab
            readings={dataset.readings}
            testResults={testResults}
            onRunTests={handleRunTests}
          />
        )}
        {activeTab === "how" && <HowItWorksTab />}
      </main>
    </div>
  );
}
