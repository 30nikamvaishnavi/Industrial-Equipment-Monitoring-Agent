export function HowItWorksTab() {
  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6">
        <h2 className="text-base font-semibold text-slate-900 dark:text-slate-100">
          System Pipeline Architecture
        </h2>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 mb-6">
          End-to-end flow from deterministic synthetic equipment telemetry to grounded maintenance
          recommendations.
        </p>

        <div className="overflow-x-auto pb-2">
          <svg
            viewBox="0 0 980 180"
            className="w-full min-w-[760px] h-auto"
            role="img"
            aria-label="Pipeline diagram: Synthetic Data to Forecast and Anomaly Detection to Agents to Guardrails to Report"
          >
            <defs>
              <marker
                id="arrow"
                viewBox="0 0 10 10"
                refX="8"
                refY="5"
                markerWidth="6"
                markerHeight="6"
                orient="auto-start-reverse"
              >
                <path d="M 0 1 L 10 5 L 0 9 z" className="fill-slate-400 dark:fill-slate-500" />
              </marker>
            </defs>

            {/* Box 1: Data */}
            <rect
              x="10"
              y="40"
              width="160"
              height="96"
              rx="12"
              className="fill-slate-50 dark:fill-slate-950 stroke-slate-300 dark:stroke-slate-700"
              strokeWidth="1.5"
            />
            <text
              x="90"
              y="76"
              textAnchor="middle"
              className="fill-slate-900 dark:fill-slate-100 text-[13px] font-semibold"
            >
              1. Synthetic Data
            </text>
            <text
              x="90"
              y="98"
              textAnchor="middle"
              className="fill-slate-500 dark:fill-slate-400 text-[11px]"
            >
              M1–M5 · 90d Hourly
            </text>
            <text
              x="90"
              y="115"
              textAnchor="middle"
              className="fill-slate-500 dark:fill-slate-400 text-[11px]"
            >
              Seed 42 · Logs & Docs
            </text>

            <line
              x1="170"
              y1="88"
              x2="206"
              y2="88"
              className="stroke-slate-400 dark:stroke-slate-500"
              strokeWidth="1.75"
              markerEnd="url(#arrow)"
            />

            {/* Box 2: Forecast & Anomaly */}
            <rect
              x="210"
              y="40"
              width="175"
              height="96"
              rx="12"
              className="fill-sky-50/70 dark:fill-sky-950/30 stroke-sky-300 dark:stroke-sky-800"
              strokeWidth="1.5"
            />
            <text
              x="297"
              y="76"
              textAnchor="middle"
              className="fill-slate-900 dark:fill-slate-100 text-[13px] font-semibold"
            >
              2. Forecast & Anomaly
            </text>
            <text
              x="297"
              y="98"
              textAnchor="middle"
              className="fill-slate-600 dark:fill-slate-300 text-[11px]"
            >
              Naive · Seasonal · Linear AR
            </text>
            <text
              x="297"
              y="115"
              textAnchor="middle"
              className="fill-slate-600 dark:fill-slate-300 text-[11px]"
            >
              Z-Score · Isolation Forest
            </text>

            <line
              x1="385"
              y1="88"
              x2="421"
              y2="88"
              className="stroke-slate-400 dark:stroke-slate-500"
              strokeWidth="1.75"
              markerEnd="url(#arrow)"
            />

            {/* Box 3: Agents */}
            <rect
              x="425"
              y="40"
              width="170"
              height="96"
              rx="12"
              className="fill-slate-50 dark:fill-slate-950 stroke-slate-300 dark:stroke-slate-700"
              strokeWidth="1.5"
            />
            <text
              x="510"
              y="76"
              textAnchor="middle"
              className="fill-slate-900 dark:fill-slate-100 text-[13px] font-semibold"
            >
              3. Agent Pipeline
            </text>
            <text
              x="510"
              y="98"
              textAnchor="middle"
              className="fill-slate-500 dark:fill-slate-400 text-[11px]"
            >
              Monitor + Diagnosis
            </text>
            <text
              x="510"
              y="115"
              textAnchor="middle"
              className="fill-slate-500 dark:fill-slate-400 text-[11px]"
            >
              SQL Logs + BM25 RAG
            </text>

            <line
              x1="595"
              y1="88"
              x2="631"
              y2="88"
              className="stroke-slate-400 dark:stroke-slate-500"
              strokeWidth="1.75"
              markerEnd="url(#arrow)"
            />

            {/* Box 4: Guardrails */}
            <rect
              x="635"
              y="40"
              width="165"
              height="96"
              rx="12"
              className="fill-amber-50/70 dark:fill-amber-950/30 stroke-amber-300 dark:stroke-amber-800"
              strokeWidth="1.5"
            />
            <text
              x="717"
              y="76"
              textAnchor="middle"
              className="fill-slate-900 dark:fill-slate-100 text-[13px] font-semibold"
            >
              4. Guardrails
            </text>
            <text
              x="717"
              y="98"
              textAnchor="middle"
              className="fill-slate-600 dark:fill-slate-300 text-[11px]"
            >
              Injection Filter
            </text>
            <text
              x="717"
              y="115"
              textAnchor="middle"
              className="fill-slate-600 dark:fill-slate-300 text-[11px]"
            >
              Numeric Grounding + Retry
            </text>

            <line
              x1="800"
              y1="88"
              x2="836"
              y2="88"
              className="stroke-slate-400 dark:stroke-slate-500"
              strokeWidth="1.75"
              markerEnd="url(#arrow)"
            />

            {/* Box 5: Report */}
            <rect
              x="840"
              y="40"
              width="130"
              height="96"
              rx="12"
              className="fill-emerald-50/70 dark:fill-emerald-950/30 stroke-emerald-300 dark:stroke-emerald-800"
              strokeWidth="1.5"
            />
            <text
              x="905"
              y="76"
              textAnchor="middle"
              className="fill-slate-900 dark:fill-slate-100 text-[13px] font-semibold"
            >
              5. Final Report
            </text>
            <text
              x="905"
              y="98"
              textAnchor="middle"
              className="fill-slate-600 dark:fill-slate-300 text-[11px]"
            >
              Gemini SDK
            </text>
            <text
              x="905"
              y="115"
              textAnchor="middle"
              className="fill-slate-600 dark:fill-slate-300 text-[11px]"
            >
              Verified Output
            </text>
          </svg>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6">
        <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-100 mb-4">
          Five Stages in Plain English
        </h3>
        <ul className="space-y-3 text-sm text-slate-700 dark:text-slate-300 list-disc pl-5">
          <li>
            <span className="font-semibold text-slate-900 dark:text-slate-100">
              1. Synthetic Data Generation:
            </span>{" "}
            A fixed random seed (42) creates 90 days of hourly temperature, vibration, pressure, and
            RPM telemetry for 5 machines, plus ~2% labelled faults (spikes, gradual drift, and
            flat-line sensor failures) and 40 matching maintenance logs.
          </li>
          <li>
            <span className="font-semibold text-slate-900 dark:text-slate-100">
              2. Forecasting & Anomaly Detection:
            </span>{" "}
            Three forecasting models (naive, 24h seasonal-naive, and least-squares linear
            autoregression) compete using a strict past-only time split, while causal Rolling
            Z-Score and a seeded Isolation Forest scan for faults and compare precision, recall,
            and F1 against ground truth.
          </li>
          <li>
            <span className="font-semibold text-slate-900 dark:text-slate-100">
              3. Multi-Step Agent Diagnosis:
            </span>{" "}
            The Monitor step runs the forecast and anomaly tools, and the Diagnosis step performs a
            read-only lookup of past maintenance logs plus BM25 keyword search across four built-in
            engineering manuals.
          </li>
          <li>
            <span className="font-semibold text-slate-900 dark:text-slate-100">
              4. Input & Output Guardrails:
            </span>{" "}
            Operator notes are screened for prompt injection, SQL injection, and script tags before
            any agent runs; after report generation, every single number in the prose is checked
            against captured tool outputs (retrying once before rejecting ungrounded claims).
          </li>
          <li>
            <span className="font-semibold text-slate-900 dark:text-slate-100">
              5. Grounded Maintenance Report:
            </span>{" "}
            Using the <span className="font-mono">@google/genai</span> SDK with{" "}
            <span className="font-mono">GEMINI_API_KEY</span>, the Report step synthesizes an
            actionable recommendation strictly grounded in verified telemetry, maintenance history,
            and manual guidance.
          </li>
        </ul>
      </div>
    </div>
  );
}
