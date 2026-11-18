import { detectAnomalies } from "./anomaly";
import {
  MANUAL_DOCS,
  type MachineId,
  type MaintenanceLog,
  type SensorName,
  type SensorReading,
} from "./data";
import { evaluateAndForecast } from "./forecast";
import {
  checkPromptInjection,
  verifyGroundedNumbers,
  type InjectionCheckResult,
  type NumberGuardrailResult,
} from "./guardrails";

// Check Google AI Studio for the current Gemini model name.
export const GEMINI_MODEL = "gemini-4-hydrogen";

export interface ToolCallTrace {
  toolName: "forecast_tool" | "anomaly_tool" | "maintenance_log_tool" | "rag_tool";
  input: string;
  output: string;
}

export interface AgentTimelineStep {
  id: string;
  agent: "Guardrail" | "Monitor Agent" | "Diagnosis Agent" | "Report Agent";
  status: "completed" | "warning" | "blocked" | "failed";
  summary: string;
  toolCalls: ToolCallTrace[];
  durationMs: number;
}

export interface ManualSearchHit {
  source: string;
  section: string;
  score: number;
  content: string;
}

export interface AgentPipelineResult {
  machineId: MachineId;
  sensor: SensorName;
  modelName: string;
  steps: AgentTimelineStep[];
  injectionCheck: InjectionCheckResult;
  guardrailCheck: NumberGuardrailResult | null;
  retried: boolean;
  apiKeyMissing: boolean;
  reportText: string | null;
  errorMessage: string | null;
}

interface ManualChunk {
  source: string;
  section: string;
  content: string;
  tokens: string[];
}

function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).filter((t) => t.length > 1);
}

function chunkManuals(): ManualChunk[] {
  const chunks: ManualChunk[] = [];
  for (const doc of MANUAL_DOCS) {
    const sections = doc.content.split(/\n(?=##\s+)/);
    for (const sec of sections) {
      const clean = sec.trim();
      if (!clean) continue;
      const firstLine = clean.split("\n")[0].replace(/^#+\s*/, "").trim();
      const sectionName = clean.startsWith("##") ? `${doc.title} - ${firstLine}` : doc.title;
      chunks.push({
        source: doc.filename,
        section: sectionName,
        content: clean,
        tokens: tokenize(clean),
      });
    }
  }
  return chunks;
}

const MANUAL_CHUNKS = chunkManuals();

/** BM25 Okapi search over the 4 built-in maintenance manual markdown documents. */
export function searchManualsBM25(query: string, topK: number = 3): ManualSearchHit[] {
  const queryTokens = tokenize(query);
  if (queryTokens.length === 0) return [];

  const nDocs = MANUAL_CHUNKS.length;
  const avgDocLen =
    MANUAL_CHUNKS.reduce((acc, c) => acc + c.tokens.length, 0) / Math.max(1, nDocs);
  const k1 = 1.5;
  const b = 0.75;

  const docFreq = new Map<string, number>();
  for (const chunk of MANUAL_CHUNKS) {
    const unique = new Set(chunk.tokens);
    for (const tok of unique) {
      docFreq.set(tok, (docFreq.get(tok) ?? 0) + 1);
    }
  }

  const scored = MANUAL_CHUNKS.map((chunk) => {
    const tfMap = new Map<string, number>();
    for (const tok of chunk.tokens) {
      tfMap.set(tok, (tfMap.get(tok) ?? 0) + 1);
    }
    let score = 0;
    for (const qTok of queryTokens) {
      const df = docFreq.get(qTok) ?? 0;
      if (df === 0) continue;
      const idf = Math.log((nDocs - df + 0.5) / (df + 0.5) + 1.0);
      const tf = tfMap.get(qTok) ?? 0;
      const denom = tf + k1 * (1 - b + (b * chunk.tokens.length) / avgDocLen);
      score += idf * ((tf * (k1 + 1)) / Math.max(denom, 1e-9));
    }
    return {
      source: chunk.source,
      section: chunk.section,
      score: Math.round(score * 10000) / 10000,
      content: chunk.content,
    };
  });

  return scored
    .sort((a, bHit) => bHit.score - a.score)
    .slice(0, topK);
}

/** Read-only query against synthetic maintenance logs for a given machine. */
export function queryMaintenanceLogs(logs: MaintenanceLog[], machineId: MachineId): MaintenanceLog[] {
  return logs
    .filter((log) => log.machine_id === machineId)
    .sort((a, b) => b.date.localeCompare(a.date));
}

async function callGeminiReportEndpoint(
  prompt: string,
  systemInstruction: string,
): Promise<{ ok: true; report: string; model: string } | { ok: false; status: number; error: string; message: string }> {
  const resp = await fetch("/api/report/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, systemInstruction }),
  });

  const payload = (await resp.json().catch(() => ({}))) as {
    report?: string;
    model?: string;
    error?: string;
    message?: string;
  };

  if (!resp.ok || !payload.report) {
    return {
      ok: false,
      status: resp.status,
      error: payload.error ?? "api_error",
      message: payload.message ?? `HTTP ${resp.status} while calling Gemini report endpoint.`,
    };
  }

  return {
    ok: true,
    report: payload.report,
    model: payload.model ?? GEMINI_MODEL,
  };
}

export async function runAgentPipeline(params: {
  readings: SensorReading[];
  logs: MaintenanceLog[];
  machineId: MachineId;
  sensor: SensorName;
  notes: string;
  onProgress?: (steps: AgentTimelineStep[]) => void;
}): Promise<AgentPipelineResult> {
  const { readings, logs, machineId, sensor, notes, onProgress } = params;
  const steps: AgentTimelineStep[] = [];

  const pushStep = (step: AgentTimelineStep) => {
    steps.push(step);
    onProgress?.([...steps]);
  };

  // 0. Prompt-injection guardrail on operator notes
  const guardStart = performance.now();
  const injectionCheck = checkPromptInjection(notes);
  if (!injectionCheck.isSafe) {
    pushStep({
      id: "input-guardrail",
      agent: "Guardrail",
      status: "blocked",
      summary: injectionCheck.reason,
      toolCalls: [],
      durationMs: Math.round(performance.now() - guardStart),
    });
    return {
      machineId,
      sensor,
      modelName: GEMINI_MODEL,
      steps,
      injectionCheck,
      guardrailCheck: null,
      retried: false,
      apiKeyMissing: false,
      reportText: null,
      errorMessage: injectionCheck.reason,
    };
  }

  pushStep({
    id: "input-guardrail",
    agent: "Guardrail",
    status: "completed",
    summary: injectionCheck.reason,
    toolCalls: [],
    durationMs: Math.round(performance.now() - guardStart),
  });

  // 1. Monitor Agent: forecast_tool + anomaly_tool
  const monStart = performance.now();
  const forecastRes = evaluateAndForecast(readings, machineId, sensor, 24);
  const forecastOutputJson = JSON.stringify(
    {
      machine_id: forecastRes.machineId,
      sensor: forecastRes.sensor,
      horizon_hours: forecastRes.horizonHours,
      last_observed_timestamp: forecastRes.lastObservedTimestamp,
      last_observed_value: forecastRes.lastObservedValue,
      best_model: forecastRes.bestModel,
      model_metrics: forecastRes.metrics.map((m) => ({
        model: m.model,
        mae: m.mae,
        rmse: m.rmse,
      })),
      next_24h_summary: {
        min_predicted: Math.min(...forecastRes.futureForecast.map((p) => p.linear_ar)),
        max_predicted: Math.max(...forecastRes.futureForecast.map((p) => p.linear_ar)),
        first_hour_ar: forecastRes.futureForecast[0]?.linear_ar,
        last_hour_ar: forecastRes.futureForecast[forecastRes.futureForecast.length - 1]?.linear_ar,
      },
    },
    null,
    2,
  );

  const anomalyRes = detectAnomalies(readings, machineId);
  const recentFlagged = anomalyRes.latestDetections.slice(0, 6);
  const anomalyOutputJson = JSON.stringify(
    {
      machine_id: anomalyRes.machineId,
      rows_scanned: anomalyRes.rowsScanned,
      detector_metrics: anomalyRes.metrics.map((m) => ({
        detector: m.detector,
        precision: m.precision,
        recall: m.recall,
        f1: m.f1,
        flagged_count: m.flaggedCount,
      })),
      latest_detections: recentFlagged.map((d) => ({
        timestamp: d.timestamp,
        temperature: d.temperature,
        vibration: d.vibration,
        pressure: d.pressure,
        rpm: d.rpm,
        max_z_score: d.maxZScore,
        isolation_score: d.isolationScore,
        reason: d.reason,
      })),
    },
    null,
    2,
  );

  pushStep({
    id: "monitor-agent",
    agent: "Monitor Agent",
    status: "completed",
    summary: `Evaluated 24h forecast (best model: ${forecastRes.bestModel}) and scanned ${anomalyRes.rowsScanned} hourly readings across both detectors.`,
    toolCalls: [
      {
        toolName: "forecast_tool",
        input: JSON.stringify({ machine_id: machineId, sensor, horizon_hours: 24 }),
        output: forecastOutputJson,
      },
      {
        toolName: "anomaly_tool",
        input: JSON.stringify({ machine_id: machineId, window: 24, z_threshold: 3.2 }),
        output: anomalyOutputJson,
      },
    ],
    durationMs: Math.round(performance.now() - monStart),
  });

  // 2. Diagnosis Agent: maintenance_log_tool + rag_tool (BM25)
  const diagStart = performance.now();
  const machineLogs = queryMaintenanceLogs(logs, machineId);
  const logsOutputJson = JSON.stringify(
    {
      sql: `SELECT log_id, machine_id, date, issue, action_taken, downtime_hours FROM maintenance_logs WHERE machine_id = '${machineId}' ORDER BY date DESC`,
      row_count: machineLogs.length,
      rows: machineLogs,
    },
    null,
    2,
  );

  const symptoms = recentFlagged
    .slice(0, 3)
    .map((d) => d.reason)
    .filter(Boolean)
    .join(" ");
  const ragQuery = `${sensor} anomalies ${symptoms} ${notes}`.trim();
  const manualHits = searchManualsBM25(ragQuery, 3);
  const ragOutputJson = JSON.stringify({ query: ragQuery, results: manualHits }, null, 2);

  pushStep({
    id: "diagnosis-agent",
    agent: "Diagnosis Agent",
    status: "completed",
    summary: `Queried ${machineLogs.length} historical maintenance logs for ${machineId} and retrieved ${manualHits.length} matching manual sections via BM25.`,
    toolCalls: [
      {
        toolName: "maintenance_log_tool",
        input: `SELECT * FROM maintenance_logs WHERE machine_id = '${machineId}' ORDER BY date DESC`,
        output: logsOutputJson,
      },
      {
        toolName: "rag_tool",
        input: JSON.stringify({ query: ragQuery, top_k: 3 }),
        output: ragOutputJson,
      },
    ],
    durationMs: Math.round(performance.now() - diagStart),
  });

  // 3. Report Agent: Gemini SDK via /api/report/generate + numeric grounding check & 1 retry
  const allToolOutputs = [forecastOutputJson, anomalyOutputJson, logsOutputJson, ragOutputJson];
  const systemInstruction =
    "You are a reliability maintenance engineer writing a concise diagnostic report. " +
    "STRICT RULE: Every single number in your response MUST appear verbatim in the provided tool outputs. " +
    "Never invent thresholds, counts, percentages, or numbered list items (use '-' bullets instead of 1., 2., 3.).";

  const basePrompt =
    `Machine: ${machineId}\nPrimary sensor: ${sensor}\nOperator notes: ${notes || "None"}\n\n` +
    `TOOL OUTPUT 1 (forecast_tool):\n${forecastOutputJson}\n\n` +
    `TOOL OUTPUT 2 (anomaly_tool):\n${anomalyOutputJson}\n\n` +
    `TOOL OUTPUT 3 (maintenance_log_tool):\n${logsOutputJson}\n\n` +
    `TOOL OUTPUT 4 (rag_tool):\n${ragOutputJson}\n\n` +
    "Write a concise maintenance recommendation report covering observed metrics, detected anomalies, likely causes from the logs/manuals, and recommended actions. Use '-' bullets only.";

  const repStart = performance.now();
  const firstCall = await callGeminiReportEndpoint(basePrompt, systemInstruction);
  if (!firstCall.ok) {
    const missingKey = firstCall.error === "missing_gemini_api_key" || firstCall.status === 503;
    pushStep({
      id: "report-agent",
      agent: "Report Agent",
      status: missingKey ? "warning" : "failed",
      summary: firstCall.message,
      toolCalls: [],
      durationMs: Math.round(performance.now() - repStart),
    });
    return {
      machineId,
      sensor,
      modelName: GEMINI_MODEL,
      steps,
      injectionCheck,
      guardrailCheck: null,
      retried: false,
      apiKeyMissing: missingKey,
      reportText: null,
      errorMessage: firstCall.message,
    };
  }

  let reportText = firstCall.report;
  let guardrailCheck = verifyGroundedNumbers(reportText, allToolOutputs);
  let retried = false;

  if (!guardrailCheck.passed) {
    retried = true;
    pushStep({
      id: "report-agent-attempt-1",
      agent: "Report Agent",
      status: "warning",
      summary: `Attempt 1 rejected by numeric grounding guardrail (ungrounded: ${guardrailCheck.ungroundedNumbers.join(", ")}). Retrying once with explicit removal list.`,
      toolCalls: [],
      durationMs: Math.round(performance.now() - repStart),
    });

    const retryStart = performance.now();
    const retryPrompt =
      `${basePrompt}\n\nCORRECTION REQUIRED: Your previous draft was rejected because these numbers were NOT in any tool output: ${guardrailCheck.ungroundedNumbers.join(", ")}. ` +
      "Rewrite the report removing ALL of those ungrounded numbers and citing only exact numbers from the tool outputs above.";

    const secondCall = await callGeminiReportEndpoint(retryPrompt, systemInstruction);
    if (!secondCall.ok) {
      pushStep({
        id: "report-agent",
        agent: "Report Agent",
        status: "failed",
        summary: secondCall.message,
        toolCalls: [],
        durationMs: Math.round(performance.now() - retryStart),
      });
      return {
        machineId,
        sensor,
        modelName: firstCall.model,
        steps,
        injectionCheck,
        guardrailCheck,
        retried,
        apiKeyMissing: false,
        reportText: null,
        errorMessage: secondCall.message,
      };
    }

    reportText = secondCall.report;
    guardrailCheck = verifyGroundedNumbers(reportText, allToolOutputs);
    if (!guardrailCheck.passed) {
      pushStep({
        id: "report-agent",
        agent: "Report Agent",
        status: "blocked",
        summary: `Retry rejected: ${guardrailCheck.reason}`,
        toolCalls: [],
        durationMs: Math.round(performance.now() - retryStart),
      });
      return {
        machineId,
        sensor,
        modelName: secondCall.model,
        steps,
        injectionCheck,
        guardrailCheck,
        retried,
        apiKeyMissing: false,
        reportText: null,
        errorMessage: guardrailCheck.reason,
      };
    }

    pushStep({
      id: "report-agent",
      agent: "Report Agent",
      status: "completed",
      summary: `Retry passed numeric grounding guardrail (${guardrailCheck.reportNumbers.length} numbers verified against tool outputs).`,
      toolCalls: [],
      durationMs: Math.round(performance.now() - retryStart),
    });
    return {
      machineId,
      sensor,
      modelName: secondCall.model,
      steps,
      injectionCheck,
      guardrailCheck,
      retried,
      apiKeyMissing: false,
      reportText,
      errorMessage: null,
    };
  }

  pushStep({
    id: "report-agent",
    agent: "Report Agent",
    status: "completed",
    summary: `Generated report with ${firstCall.model}; numeric grounding guardrail verified ${guardrailCheck.reportNumbers.length} numbers.`,
    toolCalls: [],
    durationMs: Math.round(performance.now() - repStart),
  });

  return {
    machineId,
    sensor,
    modelName: firstCall.model,
    steps,
    injectionCheck,
    guardrailCheck,
    retried,
    apiKeyMissing: false,
    reportText,
    errorMessage: null,
  };
}
