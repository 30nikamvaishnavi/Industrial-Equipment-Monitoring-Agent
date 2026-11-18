export interface InjectionCheckResult {
  isSafe: boolean;
  matchedPatterns: string[];
  reason: string;
}

export interface NumberGuardrailResult {
  passed: boolean;
  reportNumbers: string[];
  ungroundedNumbers: string[];
  reason: string;
}

const INJECTION_PATTERNS: Array<[string, RegExp]> = [
  ["ignore_instructions", /\bignore\s+(?:all\s+|previous\s+|prior\s+)?instructions\b/i],
  ["disregard_instructions", /\bdisregard\s+(?:all\s+|previous\s+|prior\s+)?instructions\b/i],
  ["system_prompt_probe", /\b(?:reveal|show|print|leak)\s+(?:your\s+|the\s+)?system\s+prompt\b/i],
  ["role_override", /\b(?:you\s+are\s+now|act\s+as\s+(?:a\s+)?(?:dan|root|admin)|jailbreak)\b/i],
  ["bypass_guardrail", /\b(?:bypass|disable|override)\s+(?:guardrails?|safety|filters?|rules?)\b/i],
  ["sql_injection", /(?:;\s*(?:drop|delete|alter|insert|update|attach|pragma)\b|--\s*$)/i],
  ["script_tag", /<\s*script\b/i],
];

const MACHINE_OR_LOG_ID_RE = /\b(?:M\d+|LOG-\d+)\b/gi;
const NUMBER_RE = /(?:^|[^A-Za-z0-9_])(-?\d+(?:,\d{3})*(?:\.\d+)?%?)(?![A-Za-z0-9_])/g;

export function checkPromptInjection(userInput: string): InjectionCheckResult {
  if (!userInput || !userInput.trim()) {
    return {
      isSafe: true,
      matchedPatterns: [],
      reason: "Empty operator notes passed prompt-injection check.",
    };
  }

  const matched = INJECTION_PATTERNS.filter(([, pattern]) => pattern.test(userInput)).map(
    ([name]) => name,
  );

  if (matched.length > 0) {
    return {
      isSafe: false,
      matchedPatterns: matched,
      reason: `Input blocked by prompt-injection guardrail: matched ${matched.join(", ")}.`,
    };
  }

  return {
    isSafe: true,
    matchedPatterns: [],
    reason: "No prompt-injection or SQL-injection patterns detected.",
  };
}

function parseNumericTokens(text: string): Array<{ raw: string; value: number }> {
  const withoutIds = text.replace(MACHINE_OR_LOG_ID_RE, " ");
  const normalizedTs = withoutIds.replace(/(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/g, "$1 $2");
  const tokens: Array<{ raw: string; value: number }> = [];

  NUMBER_RE.lastIndex = 0;
  let match: RegExpExecArray | null = NUMBER_RE.exec(normalizedTs);
  while (match !== null) {
    const raw = match[1];
    const clean = raw.replace(/%$/, "").replace(/,/g, "");
    const val = Number(clean);
    if (!Number.isNaN(val) && Number.isFinite(val)) {
      tokens.push({ raw, value: val });
    }
    match = NUMBER_RE.exec(normalizedTs);
  }
  return tokens;
}

export function verifyGroundedNumbers(
  reportText: string,
  toolOutputs: string[],
  tolerance: number = 1e-6,
): NumberGuardrailResult {
  if (!reportText || !reportText.trim()) {
    return {
      passed: false,
      reportNumbers: [],
      ungroundedNumbers: [],
      reason: "Report text is empty.",
    };
  }

  const toolTokens = parseNumericTokens(toolOutputs.join("\n"));
  const toolRawSet = new Set(toolTokens.map((t) => t.raw.replace(/%$/, "").replace(/,/g, "")));
  const toolValues = toolTokens.map((t) => t.value);

  const reportTokens = parseNumericTokens(reportText);
  const reportNumbers = reportTokens.map((t) => t.raw);
  const ungrounded: string[] = [];

  for (const tok of reportTokens) {
    const norm = tok.raw.replace(/%$/, "").replace(/,/g, "");
    if (toolRawSet.has(norm)) {
      continue;
    }
    if (toolValues.some((tv) => Math.abs(tok.value - tv) <= tolerance)) {
      continue;
    }
    ungrounded.push(tok.raw);
  }

  if (ungrounded.length > 0) {
    const uniqueUngrounded = Array.from(new Set(ungrounded));
    return {
      passed: false,
      reportNumbers,
      ungroundedNumbers: uniqueUngrounded,
      reason: `Report rejected by numeric grounding guardrail: ungrounded number(s) [${uniqueUngrounded.join(", ")}] were not present in tool outputs.`,
    };
  }

  return {
    passed: true,
    reportNumbers,
    ungroundedNumbers: [],
    reason: "All numbers in the report are grounded in tool outputs.",
  };
}
