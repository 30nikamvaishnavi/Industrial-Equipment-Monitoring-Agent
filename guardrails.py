"""Input prompt-injection and output numeric-grounding guardrails."""

from __future__ import annotations

import re
from dataclasses import dataclass

_MACHINE_OR_LOG_ID_RE = re.compile(r"\b(?:M\d+|LOG-\d+)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])-?\d+(?:,\d{3})*(?:\.\d+)?%?(?![A-Za-z0-9_])")

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_instructions",
        re.compile(r"\bignore\s+(?:all\s+|previous\s+|prior\s+)?instructions\b", re.IGNORECASE),
    ),
    (
        "disregard_instructions",
        re.compile(
            r"\bdisregard\s+(?:all\s+|previous\s+|prior\s+)?instructions\b",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_probe",
        re.compile(
            r"\b(?:reveal|show|print|leak)\s+(?:your\s+|the\s+)?system\s+prompt\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_override",
        re.compile(
            r"\b(?:you\s+are\s+now|act\s+as\s+(?:a\s+)?(?:dan|root|admin)|jailbreak)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "bypass_guardrail",
        re.compile(
            r"\b(?:bypass|disable|override)\s+(?:guardrails?|safety|filters?|rules?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "sql_injection",
        re.compile(
            r"(?:;\s*(?:drop|delete|alter|insert|update|attach|pragma)\b|--\s*$)",
            re.IGNORECASE,
        ),
    ),
    (
        "script_tag",
        re.compile(r"<\s*script\b", re.IGNORECASE),
    ),
)


@dataclass(frozen=True)
class NumberGuardrailResult:
    """Result of numeric grounding verification on an agent-generated report."""

    passed: bool
    report_numbers: list[str]
    ungrounded_numbers: list[str]
    reason: str


@dataclass(frozen=True)
class InjectionCheckResult:
    """Result of prompt-injection and unsafe-input screening."""

    is_safe: bool
    matched_patterns: list[str]
    reason: str


def _parse_numeric_tokens(text: str) -> list[tuple[str, float]]:
    """Extract numeric tokens and their float representations from text.

    Machine identifiers (e.g., M1..M5) and maintenance log IDs (e.g., LOG-0001)
    are excluded so alphanumeric labels are not confused with quantities.
    ISO-8601 'T' separators are normalized to spaces so timestamps match
    consistently whether formatted with 'T' or a space.
    """
    cleaned = _MACHINE_OR_LOG_ID_RE.sub(" ", text)
    cleaned = re.sub(r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})", r"\1 \2", cleaned)
    tokens: list[tuple[str, float]] = []
    for match in _NUMBER_RE.finditer(cleaned):
        raw = match.group(0)
        normalized = raw.rstrip("%").replace(",", "")
        try:
            val = float(normalized)
        except ValueError:
            continue
        tokens.append((raw, val))
    return tokens


def verify_grounded_numbers(
    report_text: str,
    tool_outputs: list[str],
    tolerance: float = 1e-6,
) -> NumberGuardrailResult:
    """Verify every number in report_text appeared in at least one tool output.

    Args:
        report_text: Generated maintenance report text.
        tool_outputs: List of raw tool output strings collected during execution.
        tolerance: Floating-point equivalence tolerance.

    Returns:
        NumberGuardrailResult indicating whether all report numbers are grounded.
    """
    if not report_text or not report_text.strip():
        return NumberGuardrailResult(
            passed=False,
            report_numbers=[],
            ungrounded_numbers=[],
            reason="Report text is empty.",
        )

    tool_corpus = "\n".join(tool_outputs)
    tool_tokens = _parse_numeric_tokens(tool_corpus)
    tool_values = [val for _, val in tool_tokens]
    tool_raw_set = {raw.rstrip("%").replace(",", "") for raw, _ in tool_tokens}

    report_tokens = _parse_numeric_tokens(report_text)
    report_raw = [raw for raw, _ in report_tokens]
    ungrounded: list[str] = []

    for raw, val in report_tokens:
        norm = raw.rstrip("%").replace(",", "")
        if norm in tool_raw_set:
            continue
        if any(abs(val - tv) <= tolerance for tv in tool_values):
            continue
        ungrounded.append(raw)

    if ungrounded:
        unique_ungrounded = list(dict.fromkeys(ungrounded))
        return NumberGuardrailResult(
            passed=False,
            report_numbers=report_raw,
            ungrounded_numbers=unique_ungrounded,
            reason=(
                "Report rejected by numeric grounding guardrail: ungrounded number(s) "
                f"{unique_ungrounded} were not produced by any tool output."
            ),
        )

    return NumberGuardrailResult(
        passed=True,
        report_numbers=report_raw,
        ungrounded_numbers=[],
        reason="All numbers in the report are grounded in tool outputs.",
    )


def check_prompt_injection(user_input: str) -> InjectionCheckResult:
    """Inspect user-supplied text for prompt-injection or SQL-injection patterns.

    Args:
        user_input: User query, notes, or parameter string.

    Returns:
        InjectionCheckResult specifying whether the input is safe.
    """
    if not user_input:
        return InjectionCheckResult(
            is_safe=True,
            matched_patterns=[],
            reason="Empty input is safe.",
        )

    matched = [name for name, pattern in _INJECTION_PATTERNS if pattern.search(user_input)]
    if matched:
        return InjectionCheckResult(
            is_safe=False,
            matched_patterns=matched,
            reason=(
                "Input rejected by prompt-injection guardrail: matched suspicious "
                f"pattern(s): {', '.join(matched)}."
            ),
        )

    return InjectionCheckResult(
        is_safe=True,
        matched_patterns=[],
        reason="No prompt-injection patterns detected.",
    )
