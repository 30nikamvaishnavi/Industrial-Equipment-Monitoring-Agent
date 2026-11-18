#!/usr/bin/env bash
set -euo pipefail

if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

echo "=== [1/8] INSTALL: Installing dependencies from requirements.txt ==="
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r requirements.txt

echo "=== [2/8] CHECK_INSTALL: Verifying library imports and versions ==="
"${PYTHON_BIN}" scripts/check_install.py

echo "=== [3/8] LINT: Running Ruff linter ==="
"${PYTHON_BIN}" -m ruff check .

echo "=== [4/8] DATA: Generating deterministic synthetic dataset and SQLite DB ==="
"${PYTHON_BIN}" -m data.generate_data

echo "=== [5/8] BENCH: Running forecasting and anomaly detection benchmarks ==="
"${PYTHON_BIN}" -m forecasting.forecaster
"${PYTHON_BIN}" -m anomaly.detector

echo "=== [6/8] TEST: Running Pytest suite ==="
"${PYTHON_BIN}" -m pytest -v

echo "=== [7/8] EVAL: Running 10-case CrewAI agent evaluation ==="
if [[ -n "${GOOGLE_API_KEY:-}" && -n "${GEMINI_MODEL:-}" ]]; then
  "${PYTHON_BIN}" -m evaluation.run_eval
else
  echo "[SKIP] Skipping agent evaluation: GOOGLE_API_KEY or GEMINI_MODEL is not set."
fi

echo "=== [8/8] README: Updating README.md Results tables from results/ ==="
"${PYTHON_BIN}" scripts/update_readme.py

echo "=== DONE: All steps completed successfully ==="
