$ErrorActionPreference = "Stop"

$PythonBin = if (Get-Command python -ErrorAction SilentlyContinue) {
    "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    "py"
} else {
    throw "Python 3.11 was not found on PATH."
}

function Invoke-Step {
    param(
        [string]$Title,
        [scriptblock]$Command
    )
    Write-Host "=== $Title ===" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed with exit code ${LASTEXITCODE}: $Title"
    }
}

Invoke-Step "[1/8] INSTALL: Installing dependencies from requirements.txt" {
    & $PythonBin -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $PythonBin -m pip install -r requirements.txt
}

Invoke-Step "[2/8] CHECK_INSTALL: Verifying library imports and versions" {
    & $PythonBin scripts/check_install.py
}

Invoke-Step "[3/8] LINT: Running Ruff linter" {
    & $PythonBin -m ruff check .
}

Invoke-Step "[4/8] DATA: Generating deterministic synthetic dataset and SQLite DB" {
    & $PythonBin -m data.generate_data
}

Invoke-Step "[5/8] BENCH: Running forecasting and anomaly detection benchmarks" {
    & $PythonBin -m forecasting.forecaster
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $PythonBin -m anomaly.detector
}

Invoke-Step "[6/8] TEST: Running Pytest suite" {
    & $PythonBin -m pytest -v
}

Write-Host "=== [7/8] EVAL: Running 10-case CrewAI agent evaluation ===" -ForegroundColor Cyan
if (-not [string]::IsNullOrWhiteSpace($env:GOOGLE_API_KEY) -and -not [string]::IsNullOrWhiteSpace($env:GEMINI_MODEL)) {
    & $PythonBin -m evaluation.run_eval
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed with exit code ${LASTEXITCODE}: [7/8] EVAL"
    }
} else {
    Write-Host "[SKIP] Skipping agent evaluation: GOOGLE_API_KEY or GEMINI_MODEL is not set." -ForegroundColor Yellow
}

Invoke-Step "[8/8] README: Updating README.md Results tables from results/" {
    & $PythonBin scripts/update_readme.py
}

Write-Host "=== DONE: All steps completed successfully ===" -ForegroundColor Green
