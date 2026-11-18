"""Verify that all key runtime and dev dependencies import cleanly."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys

REQUIRED_MODULES: tuple[tuple[str, str], ...] = (
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("statsmodels", "statsmodels"),
    ("langchain", "langchain"),
    ("langchain_core", "langchain-core"),
    ("langchain_community", "langchain-community"),
    ("langchain_google_genai", "langchain-google-genai"),
    ("crewai", "crewai"),
    ("litellm", "litellm"),
    ("faiss", "faiss-cpu"),
    ("rank_bm25", "rank-bm25"),
    ("google.generativeai", "google-generativeai"),
    ("dotenv", "python-dotenv"),
    ("httpx", "httpx"),
    ("pytest", "pytest"),
    ("ruff", "ruff"),
)


def _get_version(module: object, dist_name: str) -> str:
    """Resolve version from module.__version__ or distribution metadata."""
    ver = getattr(module, "__version__", None)
    if ver:
        return str(ver)
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> int:
    """Import all key libraries, print versions, and fail fast on conflicts."""
    failures: list[str] = []
    print(f"Python: {sys.version.split()[0]}")

    for import_name, dist_name in REQUIRED_MODULES:
        try:
            mod = importlib.import_module(import_name)
            version = _get_version(mod, dist_name)
            print(f"  [OK] {dist_name} ({import_name}): {version}")
        except Exception as exc:
            msg = f"  [FAIL] {dist_name} ({import_name}): {exc}"
            print(msg, file=sys.stderr)
            failures.append(msg)

    if failures:
        print(
            f"\nDependency check failed for {len(failures)} package(s). "
            "Reinstall requirements.txt in a clean Python 3.11 virtual environment.",
            file=sys.stderr,
        )
        return 1

    print("\nAll key dependencies imported successfully with zero conflicts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
