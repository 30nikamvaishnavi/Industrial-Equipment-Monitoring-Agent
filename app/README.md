# Industrial Equipment Monitoring Agent — Browser Dashboard (`app/`)

This folder contains an interactive **browser-based demo** (`React + TypeScript + Vite + Tailwind CSS + Recharts`) that re-implements the predictive maintenance pipeline in TypeScript so you can explore, inspect, and test every component directly in the AI Studio preview or locally with zero Python setup.

> **Browser Demo vs. Production Backend**:
> - **`app/` (this folder)**: Runs synthetic data generation, walk-forward forecasting, Isolation Forest / Rolling Z-Score anomaly detection, read-only maintenance log querying, BM25 manual search, input/output guardrails, and in-browser unit tests directly in TypeScript in the browser (plus a lightweight Node/Express `/api/report/generate` proxy so the `@google/genai` SDK reads `GEMINI_API_KEY` safely on the server).
> - **Repository Root (`data/`, `forecasting/`, `anomaly/`, `agents/`, `api/`, `evaluation/`, `guardrails.py`)**: The full Python 3.11 production backend featuring `statsmodels` ARIMA, `scikit-learn` Gradient Boosting & Isolation Forest, SQLite `maintenance.db`, FAISS + BM25 hybrid RAG, CrewAI multi-agent orchestration, FastAPI REST endpoints, and Docker deployment.

## Running Locally

```bash
cd app
npm install
npm run dev
```

Open `http://localhost:3000`. To enable the live Gemini Report generation step in the **Agent report** tab, export `GEMINI_API_KEY` before starting the server:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
npm run dev
```

To change the Gemini model, edit the `GEMINI_MODEL` constant at the top of `app/server.ts` and `app/src/lib/agents.ts` (check Google AI Studio for the current model name).
