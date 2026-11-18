import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";
import express from "express";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { createServer as createViteServer } from "vite";

// Check Google AI Studio for the current Gemini model name.
export const GEMINI_MODEL = "gemini-4-hydrogen";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.resolve(__dirname, "..", ".env") });
dotenv.config({ path: path.resolve(__dirname, ".env") });

interface GenerateReportBody {
  prompt?: string;
  systemInstruction?: string;
}

async function createAppServer(): Promise<void> {
  const app = express();
  app.use(express.json({ limit: "1mb" }));

  app.get("/api/config", (_req, res) => {
    const hasKey = Boolean(process.env.GEMINI_API_KEY?.trim());
    res.json({
      geminiConfigured: hasKey,
      model: GEMINI_MODEL,
    });
  });

  app.post("/api/report/generate", async (req, res) => {
    const apiKey = process.env.GEMINI_API_KEY?.trim();
    if (!apiKey) {
      res.status(503).json({
        error: "missing_gemini_api_key",
        message:
          "GEMINI_API_KEY is not configured in the server environment. Monitor and Diagnosis steps remain available.",
        model: GEMINI_MODEL,
      });
      return;
    }

    const body = (req.body ?? {}) as GenerateReportBody;
    const prompt = typeof body.prompt === "string" ? body.prompt.trim() : "";
    const systemInstruction =
      typeof body.systemInstruction === "string" ? body.systemInstruction.trim() : "";

    if (!prompt) {
      res.status(400).json({
        error: "invalid_request",
        message: "Prompt must be a non-empty string.",
      });
      return;
    }

    try {
      const ai = new GoogleGenAI({
        apiKey,
        httpOptions: {
          headers: {
            "User-Agent": "aistudio-build",
          },
        },
      });

      const response = await ai.models.generateContent({
        model: GEMINI_MODEL,
        contents: prompt,
        config: systemInstruction ? { systemInstruction } : undefined,
      });

      const text = (response.text ?? "").trim();
      if (!text) {
        res.status(502).json({
          error: "empty_gemini_response",
          message: "Gemini returned an empty response.",
          model: GEMINI_MODEL,
        });
        return;
      }

      res.json({
        model: GEMINI_MODEL,
        report: text,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      res.status(502).json({
        error: "gemini_api_error",
        message: `Gemini API call failed: ${message}`,
        model: GEMINI_MODEL,
      });
    }
  });

  const isProd = process.env.NODE_ENV === "production";
  const localDist = path.resolve(__dirname, "dist");
  const rootDist = path.resolve(__dirname, "..", "dist");
  const distDir = fs.existsSync(localDist) ? localDist : rootDist;

  if (isProd && fs.existsSync(path.join(distDir, "index.html"))) {
    app.use(express.static(distDir));
    app.get("/{*path}", (_req, res) => {
      res.sendFile(path.join(distDir, "index.html"));
    });
  } else {
    const vite = await createViteServer({
      configFile: path.resolve(__dirname, "vite.config.ts"),
      root: __dirname,
      server: {
        middlewareMode: true,
        hmr: process.env.DISABLE_HMR !== "true",
        watch: process.env.DISABLE_HMR === "true" ? null : {},
      },
      appType: "spa",
    });
    app.use(vite.middlewares);
  }

  const port = Number(process.env.PORT) || 3000;
  app.listen(port, "0.0.0.0", () => {
    console.log(`Industrial Equipment Monitoring dashboard running on http://0.0.0.0:${port}`);
  });
}

createAppServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
