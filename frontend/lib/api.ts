// lib/api.ts
// Typed fetch wrapper for the PharmaGuard FastAPI backend

import { PredictResponse } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public detail?: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * POST /api/predict
 * Upload a VCF file and list of drugs, receive structured risk prediction.
 */
export async function runPrediction(
  file: File,
  drugs: string[]
): Promise<PredictResponse | PredictResponse[]> {
  const formData = new FormData();
  formData.append("vcf_file", file);
  formData.append("drugs", drugs.join(","));

  const response = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let detail: string | undefined;
    try {
      const errBody = await response.json();
      detail = errBody?.detail ?? JSON.stringify(errBody);
    } catch {
      detail = await response.text();
    }
    throw new ApiError(response.status, `Prediction failed (${response.status})`, detail);
  }

  return response.json() as Promise<PredictResponse | PredictResponse[]>;
}

/**
 * GET /api/health
 * Liveness and readiness check.
 */
export async function checkHealth(): Promise<Record<string, unknown>> {
  const response = await fetch(`${API_BASE}/api/health`);
  if (!response.ok) throw new ApiError(response.status, "Health check failed");
  return response.json();
}
