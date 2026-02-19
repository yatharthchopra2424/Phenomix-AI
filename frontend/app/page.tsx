"use client";
// app/page.tsx — PharmaGuard main application page

import { useState } from "react";
import { Dna, Activity, ShieldAlert, Loader2, AlertCircle } from "lucide-react";
import { FileUpload } from "@/components/FileUpload/FileUpload";
import { DrugSelector } from "@/components/DrugSelector/DrugSelector";
import { ResultsPanel } from "@/components/ResultsPanel/ResultsPanel";
import { runPrediction, ApiError } from "@/lib/api";
import { PredictResponse } from "@/lib/types";

type AppState = "idle" | "loading" | "success" | "error";

export default function Home() {
  const [file, setFile]               = useState<File | null>(null);
  const [drugs, setDrugs]             = useState<string[]>([]);
  const [appState, setAppState]       = useState<AppState>("idle");
  const [results, setResults]         = useState<PredictResponse[]>([]);
  const [errorMsg, setErrorMsg]       = useState<string>("");
  const [errorDetail, setErrorDetail] = useState<string>("");

  const canSubmit = !!file && drugs.length > 0 && appState !== "loading";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || drugs.length === 0) return;

    setAppState("loading");
    setResults([]);
    setErrorMsg("");
    setErrorDetail("");

    try {
      const raw = await runPrediction(file, drugs);
      setResults(Array.isArray(raw) ? raw : [raw]);
      setAppState("success");
    } catch (err) {
      setErrorMsg(err instanceof ApiError ? err.message : "Unexpected error");
      setErrorDetail(err instanceof ApiError ? (err.detail ?? "") : String(err));
      setAppState("error");
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Nav */}
      <header className="sticky top-0 z-30 border-b border-slate-800 bg-slate-950/90 backdrop-blur-sm">
        <div className="mx-auto max-w-7xl flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-cyan-600">
              <Dna className="h-5 w-5 text-white" />
            </div>
            <div>
              <span className="font-bold text-white tracking-tight">PharmaGuard</span>
              <span className="ml-2 text-xs text-cyan-400 font-medium">Pharmacogenomic Risk AI</span>
            </div>
          </div>
          <div className="hidden sm:flex items-center gap-2 text-xs text-slate-500">
            <ShieldAlert className="h-3.5 w-3.5" />
            For research use only · Not for clinical diagnosis
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8">
        {/* Hero */}
        <div className="mb-8 text-center">
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white">
            Genomic Drug Safety Prediction
          </h1>
          <p className="mt-2 max-w-2xl mx-auto text-slate-400 text-sm leading-relaxed">
            Upload a patient VCF file and select target medications. PharmaGuard parses
            your genomic variants, calculates CPIC Activity Scores, and generates
            RAG-grounded clinical recommendations — powered by deep learning and NVIDIA GLM5.
          </p>
        </div>

        {/* Two-column layout */}
        <div className="grid grid-cols-1 lg:grid-cols-[400px_1fr] gap-6">

          {/* Left — form panel */}
          <aside className="space-y-4">
            <div className="rounded-2xl border border-slate-700 bg-slate-900 p-5 space-y-5">
              <div className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-cyan-400" />
                <h2 className="text-base font-bold text-white">Analysis Setup</h2>
              </div>

              <form onSubmit={handleSubmit} className="space-y-5">
                <div>
                  <label className="block text-xs font-semibold uppercase tracking-widest text-slate-400 mb-2">
                    VCF File
                  </label>
                  <FileUpload
                    onFileSelected={setFile}
                    onFileCleared={() => setFile(null)}
                    disabled={appState === "loading"}
                  />
                </div>

                <DrugSelector
                  selected={drugs}
                  onChange={setDrugs}
                  disabled={appState === "loading"}
                />

                <button
                  type="submit"
                  disabled={!canSubmit}
                  className={[
                    "w-full rounded-xl py-3 px-4 text-sm font-bold tracking-wide transition-all duration-200",
                    canSubmit
                      ? "bg-cyan-600 hover:bg-cyan-500 text-white shadow-lg shadow-cyan-900/40"
                      : "bg-slate-700 text-slate-500 cursor-not-allowed",
                  ].join(" ")}
                >
                  {appState === "loading" ? (
                    <span className="flex items-center justify-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Analysing genome…
                    </span>
                  ) : (
                    "Run Risk Prediction"
                  )}
                </button>
              </form>
            </div>

            {/* Gene-drug legend */}
            <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 p-4">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
                Gene–Drug Coverage
              </p>
              <ul className="space-y-1.5">
                {[
                  ["CYP2D6",  "Codeine",       "prodrug"],
                  ["CYP2C19", "Clopidogrel",   "prodrug"],
                  ["CYP2C9",  "Warfarin",      "active drug"],
                  ["SLCO1B1", "Simvastatin",   "transporter"],
                  ["TPMT",    "Azathioprine",  "active drug"],
                  ["DPYD",    "Fluorouracil",  "active drug"],
                ].map(([gene, drug, type]) => (
                  <li key={drug} className="flex items-center justify-between text-xs">
                    <span>
                      <span className="font-mono text-cyan-400">{gene}</span>
                      <span className="text-slate-500 mx-1">→</span>
                      <span className="text-slate-200">{drug}</span>
                    </span>
                    <span className="text-slate-600">{type}</span>
                  </li>
                ))}
              </ul>
            </div>
          </aside>

          {/* Right — results panel */}
          <section>
            {appState === "idle"    && <EmptyState />}
            {appState === "loading" && <LoadingState drugs={drugs} />}
            {appState === "error"   && <ErrorState message={errorMsg} detail={errorDetail} />}
            {appState === "success" && results.length > 0 && <ResultsPanel results={results} />}
          </section>

        </div>
      </main>

      <footer className="mt-16 border-t border-slate-800 py-6 text-center text-xs text-slate-600">
        PharmaGuard · Precision medicine AI · Next.js + FastAPI + ChromaDB + NVIDIA GLM5
      </footer>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-64 rounded-2xl border border-dashed border-slate-700 text-center p-8">
      <Dna className="h-12 w-12 text-slate-700 mb-4" />
      <p className="text-slate-400 font-medium">No results yet</p>
      <p className="text-slate-600 text-sm mt-1">Upload a VCF file and select drugs to begin.</p>
    </div>
  );
}

function LoadingState({ drugs }: { drugs: string[] }) {
  return (
    <div className="flex flex-col items-center justify-center h-64 rounded-2xl border border-slate-700 bg-slate-900/50 text-center p-8 space-y-4">
      <div className="relative">
        <Dna className="h-12 w-12 text-cyan-700" />
        <Loader2 className="absolute -top-1 -right-1 h-5 w-5 text-cyan-400 animate-spin" />
      </div>
      <div>
        <p className="text-white font-semibold">Analysing genomic profile…</p>
        <p className="text-slate-400 text-sm mt-1">
          Parsing VCF · Annotating variants · CPIC mapping · RAG retrieval · LLM explanation
        </p>
        {drugs.length > 0 && (
          <p className="text-slate-500 text-xs mt-2">Drugs: {drugs.join(", ")}</p>
        )}
      </div>
    </div>
  );
}

function ErrorState({ message, detail }: { message: string; detail: string }) {
  return (
    <div className="rounded-2xl border border-red-800 bg-red-950/30 p-6">
      <div className="flex gap-3">
        <AlertCircle className="h-6 w-6 text-red-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="font-semibold text-red-300">{message}</p>
          {detail && <p className="text-red-400/70 text-sm whitespace-pre-wrap break-words">{detail}</p>}
        </div>
      </div>
    </div>
  );
}
