"use client";
// components/ResultsPanel/ResultsPanel.tsx
// Main results display for pharmacogenomic risk prediction

import { useState } from "react";
import { Download, Copy, CheckCheck, Clock, Beaker, Dna, AlertTriangle, BookOpen, Cpu } from "lucide-react";
import { PredictResponse } from "@/lib/types";
import { RiskBadge } from "@/components/RiskBadge/RiskBadge";
import { AccordionSection } from "@/components/AccordionSection/AccordionSection";

interface ResultsPanelProps {
  results: PredictResponse[];
}

export function ResultsPanel({ results }: ResultsPanelProps) {
  const [copied, setCopied] = useState(false);

  const handleDownload = () => {
    const payload = results.length === 1 ? results[0] : results;
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pharmaguard_${results[0]?.patient_id ?? "result"}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleCopy = async () => {
    const payload = results.length === 1 ? results[0] : results;
    await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  return (
    <div className="space-y-6">
      {/* Header actions */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h2 className="text-lg font-bold text-white">Risk Assessment Results</h2>
        <div className="flex gap-2">
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 rounded-lg border border-slate-600 px-3 py-1.5 text-xs text-slate-300 hover:border-cyan-600 hover:text-cyan-300 transition-colors"
          >
            {copied ? (
              <><CheckCheck className="h-3.5 w-3.5 text-emerald-400" /> Copied!</>
            ) : (
              <><Copy className="h-3.5 w-3.5" /> Copy JSON</>
            )}
          </button>
          <button
            onClick={handleDownload}
            className="flex items-center gap-1.5 rounded-lg border border-slate-600 px-3 py-1.5 text-xs text-slate-300 hover:border-cyan-600 hover:text-cyan-300 transition-colors"
          >
            <Download className="h-3.5 w-3.5" /> Download JSON
          </button>
        </div>
      </div>

      {/* One card per drug result */}
      {results.map((result, idx) => (
        <ResultCard key={`${result.drug}-${idx}`} result={result} />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Individual result card
// ---------------------------------------------------------------------------

function ResultCard({ result }: { result: PredictResponse }) {
  const {
    patient_id, drug, timestamp,
    risk_assessment,
    pharmacogenomic_profile: pgx,
    clinical_recommendation,
    llm_generated_explanation,
    quality_metrics,
  } = result;

  const detectedRsids = pgx.detected_variants
    .map((v) => v.rsid)
    .filter((r) => r && r !== ".");

  const riskBorder: Record<string, string> = {
    Safe:           "border-emerald-800",
    "Adjust Dosage":"border-amber-700",
    Toxic:          "border-red-800",
    Ineffective:    "border-red-800",
  };

  return (
    <div
      className={[
        "rounded-2xl border bg-slate-900/80 overflow-hidden",
        riskBorder[risk_assessment.risk_label] ?? "border-slate-700",
      ].join(" ")}
    >
      {/* Card header */}
      <div className="flex items-start justify-between gap-4 px-5 py-4 bg-slate-800/60 border-b border-slate-700">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-lg font-bold text-white">{drug}</span>
            <RiskBadge
              riskLabel={risk_assessment.risk_label}
              severity={risk_assessment.severity}
              large
            />
          </div>
          <p className="mt-1 text-xs text-slate-400 flex items-center gap-1">
            <Clock className="h-3 w-3" />
            {new Date(timestamp).toLocaleString()} · {patient_id}
          </p>
        </div>
        <div className="text-right flex-shrink-0">
          <p className="text-xs text-slate-500">Confidence</p>
          <p className="text-2xl font-bold text-white">
            {(risk_assessment.confidence_score * 100).toFixed(0)}
            <span className="text-sm text-slate-400">%</span>
          </p>
        </div>
      </div>

      {/* Genomic profile strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-px bg-slate-700 border-b border-slate-700">
        {[
          { label: "Gene",         value: pgx.primary_gene },
          { label: "Diplotype",    value: pgx.diplotype },
          { label: "Phenotype",    value: pgx.phenotype_code },
          { label: "Activity Score", value: pgx.activity_score.toFixed(2) },
        ].map(({ label, value }) => (
          <div key={label} className="flex flex-col px-4 py-3 bg-slate-900/80">
            <span className="text-xs text-slate-500 uppercase tracking-wide">{label}</span>
            <span className="text-sm font-bold text-white font-mono">{value}</span>
          </div>
        ))}
      </div>

      {/* Accordion sections */}
      <div className="p-4 space-y-2">

        {/* Clinical recommendation */}
        <AccordionSection
          title="Clinical Recommendation"
          badge={
            <span className="text-xs rounded bg-cyan-900/50 text-cyan-400 px-2 py-0.5 border border-cyan-800">
              {clinical_recommendation.guideline_source}
            </span>
          }
          defaultOpen
        >
          <div className="flex gap-2">
            <AlertTriangle className={[
              "h-4 w-4 flex-shrink-0 mt-0.5",
              risk_assessment.risk_label === "Safe"
                ? "text-emerald-400"
                : risk_assessment.severity === "critical"
                ? "text-red-400"
                : "text-amber-400",
            ].join(" ")} />
            <p className="leading-relaxed">{clinical_recommendation.recommendation}</p>
          </div>
        </AccordionSection>

        {/* LLM explanation */}
        <AccordionSection
          title="AI-Generated Clinical Explanation"
          badge={
            <span className="text-xs rounded bg-violet-900/50 text-violet-400 px-2 py-0.5 border border-violet-800">
              RAG · {llm_generated_explanation.rag_chunks_used} chunks
            </span>
          }
          defaultOpen={false}
        >
          <div className="flex gap-2">
            <BookOpen className="h-4 w-4 flex-shrink-0 mt-0.5 text-violet-400" />
            <p className="whitespace-pre-wrap leading-relaxed">
              {llm_generated_explanation.summary}
            </p>
          </div>
          <p className="text-xs text-slate-500 mt-2">
            Model: {llm_generated_explanation.model_used}
          </p>
        </AccordionSection>

        {/* Detected variants */}
        <AccordionSection
          title="Detected PGx Variants"
          badge={
            <span className="text-xs rounded bg-slate-800 text-slate-400 px-2 py-0.5 border border-slate-700">
              {pgx.detected_variants.length} variant{pgx.detected_variants.length !== 1 ? "s" : ""}
            </span>
          }
          defaultOpen={false}
        >
          {pgx.detected_variants.length === 0 ? (
            <p className="text-slate-400 italic">
              No pharmacogenomically relevant variants detected at known positions.
              Wild-type alleles assumed.
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-500 border-b border-slate-700">
                    {["rsID", "Position", "REF/ALT", "Star Allele", "Function", "AS", "Zygosity", "ML"].map((h) => (
                      <th key={h} className="text-left pb-2 pr-4 font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {pgx.detected_variants.map((v, i) => (
                    <tr key={i} className="border-b border-slate-800/50 hover:bg-slate-800/30">
                      <td className="py-1.5 pr-4 font-mono text-cyan-400">{v.rsid}</td>
                      <td className="py-1.5 pr-4 font-mono text-slate-300">
                        {v.chrom}:{v.pos}
                      </td>
                      <td className="py-1.5 pr-4 font-mono">
                        <span className="text-slate-400">{v.ref}</span>
                        <span className="text-slate-500">/</span>
                        <span className="text-amber-300">{v.alt}</span>
                      </td>
                      <td className="py-1.5 pr-4 text-violet-300 font-mono">{v.star_allele ?? "—"}</td>
                      <td className="py-1.5 pr-4">
                        <FunctionPill func={v.function_class} />
                      </td>
                      <td className="py-1.5 pr-4 font-mono">
                        {v.activity_score !== null ? v.activity_score?.toFixed(2) : "—"}
                      </td>
                      <td className="py-1.5 pr-4 capitalize text-slate-300">{v.zygosity}</td>
                      <td className="py-1.5">
                        {v.ml_predicted ? (
                          <span className="flex items-center gap-1 text-yellow-400">
                            <Cpu className="h-3 w-3" />
                            {v.ml_confidence ? `${(v.ml_confidence * 100).toFixed(0)}%` : "demo"}
                          </span>
                        ) : (
                          <span className="text-slate-500">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </AccordionSection>

        {/* Quality metrics */}
        <AccordionSection title="Quality Metrics" defaultOpen={false}>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {[
              { label: "VCF Parsed",       value: quality_metrics.vcf_parsing_success, bool: true },
              { label: "Total Variants",   value: quality_metrics.total_variants_parsed, bool: false },
              { label: "PGx Hits",         value: quality_metrics.pgx_variants_found, bool: false },
              { label: "ML Predictions",   value: quality_metrics.ml_predictions_made, bool: false },
              { label: "RAG Success",      value: quality_metrics.rag_retrieval_success, bool: true },
              { label: "LLM Success",      value: quality_metrics.llm_call_success, bool: true },
            ].map(({ label, value, bool }) => (
              <div key={label} className="rounded-lg bg-slate-800 px-3 py-2">
                <p className="text-xs text-slate-500">{label}</p>
                <p className={(bool
                  ? (value ? "text-emerald-400" : "text-red-400")
                  : "text-white") + " font-bold text-sm"}>
                  {bool ? (value ? "✓ Yes" : "✗ No") : String(value)}
                </p>
              </div>
            ))}
          </div>
        </AccordionSection>

      </div>
    </div>
  );
}


// ---------------------------------------------------------------------------
// Small helper: function class pill
// ---------------------------------------------------------------------------
function FunctionPill({ func }: { func: string | null }) {
  const map: Record<string, string> = {
    normal_function:    "bg-emerald-900/60 text-emerald-300 border-emerald-700",
    decreased_function: "bg-amber-900/60 text-amber-300 border-amber-700",
    increased_function: "bg-blue-900/60 text-blue-300 border-blue-700",
    no_function:        "bg-red-900/60 text-red-300 border-red-700",
  };
  if (!func) return <span className="text-slate-500">—</span>;
  const style = map[func] ?? "bg-slate-800 text-slate-400 border-slate-600";
  return (
    <span className={`inline-block rounded border px-1.5 py-0.5 text-xs font-mono ${style}`}>
      {func.replace("_", " ")}
    </span>
  );
}
