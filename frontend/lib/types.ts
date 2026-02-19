// lib/types.ts
// Mirrors the PredictResponse Pydantic schema from backend/schemas/response.py

export interface DetectedVariant {
  rsid: string;
  chrom: string;
  pos: number;
  ref: string;
  alt: string;
  zygosity: string;
  star_allele: string | null;
  function_class: string | null;
  activity_score: number | null;
  km_score: number | null;
  mtd_methods: string[];
  ml_predicted: boolean;
  ml_confidence: number | null;
}

export interface RiskAssessment {
  risk_label: "Safe" | "Adjust Dosage" | "Toxic" | "Ineffective";
  confidence_score: number;
  severity: "low" | "moderate" | "high" | "critical";
}

export interface PharmacoGenomicProfile {
  primary_gene: string;
  diplotype: string;
  phenotype: string;
  phenotype_code: string;
  activity_score: number;
  detected_variants: DetectedVariant[];
}

export interface ClinicalRecommendation {
  guideline_source: string;
  recommendation: string;
}

export interface LLMExplanation {
  summary: string;
  model_used: string;
  rag_chunks_used: number;
}

export interface QualityMetrics {
  vcf_parsing_success: boolean;
  total_variants_parsed: number;
  pgx_variants_found: number;
  ml_predictions_made: number;
  rag_retrieval_success: boolean;
  llm_call_success: boolean;
}

export interface PredictResponse {
  patient_id: string;
  drug: string;
  timestamp: string;
  risk_assessment: RiskAssessment;
  pharmacogenomic_profile: PharmacoGenomicProfile;
  clinical_recommendation: ClinicalRecommendation;
  llm_generated_explanation: LLMExplanation;
  quality_metrics: QualityMetrics;
}
