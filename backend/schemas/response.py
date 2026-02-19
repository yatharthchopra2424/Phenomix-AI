"""
schemas/response.py
===================
Pydantic v2 models that exactly mirror the required JSON output schema
defined in the PharmaGuard architectural specification.

All fields are validated and serialised strictly; no malformed data
can propagate to the client interface.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class DetectedVariant(BaseModel):
    rsid: str
    chrom: str
    pos: int
    ref: str
    alt: str
    zygosity: str
    star_allele: Optional[str] = None
    function_class: Optional[str] = None
    activity_score: Optional[float] = None
    km_score: Optional[float] = Field(None, description="Minimum k-mer count QC metric")
    mtd_methods: List[str] = Field(default_factory=list)
    ml_predicted: bool = False
    ml_confidence: Optional[float] = None


class RiskAssessment(BaseModel):
    risk_label: str           # "Safe" | "Adjust Dosage" | "Toxic" | "Ineffective"
    confidence_score: float
    severity: str             # "low" | "moderate" | "high" | "critical"


class PharmacoGenomicProfile(BaseModel):
    primary_gene: str
    diplotype: str
    phenotype: str
    phenotype_code: str       # "PM" | "IM" | "NM" | "UM" | "PF" | "DF" | "NF"
    activity_score: float
    detected_variants: List[DetectedVariant] = Field(default_factory=list)


class ClinicalRecommendation(BaseModel):
    guideline_source: str
    recommendation: str


class LLMExplanation(BaseModel):
    summary: str
    model_used: str = "z-ai/glm5"
    rag_chunks_used: int = 0


class QualityMetrics(BaseModel):
    vcf_parsing_success: bool
    total_variants_parsed: int
    pgx_variants_found: int
    ml_predictions_made: int
    rag_retrieval_success: bool
    llm_call_success: bool


# ---------------------------------------------------------------------------
# Top-level response
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    patient_id: str
    drug: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    risk_assessment: RiskAssessment
    pharmacogenomic_profile: PharmacoGenomicProfile
    clinical_recommendation: ClinicalRecommendation
    llm_generated_explanation: LLMExplanation
    quality_metrics: QualityMetrics

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ensure_utc_string(cls, v: Any) -> str:
        if isinstance(v, datetime):
            return v.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        return str(v)


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
