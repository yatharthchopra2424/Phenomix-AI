"""
api/predict.py
==============
POST /api/predict
-----------------
Accepts a multipart/form-data request with:
  • vcf_file : UploadFile (.vcf, max 5 MB)
  • drugs    : str        (comma-separated drug names, e.g. "Codeine,Warfarin")

Orchestrates the full PharmaGuard pipeline:

  1. Save uploaded VCF to a temporary file
  2. Validate VCF header
  3. Parse VCF with vcf_parser.parse_vcf()
  4. Annotate variants against PGx reference table (annotator.annotate_variants)
  5. ML fallback for unrecognised variants in PGx gene regions (predictor)
  6. Build per-gene diplotypes (diplotype_mapper.build_diplotypes)
  7. Score phenotype (phenotype_engine.score_to_phenotype)
  8. Classify risk (risk_classifier.classify_risk)
  9. Retrieve CPIC context from ChromaDB (retriever.retrieve_context)
 10. Generate LLM explanation (llm_engine.generate_explanation)
 11. Serialise to strict PredictResponse JSON schema

One PredictResponse is returned per requested drug.
If multiple drugs are requested the endpoint returns a JSON array.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from genomics_engine.vcf_parser import parse_vcf, validate_vcf_header
from genomics_engine.annotator import annotate_variants, AnnotatedVariant
from genomics_engine.diplotype_mapper import build_diplotypes
from genomics_engine.phenotype_engine import score_to_phenotype, phenotype_to_code
from genomics_engine.risk_classifier import classify_risk
from genomics_engine.pgx_reference import DRUG_GENE_MAP

from ml_models.predictor import predict_variant_function

from rag_pipeline.retriever import retrieve_context
from rag_pipeline.llm_engine import generate_explanation

from backend.schemas.response import (
    PredictResponse, RiskAssessment, PharmacoGenomicProfile,
    ClinicalRecommendation, LLMExplanation, QualityMetrics, DetectedVariant,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/api/predict")
async def predict(
    vcf_file: UploadFile = File(..., description="VCF file (≤5 MB)"),
    drugs: str = Form(..., description="Comma-separated drug names"),
):
    """Run the full pharmacogenomic risk prediction pipeline."""

    # ── 1. Validate upload size ────────────────────────────────────────────
    content = await vcf_file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"VCF file exceeds 5 MB limit ({len(content) / 1e6:.2f} MB uploaded).",
        )

    # ── 2. Validate VCF header (first line must start with ##fileformat=VCF) ──
    first_line = content[:100].decode("utf-8", errors="replace").split("\n")[0]
    if not first_line.startswith("##fileformat=VCF"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid VCF file: missing ##fileformat=VCF header.",
        )

    # ── 3. Write to temp file for cyvcf2 / pure-Python parser ─────────────
    tmp_path: str = ""
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".vcf", delete=False, mode="wb"
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # ── 4. Parse VCF (blocking — offloaded to thread pool) ────────────
        variants = await asyncio.to_thread(parse_vcf, tmp_path)

    except Exception as exc:
        logger.error("VCF parsing failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"VCF parsing error: {exc}",
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    total_parsed = len(variants)

    # ── 5. Annotate variants ───────────────────────────────────────────────
    annotated = await asyncio.to_thread(annotate_variants, variants)
    pgx_found = sum(1 for a in annotated if a.gene and not a.needs_ml_prediction)

    # ── 6. ML fallback for unannotated PGx-region variants ────────────────
    ml_results: Dict = {}
    ml_count = 0
    needs_ml = [a for a in annotated if a.needs_ml_prediction]
    for av in needs_ml:
        key = (av.chrom, av.pos, av.ref, av.alt)
        func_class, confidence = await asyncio.to_thread(
            predict_variant_function, av.chrom, av.pos, av.ref, av.alt
        )
        ml_results[key] = (func_class, confidence)
        ml_count += 1

    # ── 7. Build diplotypes ────────────────────────────────────────────────
    diplotypes = await asyncio.to_thread(build_diplotypes, annotated, ml_results)

    # ── 8 & 9 & 10. Per-drug risk prediction, RAG, LLM ────────────────────
    drug_list = [d.strip().upper() for d in drugs.split(",") if d.strip()]
    if not drug_list:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No valid drug names provided.",
        )

    patient_id = f"PATIENT_{vcf_file.filename or uuid.uuid4().hex[:8].upper()}"
    results: List[PredictResponse] = []

    for drug in drug_list:
        gene = DRUG_GENE_MAP.get(drug)
        if not gene:
            logger.warning("Unsupported drug: %s — skipped.", drug)
            continue

        diplo_result = diplotypes.get(gene)
        if diplo_result is None:
            logger.warning("No diplotype result for gene %s.", gene)
            continue

        phenotype    = score_to_phenotype(gene, diplo_result.total_activity_score)
        pheno_code   = phenotype_to_code(phenotype)
        diplotype_str = diplo_result.diplotype_string

        # Risk classification
        risk_result = classify_risk(gene, phenotype, drug, diplotype_str)

        # Build detected variant list
        det_variants = _build_detected_variants(diplo_result.detected_variants, ml_results)
        rsids = [dv.rsid for dv in det_variants if dv.rsid and dv.rsid != "."]

        # RAG retrieval
        rag_ok = True
        context_chunks: List[str] = []
        try:
            context_chunks = await asyncio.to_thread(
                retrieve_context, gene, drug, phenotype, diplotype_str, rsids
            )
        except Exception as exc:
            logger.error("RAG retrieval failed: %s", exc)
            rag_ok = False

        # LLM explanation
        llm_ok = True
        llm_text = ""
        try:
            patient_profile = {
                "gene":       gene,
                "diplotype":  diplotype_str,
                "phenotype":  phenotype,
                "drug":       drug.title(),
                "risk_label": risk_result.risk_label,
                "rsids":      rsids,
            }
            llm_text = await asyncio.to_thread(
                generate_explanation, context_chunks, patient_profile
            )
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            llm_ok = False
            llm_text = (
                f"LLM unavailable. {risk_result.phenotype} {risk_result.risk_label} "
                f"for {drug.title()} (gene: {gene}, diplotype: {diplotype_str})."
            )

        # Serialise
        response = PredictResponse(
            patient_id = patient_id,
            drug       = drug.title(),
            risk_assessment = RiskAssessment(
                risk_label       = risk_result.risk_label,
                confidence_score = risk_result.confidence_score,
                severity         = risk_result.severity,
            ),
            pharmacogenomic_profile = PharmacoGenomicProfile(
                primary_gene       = gene,
                diplotype          = diplotype_str,
                phenotype          = phenotype,
                phenotype_code     = pheno_code,
                activity_score     = round(diplo_result.total_activity_score, 4),
                detected_variants  = det_variants,
            ),
            clinical_recommendation = ClinicalRecommendation(
                guideline_source = risk_result.guideline_source,
                recommendation   = risk_result.recommendation,
            ),
            llm_generated_explanation = LLMExplanation(
                summary         = llm_text,
                rag_chunks_used = len(context_chunks),
            ),
            quality_metrics = QualityMetrics(
                vcf_parsing_success    = True,
                total_variants_parsed  = total_parsed,
                pgx_variants_found     = pgx_found,
                ml_predictions_made    = ml_count,
                rag_retrieval_success  = rag_ok,
                llm_call_success       = llm_ok,
            ),
        )
        results.append(response)

    if not results:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No supported drugs were matched. Supported: Codeine, Warfarin, "
                   "Clopidogrel, Simvastatin, Azathioprine, Fluorouracil.",
        )

    # Return single object if one drug, array if multiple
    if len(results) == 1:
        return results[0].model_dump()

    return [r.model_dump() for r in results]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_detected_variants(
    annotated: List[AnnotatedVariant],
    ml_results: Dict,
) -> List[DetectedVariant]:
    dvs: List[DetectedVariant] = []
    for av in annotated:
        ml_predicted = av.needs_ml_prediction
        ml_conf = None
        if ml_predicted:
            key = (av.chrom, av.pos, av.ref, av.alt)
            ml_conf = ml_results.get(key, (None, None))[1]

        dvs.append(DetectedVariant(
            rsid           = av.rsid or av.variant_id or ".",
            chrom          = av.chrom,
            pos            = av.pos,
            ref            = av.ref,
            alt            = av.alt,
            zygosity       = av.zygosity,
            star_allele    = av.star_allele,
            function_class = av.function_class or av.ml_function_class,
            activity_score = av.activity_score,
            km_score       = av.km,
            mtd_methods    = av.mtd_methods,
            ml_predicted   = ml_predicted,
            ml_confidence  = ml_conf,
        ))
    return dvs
