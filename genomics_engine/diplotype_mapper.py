"""
diplotype_mapper.py
===================
Aggregate per-gene annotated variants into diplotypes.

A diplotype is the combination of the maternal and paternal star alleles,
e.g. CYP2D6 *1/*4.  Because NA12877.vcf is fully phased (pipe '|' separator),
we can directly assign ALT alleles to haplotype-0 (left of '|') or
haplotype-1 (right of '|') without statistical phasing.

Algorithm:
  1. Group AnnotatedVariant objects by gene.
  2. For each gene, collect all allele assignments for haplotype-0 and haplotype-1.
  3. Pick the highest-impact (lowest activity-score) star allele for each haplotype.
  4. If no variant was detected on a haplotype, assign the default wild-type allele (*1).
  5. Sum the activity scores of the two haplotypes → total_activity_score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from genomics_engine.annotator import AnnotatedVariant
from genomics_engine.pgx_reference import (
    GENE_DEFAULT_ALLELE,
    TARGET_GENES,
    PGxVariant,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class HaplotypeCall:
    """A single haplotype assignment for one gene."""
    star: str
    function_class: str
    activity: float
    rsids: List[str] = field(default_factory=list)
    source: str = "reference"   # "reference" | "ml_prediction" | "default"


@dataclass
class DiplotypeResult:
    gene: str
    haplotype_0: HaplotypeCall    # copy from allele1 (left of '|')
    haplotype_1: HaplotypeCall    # copy from allele2 (right of '|')
    total_activity_score: float
    diplotype_string: str         # e.g. "*1/*4"
    detected_variants: List[AnnotatedVariant] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def build_diplotypes(
    annotated: List[AnnotatedVariant],
    ml_results: Optional[Dict[Tuple, Tuple[str, float]]] = None,
) -> Dict[str, DiplotypeResult]:
    """
    Build a diplotype for every target gene.

    Parameters
    ----------
    annotated   : list of AnnotatedVariant returned by annotator.annotate_variants()
    ml_results  : dict keyed by (chrom, pos, ref, alt) → (function_class, confidence)
                  for variants that were sent to the DL model.

    Returns
    -------
    dict mapping gene symbol → DiplotypeResult
    """
    ml_results = ml_results or {}

    # Group variants by gene
    gene_variants: Dict[str, List[AnnotatedVariant]] = {g: [] for g in TARGET_GENES}
    for av in annotated:
        if av.gene in gene_variants:
            gene_variants[av.gene].append(av)

    diplotypes: Dict[str, DiplotypeResult] = {}

    for gene in TARGET_GENES:
        variants = gene_variants[gene]
        default = GENE_DEFAULT_ALLELE[gene]

        # Haplotype bins: index → list of PGxVariant-like callss
        hap_bins: Dict[int, List[HaplotypeCall]] = {0: [], 1: []}

        for av in variants:
            # Resolve function class: prefer static table, fall back to ML result
            func  = av.function_class
            score = av.activity_score
            src   = "reference"
            rsid  = av.rsid or av.variant_id or "."

            if av.needs_ml_prediction:
                key = (av.chrom, av.pos, av.ref, av.alt)
                if key in ml_results:
                    func, confidence = ml_results[key]
                    score = _function_to_default_score(func)
                    av.ml_function_class = func
                    av.ml_confidence     = confidence
                    src = "ml_prediction"
                else:
                    # ML model not available — treat as normal for this haplotype
                    func  = "normal_function"
                    score = 1.0
                    src   = "default"

            if func is None:
                func  = "normal_function"
                score = 1.0
                src   = "default"

            call = HaplotypeCall(
                star           = av.star_allele or "novel",
                function_class = func,
                activity       = float(score) if score is not None else 1.0,
                rsids          = [rsid] if rsid != "." else [],
                source         = src,
            )

            for hap_idx in av.haplotype_alt_copies:
                if hap_idx in hap_bins:
                    hap_bins[hap_idx].append(call)

        # Resolve each haplotype — pick the most deleterious allele (lowest score)
        hap_0 = _resolve_haplotype(hap_bins[0], default)
        hap_1 = _resolve_haplotype(hap_bins[1], default)

        total_as = round(hap_0.activity + hap_1.activity, 4)
        diplotype_str = f"{hap_0.star}/{hap_1.star}"

        diplotypes[gene] = DiplotypeResult(
            gene               = gene,
            haplotype_0        = hap_0,
            haplotype_1        = hap_1,
            total_activity_score = total_as,
            diplotype_string   = diplotype_str,
            detected_variants  = variants,
        )

        logger.debug(
            "Gene %s → diplotype %s (AS=%.2f)",
            gene, diplotype_str, total_as,
        )

    return diplotypes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_haplotype(
    calls: List[HaplotypeCall],
    default: PGxVariant,
) -> HaplotypeCall:
    """
    From a list of HaplotypeCalls on one chromosomal copy, pick the most
    clinically impactful one (lowest activity score / most deleterious).
    If no ALT calls were assigned, return the wild-type default allele.
    """
    if not calls:
        return HaplotypeCall(
            star           = default.star,
            function_class = default.function,
            activity       = default.activity,
            source         = "default",
        )
    # Sort ascending by activity (lowest = most deleterious wins)
    calls_sorted = sorted(calls, key=lambda c: c.activity)
    return calls_sorted[0]


_FUNCTION_SCORE_MAP = {
    "no_function":       0.0,
    "decreased_function": 0.5,
    "normal_function":   1.0,
    "increased_function": 1.5,
}


def _function_to_default_score(func: str) -> float:
    """Map a function class string to its default CPIC activity score."""
    return _FUNCTION_SCORE_MAP.get(func, 1.0)
