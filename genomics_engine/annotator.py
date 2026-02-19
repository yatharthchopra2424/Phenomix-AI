"""
annotator.py
============
Annotate parsed VCF variants with pharmacogenomic metadata by performing
(CHROM, POS, REF, ALT) coordinate-based lookups against the PGX_VARIANTS
reference table (hg38).

Variants not found in the reference table are flagged for ML-based
functional prediction (handled downstream by ml_models.predictor).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from genomics_engine.vcf_parser import ParsedVariant
from genomics_engine.pgx_reference import PGX_VARIANTS, TARGET_GENES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedVariant:
    """A parsed variant enriched with pharmacogenomic annotation."""

    # Original VCF fields
    chrom: str
    pos: int
    ref: str
    alt: str
    variant_id: str
    gt_raw: str
    phased: bool
    allele1: int   # 0=ref, 1=alt
    allele2: int
    zygosity: str  # "hom_ref" | "het" | "hom_alt" | "missing"

    # QC metrics
    km: Optional[float]
    kfp: Optional[int]
    kff: Optional[int]
    mtd_methods: List[str]

    # PGx annotation (populated if found in reference table)
    gene: Optional[str]        = None
    star_allele: Optional[str] = None
    rsid: Optional[str]        = None
    function_class: Optional[str] = None   # "normal_function" | "decreased_function" | …
    activity_score: Optional[float] = None

    # ML fallback flags
    needs_ml_prediction: bool = False
    ml_function_class: Optional[str] = None
    ml_confidence: Optional[float] = None

    # Computed — which chromosomal copies carry the ALT allele?
    haplotype_alt_copies: List[int] = field(default_factory=list)  # 0-indexed haplotype indices


# ---------------------------------------------------------------------------
# Core annotation function
# ---------------------------------------------------------------------------

def annotate_variants(variants: List[ParsedVariant]) -> List[AnnotatedVariant]:
    """
    Cross-reference each ParsedVariant against PGX_VARIANTS.

    Returns a list of AnnotatedVariant objects.  Variants that belong to a
    target pharmacogene but are *not* in the reference table are marked with
    needs_ml_prediction=True for downstream deep-learning classification.
    """
    annotated: List[AnnotatedVariant] = []
    found_genes: set = set()
    pgx_hit_count = 0
    ml_fallback_count = 0

    for v in variants:
        # Skip purely homozygous-reference calls (no variant present)
        if v.zygosity == "hom_ref":
            continue

        # Build lookup key
        key = (v.chrom, v.pos, v.ref, v.alt)
        pgx = PGX_VARIANTS.get(key)

        # Determine which haplotype copies carry the ALT allele
        alt_copies: List[int] = []
        if v.allele1 == 1:
            alt_copies.append(0)  # haplotype index 0 (maternal/paternal copy 1)
        if v.allele2 == 1:
            alt_copies.append(1)  # haplotype index 1

        av = AnnotatedVariant(
            chrom=v.chrom, pos=v.pos, ref=v.ref, alt=v.alt,
            variant_id=v.variant_id,
            gt_raw=v.gt_raw, phased=v.phased,
            allele1=v.allele1, allele2=v.allele2,
            zygosity=v.zygosity,
            km=v.km, kfp=v.kfp, kff=v.kff,
            mtd_methods=v.mtd_methods,
            haplotype_alt_copies=alt_copies,
        )

        if pgx is not None:
            # Known PGx variant — populate annotation
            av.gene           = pgx.gene
            av.star_allele    = pgx.star
            av.rsid           = pgx.rsid
            av.function_class = pgx.function
            av.activity_score = pgx.activity
            found_genes.add(pgx.gene)
            pgx_hit_count += 1
        else:
            # Not in the static table — check if this position might be in a
            # pharmacogene region (broad positional filter) and flag for ML
            gene_region = _belongs_to_pgx_region(v.chrom, v.pos)
            if gene_region:
                av.gene = gene_region
                av.needs_ml_prediction = True
                av.rsid = v.variant_id if v.variant_id != "." else None
                ml_fallback_count += 1

        if av.gene:
            annotated.append(av)

    logger.info(
        "Annotation complete — %d PGx hits, %d ML fallback, genes found: %s",
        pgx_hit_count, ml_fallback_count, sorted(found_genes),
    )

    return annotated


# ---------------------------------------------------------------------------
# Broad positional filter: is this variant in a known PGx gene region?
# Coordinates are approximate exon-level ranges on hg38.
# ---------------------------------------------------------------------------

_GENE_REGIONS = {
    "CYP2D6":  ("chr22", 42_512_000, 42_530_000),
    "CYP2C19": ("chr10", 94_762_000, 94_855_000),
    "CYP2C9":  ("chr10", 96_698_000, 96_749_000),
    "SLCO1B1": ("chr12", 21_131_000, 21_239_000),
    "TPMT":    ("chr6",  18_126_000, 18_157_000),
    "DPYD":    ("chr1",  97_543_000, 98_387_000),
}


def _belongs_to_pgx_region(chrom: str, pos: int) -> Optional[str]:
    """Return gene symbol if (chrom, pos) falls inside a known PGx gene window."""
    for gene, (g_chrom, g_start, g_end) in _GENE_REGIONS.items():
        if chrom == g_chrom and g_start <= pos <= g_end:
            return gene
    return None


# ---------------------------------------------------------------------------
# Convenience helper: filter annotated variants by gene
# ---------------------------------------------------------------------------

def filter_by_gene(annotated: List[AnnotatedVariant], gene: str) -> List[AnnotatedVariant]:
    return [v for v in annotated if v.gene == gene]
