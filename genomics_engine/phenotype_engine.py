"""
phenotype_engine.py
===================
Translate a gene + total Activity Score into a standardized CPIC metabolic phenotype.

CPIC Activity Score thresholds (per-gene, sourced from CPIC v4 guidelines):

  CYP2D6 / CYP2C19 / CYP2C9 / DPYD (metabolic enzymes):
    > 2.0   → Ultra-Rapid Metabolizer (UM)
    1.5–2.0 → Normal Metabolizer (NM)
    0.5–1.4 → Intermediate Metabolizer (IM)
    < 0.5   → Poor Metabolizer (PM)

  SLCO1B1 (transporter — activity score maps to transport function):
    Score 2 (both alleles normal) → Normal Function
    Score 1 (one decreased allele) → Decreased Function
    Score 0 (both alleles impaired) → Poor Function

  TPMT (thiopurine methyltransferase):
    AS ≥ 2.0 → Normal Metabolizer
    AS 1.0–1.9 → Intermediate Metabolizer
    AS < 1.0 → Poor Metabolizer
"""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phenotype string constants
# ---------------------------------------------------------------------------
PHENOTYPE_UM  = "Ultra-Rapid Metabolizer"
PHENOTYPE_NM  = "Normal Metabolizer"
PHENOTYPE_IM  = "Intermediate Metabolizer"
PHENOTYPE_PM  = "Poor Metabolizer"
PHENOTYPE_NF  = "Normal Function"          # SLCO1B1
PHENOTYPE_DF  = "Decreased Function"       # SLCO1B1
PHENOTYPE_PF  = "Poor Function"            # SLCO1B1


# ---------------------------------------------------------------------------
# Per-gene phenotype classification
# ---------------------------------------------------------------------------

def score_to_phenotype(gene: str, total_activity_score: float) -> str:
    """
    Translate a gene-specific CPIC activity score to a metabolic phenotype string.

    Parameters
    ----------
    gene                 : pharmacogene symbol (e.g. "CYP2D6")
    total_activity_score : sum of both haplotype activity scores

    Returns
    -------
    Standardized phenotype string e.g. "Poor Metabolizer"
    """
    as_ = round(total_activity_score, 4)

    if gene == "SLCO1B1":
        return _slco1b1_phenotype(as_)
    if gene == "TPMT":
        return _tpmt_phenotype(as_)
    # CYP2D6, CYP2C19, CYP2C9, DPYD — standard CPIC enzyme phenotype
    return _standard_enzyme_phenotype(as_)


def _standard_enzyme_phenotype(as_: float) -> str:
    if as_ > 2.0:
        return PHENOTYPE_UM
    if as_ >= 1.5:
        return PHENOTYPE_NM
    if as_ >= 0.5:
        return PHENOTYPE_IM
    return PHENOTYPE_PM


def _slco1b1_phenotype(as_: float) -> str:
    """
    SLCO1B1 activity score maps to transport function rather than
    metabolic rate.  CPIC uses a simpler 3-tier classification.
    """
    if as_ >= 2.0:
        return PHENOTYPE_NF
    if as_ >= 1.0:
        return PHENOTYPE_DF
    return PHENOTYPE_PF


def _tpmt_phenotype(as_: float) -> str:
    if as_ >= 2.0:
        return PHENOTYPE_NM
    if as_ >= 1.0:
        return PHENOTYPE_IM
    return PHENOTYPE_PM


# ---------------------------------------------------------------------------
# Short phenotype code used in JSON output
# ---------------------------------------------------------------------------

PHENOTYPE_CODE: Dict[str, str] = {
    PHENOTYPE_UM: "UM",
    PHENOTYPE_NM: "NM",
    PHENOTYPE_IM: "IM",
    PHENOTYPE_PM: "PM",
    PHENOTYPE_NF: "NF",
    PHENOTYPE_DF: "DF",
    PHENOTYPE_PF: "PF",
}


def phenotype_to_code(phenotype: str) -> str:
    """Return short phenotype code e.g. 'PM'. Falls back to the full string."""
    return PHENOTYPE_CODE.get(phenotype, phenotype)
