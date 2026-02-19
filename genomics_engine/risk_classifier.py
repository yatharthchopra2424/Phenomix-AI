"""
risk_classifier.py
==================
Map (gene, phenotype, drug) → a structured RiskResult.

Encodes all six CPIC gene-drug interaction rules described in the
PharmaGuard architectural spec.  The logic differentiates between:

  • Prodrugs (require enzymatic ACTIVATION) — Codeine, Clopidogrel
      - Ultra-Rapid Metabolizer → Toxic (over-activation)
      - Poor Metabolizer        → Ineffective (cannot activate)
      - Intermediate Metabolizer→ Adjust Dosage
      - Normal Metabolizer      → Safe

  • Active drugs (require enzymatic CLEARANCE) — Warfarin, Azathioprine, Fluorouracil
      - Poor Metabolizer        → Toxic (accumulation)
      - Intermediate Metabolizer→ Adjust Dosage
      - Normal Metabolizer      → Safe
      - Ultra-Rapid Metabolizer → Ineffective (cleared too fast)

  • Transporters (require hepatic UPTAKE) — Simvastatin via SLCO1B1
      - Poor / Decreased Function → Toxic (drug stays in circulation)
      - Normal Function           → Safe
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from genomics_engine.phenotype_engine import (
    PHENOTYPE_UM, PHENOTYPE_NM, PHENOTYPE_IM, PHENOTYPE_PM,
    PHENOTYPE_NF, PHENOTYPE_DF, PHENOTYPE_PF,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk label constants
# ---------------------------------------------------------------------------
RISK_SAFE    = "Safe"
RISK_ADJUST  = "Adjust Dosage"
RISK_TOXIC   = "Toxic"
RISK_INEFFECTIVE = "Ineffective"

SEVERITY_LEVEL: Dict[str, str] = {
    RISK_SAFE:        "low",
    RISK_ADJUST:      "moderate",
    RISK_TOXIC:       "critical",
    RISK_INEFFECTIVE: "high",
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RiskResult:
    risk_label: str
    severity: str
    confidence_score: float
    guideline_source: str
    recommendation: str
    primary_gene: str
    phenotype: str
    diplotype: str
    drug: str


# ---------------------------------------------------------------------------
# Gene-drug interaction rules
# ---------------------------------------------------------------------------

# Each rule is (risk_label, confidence, recommendation_text)
_Rule = Tuple[str, float, str]

_RULES: Dict[
    Tuple[str, str, str],   # (gene, drug_upper, phenotype)
    _Rule
] = {
    # ── CYP2D6 / CODEINE (prodrug) ──────────────────────────────────────────
    ("CYP2D6", "CODEINE", PHENOTYPE_UM): (
        RISK_TOXIC, 0.97,
        "Avoid codeine. Ultra-rapid CYP2D6 metabolizers convert codeine to morphine "
        "at an accelerated rate, risking life-threatening respiratory depression. "
        "Use alternative analgesics (e.g. non-opioid agents or morphine with dose monitoring)."
    ),
    ("CYP2D6", "CODEINE", PHENOTYPE_NM): (
        RISK_SAFE, 0.95,
        "Standard codeine dosing is appropriate. Monitor for standard opioid side-effects."
    ),
    ("CYP2D6", "CODEINE", PHENOTYPE_IM): (
        RISK_ADJUST, 0.90,
        "Consider reducing codeine dose or substituting a non-CYP2D6-dependent analgesic. "
        "Reduced CYP2D6 activity may result in diminished analgesia at standard doses."
    ),
    ("CYP2D6", "CODEINE", PHENOTYPE_PM): (
        RISK_INEFFECTIVE, 0.97,
        "Avoid codeine. Poor CYP2D6 metabolizers cannot convert codeine to its active "
        "metabolite morphine. Expected analgesic failure. Prescribe a non-prodrug opioid."
    ),

    # ── CYP2C19 / CLOPIDOGREL (prodrug) ─────────────────────────────────────
    ("CYP2C19", "CLOPIDOGREL", PHENOTYPE_UM): (
        RISK_ADJUST, 0.85,
        "Standard clopidogrel dose is likely adequate. Consider monitoring platelet "
        "reactivity if standard therapy appears insufficient."
    ),
    ("CYP2C19", "CLOPIDOGREL", PHENOTYPE_NM): (
        RISK_SAFE, 0.95,
        "Standard clopidogrel dosing is appropriate per CPIC guidelines."
    ),
    ("CYP2C19", "CLOPIDOGREL", PHENOTYPE_IM): (
        RISK_INEFFECTIVE, 0.88,
        "Reduced clopidogrel activation. Consider alternative antiplatelet therapy "
        "(e.g. prasugrel, ticagrelor) to prevent stent thrombosis or cardiovascular events."
    ),
    ("CYP2C19", "CLOPIDOGREL", PHENOTYPE_PM): (
        RISK_INEFFECTIVE, 0.97,
        "Avoid clopidogrel. Poor CYP2C19 metabolizers cannot adequately activate the "
        "prodrug. Risk of fatal stent thrombosis or stroke. Prescribe prasugrel or "
        "ticagrelor per CPIC guidelines."
    ),

    # ── CYP2C9 / WARFARIN (active drug) ─────────────────────────────────────
    ("CYP2C9", "WARFARIN", PHENOTYPE_UM): (
        RISK_ADJUST, 0.85,
        "Warfarin may be cleared rapidly. Consider higher initial dose and close INR "
        "monitoring. Standard CPIC dosing algorithm with CYP2C9 genotype adjustment is "
        "recommended."
    ),
    ("CYP2C9", "WARFARIN", PHENOTYPE_NM): (
        RISK_SAFE, 0.95,
        "Standard warfarin dosing algorithm is appropriate. Routine INR monitoring "
        "per clinical guidelines."
    ),
    ("CYP2C9", "WARFARIN", PHENOTYPE_IM): (
        RISK_ADJUST, 0.90,
        "Reduce warfarin starting dose by 25–50% (CPIC guideline). CYP2C9 "
        "intermediate metabolizers have reduced clearance. Close INR monitoring required "
        "during initiation."
    ),
    ("CYP2C9", "WARFARIN", PHENOTYPE_PM): (
        RISK_TOXIC, 0.95,
        "Reduce warfarin starting dose significantly (≥50% CPIC recommendation). "
        "Poor CYP2C9 metabolizers clear warfarin very slowly, leading to dangerous "
        "systemic accumulation and severe hemorrhagic risk. Intensive INR monitoring is "
        "mandatory."
    ),

    # ── SLCO1B1 / SIMVASTATIN (transporter substrate) ───────────────────────
    ("SLCO1B1", "SIMVASTATIN", PHENOTYPE_NF): (
        RISK_SAFE, 0.94,
        "Standard simvastatin dose is appropriate. Normal SLCO1B1 transport function."
    ),
    ("SLCO1B1", "SIMVASTATIN", PHENOTYPE_DF): (
        RISK_TOXIC, 0.90,
        "Avoid high-dose simvastatin (≥40 mg/day). Reduced SLCO1B1 transport "
        "function increases systemic statin exposure, significantly elevating risk of "
        "statin-induced myopathy or rhabdomyolysis. Consider pravastatin or rosuvastatin "
        "which are alternative transport substrates per CPIC."
    ),
    ("SLCO1B1", "SIMVASTATIN", PHENOTYPE_PF): (
        RISK_TOXIC, 0.96,
        "Avoid simvastatin. Poor SLCO1B1 function results in markedly elevated plasma "
        "statin concentrations, with high risk of rhabdomyolysis and acute kidney injury. "
        "Prescribe a low-risk statin (pravastatin, rosuvastatin) at conservative doses."
    ),

    # ── TPMT / AZATHIOPRINE (active drug) ────────────────────────────────────
    ("TPMT", "AZATHIOPRINE", PHENOTYPE_NM): (
        RISK_SAFE, 0.95,
        "Standard azathioprine or 6-mercaptopurine dosing is appropriate per CPIC guidelines."
    ),
    ("TPMT", "AZATHIOPRINE", PHENOTYPE_IM): (
        RISK_ADJUST, 0.90,
        "Reduce azathioprine starting dose by 30–70% (CPIC guideline). Intermediate "
        "TPMT metabolizers accumulate thiopurine metabolites at elevated concentrations. "
        "Monitor blood counts closely."
    ),
    ("TPMT", "AZATHIOPRINE", PHENOTYPE_PM): (
        RISK_TOXIC, 0.98,
        "Avoid azathioprine / 6-mercaptopurine or reduce dose by ≥90%. Poor TPMT "
        "metabolizers accumulate cytotoxic thiopurine nucleotides, causing severe, "
        "potentially fatal myelosuppression and bone-marrow failure. Consider alternative "
        "immunosuppressant therapy."
    ),

    # ── DPYD / FLUOROURACIL (active drug) ────────────────────────────────────
    ("DPYD", "FLUOROURACIL", PHENOTYPE_NM): (
        RISK_SAFE, 0.95,
        "Standard fluorouracil dosing is appropriate. Monitor for common 5-FU toxicities."
    ),
    ("DPYD", "FLUOROURACIL", PHENOTYPE_IM): (
        RISK_ADJUST, 0.90,
        "Reduce fluorouracil starting dose by 25–50% (CPIC guideline). Decreased DPYD "
        "activity slows 5-FU clearance, increasing risk of severe mucositis, neutropenia, "
        "and treatment-limiting toxicity."
    ),
    ("DPYD", "FLUOROURACIL", PHENOTYPE_PM): (
        RISK_TOXIC, 0.98,
        "Avoid fluorouracil / capecitabine. Poor DPYD metabolizers cannot clear 5-FU; "
        "standard doses cause rapid, life-threatening systemic toxicity including "
        "profound neutropenia and severe mucositis. Consider alternative chemotherapy."
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_risk(
    gene: str,
    phenotype: str,
    drug: str,
    diplotype: str,
) -> RiskResult:
    """
    Classify pharmacogenomic risk for a given gene-drug-phenotype combination.

    Parameters
    ----------
    gene      : e.g. "CYP2D6"
    phenotype : e.g. "Poor Metabolizer"
    drug      : e.g. "CODEINE"  (case-insensitive — normalised to upper)
    diplotype : e.g. "*4/*4"

    Returns
    -------
    RiskResult with risk_label, severity, confidence_score, recommendation.
    Falls back to a conservative "Adjust Dosage" result if gene-drug pair is
    not in the look-up table.
    """
    drug_key = drug.upper().strip()
    key = (gene, drug_key, phenotype)
    rule = _RULES.get(key)

    if rule is None:
        logger.warning(
            "No risk rule for (%s, %s, %s) — applying conservative fallback.",
            gene, drug_key, phenotype,
        )
        risk_label = RISK_ADJUST
        confidence = 0.60
        recommendation = (
            f"No specific CPIC guideline available for {gene} / {drug.title()} / "
            f"{phenotype}. Exercise clinical caution and consult a clinical pharmacist."
        )
    else:
        risk_label, confidence, recommendation = rule

    severity = SEVERITY_LEVEL.get(risk_label, "moderate")

    return RiskResult(
        risk_label       = risk_label,
        severity         = severity,
        confidence_score = confidence,
        guideline_source = "CPIC",
        recommendation   = recommendation,
        primary_gene     = gene,
        phenotype        = phenotype,
        diplotype        = diplotype,
        drug             = drug.title(),
    )
