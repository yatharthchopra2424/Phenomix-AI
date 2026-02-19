"""
vcf_parser.py
=============
Parse a VCF file (VCFv4.1 / 4.2) and return a list of ParsedVariant objects.

Strategy:
  • Primary path  — cyvcf2 (Cython/htslib wrapper, Linux/macOS).
  • Fallback path — pure-Python line-by-line reader (Windows dev environment).

The provided NA12877.vcf characteristics:
  - Reference assembly: hg38
  - FORMAT: GT only (phased with '|' separator)
  - INFO tags: KM, KFP, KFF, MTD (NO GENE/STAR/RS pharmacogenomic annotations)
  - QUAL: always 0 (uninformative — use KM/KFP/KFF for QC)
  - Single sample: NA12877
"""

from __future__ import annotations

import io
import os
import sys
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ParsedVariant:
    chrom: str
    pos: int          # 1-based genomic position
    ref: str
    alt: str          # primary ALT allele (multi-allelic sites split to first ALT)
    variant_id: str   # rsID from ID column, or '.'
    # Genotype fields
    gt_raw: str       # raw genotype string e.g. "0|1"
    phased: bool      # True when separator is '|'
    allele1: int      # copy-1 allele index (0=ref, 1=alt)
    allele2: int      # copy-2 allele index
    zygosity: str     # "hom_ref" | "het" | "hom_alt" | "missing"
    # QC fields from INFO
    km: Optional[float] = None     # minimum k-mer count (quality proxy)
    kfp: Optional[int] = None      # k-mer pedigree failures
    kff: Optional[int] = None      # k-mer founder failures
    mtd_methods: List[str] = field(default_factory=list)  # calling methods


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _zygosity(a1: int, a2: int) -> str:
    if a1 == -1 or a2 == -1:
        return "missing"
    if a1 == 0 and a2 == 0:
        return "hom_ref"
    if a1 == a2:
        return "hom_alt"
    return "het"


def _parse_gt(gt_str: str):
    """Return (allele1, allele2, phased) from a raw GT string like '0|1' or '1/0'."""
    if "|" in gt_str:
        sep, phased = "|", True
    else:
        sep, phased = "/", False
    parts = gt_str.split(sep)
    def _to_int(s: str) -> int:
        return -1 if s in (".", "") else int(s)
    a1 = _to_int(parts[0]) if len(parts) > 0 else -1
    a2 = _to_int(parts[1]) if len(parts) > 1 else -1
    return a1, a2, phased


def _parse_info(info_str: str) -> dict:
    """Parse semicolon-separated INFO field into a dict."""
    result: dict = {}
    if info_str in (".", ""):
        return result
    for token in info_str.split(";"):
        if "=" in token:
            k, v = token.split("=", 1)
            result[k] = v
        else:
            result[token] = True
    return result


# ---------------------------------------------------------------------------
# cyvcf2 parser (preferred)
# ---------------------------------------------------------------------------

def _parse_with_cyvcf2(filepath: str) -> List[ParsedVariant]:
    """Parse using cyvcf2 (Cython/htslib). Raises ImportError on Windows."""
    import cyvcf2  # type: ignore

    variants: List[ParsedVariant] = []
    vcf = cyvcf2.VCF(filepath)
    sample_idx = 0  # NA12877 is the first (and only) sample

    for rec in vcf:
        chrom = rec.CHROM
        pos   = rec.POS
        ref   = rec.REF
        alts  = rec.ALT
        if not alts:
            continue
        alt = alts[0]  # take primary ALT

        variant_id = rec.ID if rec.ID else "."

        # Genotype
        gt_arr = rec.genotypes[sample_idx]  # e.g. [0, 1, True] = [a1, a2, phased]
        a1, a2 = int(gt_arr[0]), int(gt_arr[1])
        phased = bool(gt_arr[2])
        zyg = _zygosity(a1, a2)

        # GT raw string reconstruction
        sep = "|" if phased else "/"
        gt_raw = f"{a1}{sep}{a2}"

        # INFO
        km  = rec.INFO.get("KM")
        kfp = rec.INFO.get("KFP")
        kff = rec.INFO.get("KFF")
        mtd_raw = rec.INFO.get("MTD", "")
        mtd = mtd_raw.split(",") if mtd_raw else []

        variants.append(ParsedVariant(
            chrom=chrom, pos=pos, ref=ref, alt=alt,
            variant_id=str(variant_id),
            gt_raw=gt_raw, phased=phased,
            allele1=a1, allele2=a2, zygosity=zyg,
            km=float(km) if km is not None else None,
            kfp=int(kfp) if kfp is not None else None,
            kff=int(kff) if kff is not None else None,
            mtd_methods=mtd,
        ))

    vcf.close()
    logger.info("cyvcf2 parsed %d variants from %s", len(variants), filepath)
    return variants


# ---------------------------------------------------------------------------
# Pure-Python fallback parser (Windows dev / no htslib)
# ---------------------------------------------------------------------------

def _parse_pure_python(filepath: str) -> List[ParsedVariant]:
    """Fallback pure-Python VCF parser — no external dependencies."""
    variants: List[ParsedVariant] = []
    sample_col_idx: Optional[int] = None

    opener = io.open
    with opener(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n\r")

            # Skip meta lines
            if line.startswith("##"):
                continue

            # Header line — locate the sample column
            if line.startswith("#CHROM"):
                cols = line.lstrip("#").split("\t")
                # Standard 9-col VCF: CHROM POS ID REF ALT QUAL FILTER INFO FORMAT [SAMPLE…]
                try:
                    fmt_idx = cols.index("FORMAT")
                    sample_col_idx = fmt_idx + 1
                except ValueError:
                    sample_col_idx = 9
                continue

            # Data lines
            cols = line.split("\t")
            if len(cols) < 9:
                continue

            chrom      = cols[0]
            pos        = int(cols[1])
            variant_id = cols[2]
            ref        = cols[3]
            alt_field  = cols[4]
            info_str   = cols[7]

            alt = alt_field.split(",")[0] if "," in alt_field else alt_field

            # FORMAT and sample
            fmt_fields = cols[8].split(":") if len(cols) > 8 else []
            s_idx = sample_col_idx if sample_col_idx is not None else 9
            sample_val = cols[s_idx] if len(cols) > s_idx else "."
            sample_fields = sample_val.split(":")

            gt_raw = "."
            if "GT" in fmt_fields:
                gt_pos = fmt_fields.index("GT")
                if gt_pos < len(sample_fields):
                    gt_raw = sample_fields[gt_pos]

            a1, a2, phased = _parse_gt(gt_raw)
            zyg = _zygosity(a1, a2)

            # Parse INFO
            info = _parse_info(info_str)
            km   = float(info["KM"])  if "KM"  in info else None
            kfp  = int(info["KFP"])   if "KFP" in info else None
            kff  = int(info["KFF"])   if "KFF" in info else None
            mtd  = info.get("MTD", "").split(",") if "MTD" in info else []

            variants.append(ParsedVariant(
                chrom=chrom, pos=pos, ref=ref, alt=alt,
                variant_id=variant_id,
                gt_raw=gt_raw, phased=phased,
                allele1=a1, allele2=a2, zygosity=zyg,
                km=km, kfp=kfp, kff=kff, mtd_methods=mtd,
            ))

    logger.info("Pure-Python parser: %d variants from %s", len(variants), filepath)
    return variants


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_vcf(filepath: str) -> List[ParsedVariant]:
    """
    Parse a VCF file and return a list of ParsedVariant objects.

    Automatically selects cyvcf2 (if available) or falls back to the
    pure-Python parser for Windows / environments without htslib.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"VCF file not found: {filepath}")

    if sys.platform != "win32":
        try:
            return _parse_with_cyvcf2(filepath)
        except ImportError:
            logger.warning("cyvcf2 not available — falling back to pure-Python parser.")

    return _parse_pure_python(filepath)


def validate_vcf_header(filepath: str) -> bool:
    """
    Quick header validation — confirms the file starts with ##fileformat=VCF.
    Used by the API layer before committing to full parsing.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            first_line = fh.readline().strip()
        return first_line.startswith("##fileformat=VCF")
    except Exception:
        return False
