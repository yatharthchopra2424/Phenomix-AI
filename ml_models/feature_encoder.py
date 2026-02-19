"""
feature_encoder.py
==================
One-hot encode a genomic window around a variant for input into the
1D-CNN + BiLSTM variant function classifier.

Encoding:
  • Nucleotide → orthogonal binary vector: A=[1,0,0,0], C=[0,1,0,0],
    G=[0,0,1,0], T=[0,0,0,1], N=[0,0,0,0]
  • Output tensor shape: (4, window_size * 2 + 1)
    e.g. 50 bp flanking on each side → (4, 101)

Because we do not mount a live reference genome in this prototype, the
flanking sequence is synthesised deterministically from the genomic
coordinates (reproducible across calls but not biologically accurate).
For production, replace _fetch_sequence() with a pyfaidx / Ensembl REST
call against hg38.
"""

from __future__ import annotations

import hashlib
import numpy as np
from typing import Tuple

# Nucleotide ordering used throughout the model
BASES = ["A", "C", "G", "T"]
BASE_INDEX = {b: i for i, b in enumerate(BASES)}
ONE_HOT = np.eye(4, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_variant(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    window: int = 50,
) -> np.ndarray:
    """
    One-hot encode a (window * 2 + 1)-length nucleotide window centred on the
    variant position.

    Returns
    -------
    np.ndarray of shape (4, window * 2 + 1), dtype float32.
    """
    seq = _fetch_sequence(chrom, pos, ref, alt, window)
    return _one_hot_encode(seq)


def _fetch_sequence(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    window: int,
) -> str:
    """
    Fetch the (window * 2 + 1)-bp sequence centred on `pos`.

    Production implementation: replace with pyfaidx or Ensembl REST.
    Prototype: generates a deterministic synthetic sequence derived from
    a hash of the coordinate, ensuring reproducibility.
    """
    total_len = window * 2 + 1
    # Seed a deterministic pseudo-sequence from the coordinate hash
    seed_material = f"{chrom}:{pos}:{ref}:{alt}".encode()
    digest = hashlib.sha256(seed_material).hexdigest()

    # Map hex digits to ACGT (4 symbols from 16 possibilities)
    hex_to_base = {
        "0": "A", "1": "A", "2": "A", "3": "A",
        "4": "C", "5": "C", "6": "C", "7": "C",
        "8": "G", "9": "G", "a": "G", "b": "G",
        "c": "T", "d": "T", "e": "T", "f": "T",
    }
    # Repeat digest until we have enough characters
    raw = "".join(hex_to_base[c] for c in (digest * ((total_len // 64) + 2)))
    left_flank  = raw[:window]
    right_flank = raw[window + 1: window + 1 + window]

    # Centre position holds the ALT allele (first base)
    centre = (alt[0] if alt and alt[0] in BASE_INDEX else ref[0]).upper()
    if centre not in BASE_INDEX:
        centre = "A"

    sequence = left_flank + centre + right_flank
    return sequence[:total_len]


def _one_hot_encode(sequence: str) -> np.ndarray:
    """Convert a nucleotide string to a (4, L) float32 one-hot matrix."""
    L = len(sequence)
    matrix = np.zeros((4, L), dtype=np.float32)
    for i, base in enumerate(sequence.upper()):
        idx = BASE_INDEX.get(base, -1)
        if idx >= 0:
            matrix[idx, i] = 1.0
    return matrix
