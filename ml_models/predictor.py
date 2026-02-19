"""
predictor.py
============
Loads the trained VariantFunctionClassifier and exposes a predict_variant_function()
call for use by the annotation pipeline.

If no checkpoint file exists (dev / demo mode), the model is initialised with
random weights and predictions are labelled with confidence=0.50 to signal that
results are illustrative only.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from ml_models.architecture import VariantFunctionClassifier, CLASS_LABELS
from ml_models.feature_encoder import encode_variant

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default weight file path
# ---------------------------------------------------------------------------
_WEIGHTS_DIR  = Path(__file__).parent / "weights"
_WEIGHTS_FILE = _WEIGHTS_DIR / "variant_classifier.pt"

_DEMO_MODE_CONFIDENCE = 0.50  # advertised confidence when weights are random

# ---------------------------------------------------------------------------
# Module-level singleton (loaded once at startup)
# ---------------------------------------------------------------------------
_model: VariantFunctionClassifier = None  # type: ignore
_demo_mode: bool = True
_device: torch.device = torch.device("cpu")


def load_model() -> None:
    """
    Initialise the model.  Called once from the FastAPI lifespan handler.
    Loads serialised weights from disk if available, otherwise runs in demo mode.
    """
    global _model, _demo_mode, _device

    if torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")

    _model = VariantFunctionClassifier(seq_len=101).to(_device)
    _model.eval()

    if _WEIGHTS_FILE.exists():
        try:
            state = torch.load(str(_WEIGHTS_FILE), map_location=_device)
            _model.load_state_dict(state)
            _demo_mode = False
            logger.info("Loaded pre-trained weights from %s", _WEIGHTS_FILE)
        except Exception as exc:
            logger.warning("Failed to load weights (%s) — running in demo mode.", exc)
            _demo_mode = True
    else:
        _demo_mode = True
        logger.warning(
            "No weights file found at %s — model running in demo mode "
            "(random weights, confidence=%.2f).",
            _WEIGHTS_FILE, _DEMO_MODE_CONFIDENCE,
        )


def is_loaded() -> bool:
    return _model is not None


def is_demo_mode() -> bool:
    return _demo_mode


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_variant_function(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    window: int = 50,
) -> Tuple[str, float]:
    """
    Predict the functional class of a variant from its one-hot encoded
    genomic sequence window.

    Returns
    -------
    (function_class, confidence_score)
    e.g. ("no_function", 0.93)

    In demo mode, confidence is capped at _DEMO_MODE_CONFIDENCE.
    """
    if _model is None:
        load_model()

    # Encode sequence
    seq_array = encode_variant(chrom, pos, ref, alt, window=window)  # (4, 101)
    tensor = torch.from_numpy(seq_array).unsqueeze(0).to(_device)    # (1, 4, 101)

    with torch.no_grad():
        log_probs = _model(tensor)              # (1, 4)
        probs = torch.exp(log_probs).squeeze()  # (4,)

    pred_idx  = int(probs.argmax().item())
    raw_conf  = float(probs[pred_idx].item())
    func_class = CLASS_LABELS[pred_idx]

    # In demo mode, cap confidence to signal unreliability
    confidence = min(raw_conf, _DEMO_MODE_CONFIDENCE) if _demo_mode else raw_conf

    logger.debug(
        "ML prediction (%s): chrom=%s pos=%d ref=%s alt=%s → %s (conf=%.3f%s)",
        "DEMO" if _demo_mode else "LIVE",
        chrom, pos, ref, alt, func_class, confidence,
        " [demo]" if _demo_mode else "",
    )
    return func_class, round(confidence, 4)
