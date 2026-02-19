"""
llm_engine.py
=============
NVIDIA GLM5 streaming LLM for RAG-grounded clinical explanation generation.

Uses the exact client configuration specified in the project requirements:
  • base_url : https://integrate.api.nvidia.com/v1
  • model    : z-ai/glm5
  • enable_thinking=True / clear_thinking=False

The LLM is strictly confined to the provided RAG context chunks.
It must not introduce information not present in the retrieved guidelines.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Colour support for thinking-block output in TTY environments
_USE_COLOR      = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
_REASONING_COLOR = "\033[90m" if _USE_COLOR else ""
_RESET_COLOR     = "\033[0m"  if _USE_COLOR else ""


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a board-certified clinical pharmacogenomics consultant.

Your task is to provide a clear, concise, and medically accurate explanation of a
pharmacogenomic risk assessment for a healthcare provider.

STRICT RULES:
1. You MUST base every statement ONLY on the CPIC / PharmGKB guideline excerpts
   provided below in the [CONTEXT] section.  Do NOT introduce any information that
   is not explicitly present in the context.
2. If the context does not contain enough information to answer a specific sub-question,
   state "The retrieved guidelines do not address this specific case" rather than guessing.
3. Cite specific rsIDs, star alleles, and Activity Scores when they appear in the context.
4. Write for a clinician audience: precise but not excessively technical.
5. Maximum 4 paragraphs.  Do not use bullet points.

[CONTEXT]
{context}
[END CONTEXT]
"""

_USER_PROMPT_TEMPLATE = """
Patient genomic profile:
  Gene         : {gene}
  Diplotype    : {diplotype}
  Phenotype    : {phenotype}
  Drug         : {drug}
  Risk Label   : {risk_label}
  Detected rsIDs: {rsids}

Using ONLY the provided guideline context above, explain:
1. The biological mechanism by which this patient's CYP2D6/CYP2C19/etc. variant
   affects drug metabolism.
2. Why the calculated phenotype ({phenotype}) results in the risk label ({risk_label}).
3. What specific variants (rsIDs / star alleles) were detected and their functional impact.
4. The CPIC-recommended clinical action for this patient.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_explanation(
    context_chunks: List[str],
    patient_profile: Dict,
) -> str:
    """
    Call NVIDIA GLM5 with streaming to generate a RAG-grounded clinical explanation.

    Parameters
    ----------
    context_chunks   : list of raw CPIC guideline text strings from the retriever
    patient_profile  : dict with keys: gene, diplotype, phenotype, drug,
                       risk_label, rsids (list of str)

    Returns
    -------
    Full LLM explanation as a plain text string.
    """
    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else (
        "No specific CPIC guideline chunks were retrieved for this gene-drug combination."
    )

    system_prompt = _SYSTEM_PROMPT.format(context=context_text)

    rsids_str = ", ".join(patient_profile.get("rsids", [])) or "None detected"
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        gene       = patient_profile.get("gene",       "Unknown"),
        diplotype  = patient_profile.get("diplotype",  "Unknown"),
        phenotype  = patient_profile.get("phenotype",  "Unknown"),
        drug       = patient_profile.get("drug",       "Unknown"),
        risk_label = patient_profile.get("risk_label", "Unknown"),
        rsids      = rsids_str,
    )

    try:
        return _call_nvidia_glm5(system_prompt, user_prompt)
    except Exception as exc:
        logger.error("LLM call failed: %s", exc, exc_info=True)
        return _fallback_explanation(patient_profile, context_chunks)


# ---------------------------------------------------------------------------
# NVIDIA GLM5 streaming caller
# ---------------------------------------------------------------------------

def _call_nvidia_glm5(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI  # type: ignore

    client = OpenAI(
        base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key  = os.environ.get("NVIDIA_API_KEY", ""),
    )
    model = os.environ.get("NVIDIA_MODEL", "z-ai/glm5")

    completion = client.chat.completions.create(
        model    = model,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature = 0.2,        # low temperature for factual clinical text
        top_p       = 0.9,
        max_tokens  = 1024,
        extra_body  = {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "clear_thinking":  False,
            }
        },
        stream = True,
    )

    output_parts: List[str] = []

    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        if not chunk.choices or chunk.choices[0].delta is None:
            continue
        delta = chunk.choices[0].delta

        # Reasoning tokens (printed to stderr in TTY — not included in output)
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            logger.debug("%s<thinking>%s%s", _REASONING_COLOR, reasoning, _RESET_COLOR)

        # Main content tokens
        if getattr(delta, "content", None) is not None:
            output_parts.append(delta.content)

    return "".join(output_parts).strip()


# ---------------------------------------------------------------------------
# Fallback when LLM is unavailable
# ---------------------------------------------------------------------------

def _fallback_explanation(
    profile: Dict,
    context_chunks: List[str],
) -> str:
    """
    Return a structured plain-text explanation assembled from retrieved context
    when the LLM API is unreachable.  Clearly labelled as a template fallback.
    """
    gene      = profile.get("gene",       "the relevant pharmacogene")
    diplotype = profile.get("diplotype",  "unknown")
    phenotype = profile.get("phenotype",  "unknown phenotype")
    drug      = profile.get("drug",       "the prescribed drug")
    risk      = profile.get("risk_label", "undetermined risk")

    summary = (
        f"[Offline fallback — LLM unavailable]\n\n"
        f"The patient carries the {gene} diplotype {diplotype}, classifying them as "
        f"a {phenotype}. Based on CPIC guidelines, this genotype confers a '{risk}' "
        f"classification for {drug}. "
    )

    if context_chunks:
        # Append the first retrieved chunk as verbatim guideline reference
        summary += (
            f"\n\nRelevant CPIC guideline excerpt:\n{context_chunks[0][:600]}…"
        )

    return summary
