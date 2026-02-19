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


class LLMUnavailableError(RuntimeError):
    """Raised when LLM output is required but the remote LLM is unavailable."""

# Colour support for thinking-block output in TTY environments
_USE_COLOR      = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
_REASONING_COLOR = "\033[90m" if _USE_COLOR else ""
_RESET_COLOR     = "\033[0m"  if _USE_COLOR else ""
_nvidia_llm_auth_disabled: bool = False


def _clean_env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip().strip('"').strip("'")


def _is_auth_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in (401, 403):
        return True

    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) in (401, 403):
        return True

    text = str(exc).lower()
    return "403" in text or "401" in text or "forbidden" in text or "unauthor" in text


def _should_use_nvidia_llm() -> bool:
    if _nvidia_llm_auth_disabled:
        return False

    api_key = _clean_env("NVIDIA_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return False

    return True


def _diagnose_nvidia_access() -> str:
    """Return a short diagnostic string for NVIDIA API access state."""
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(
            base_url = _clean_env("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            api_key  = _clean_env("NVIDIA_API_KEY", ""),
        )
        models = client.models.list()
        count = len(getattr(models, "data", []) or [])
        return (
            "NVIDIA API key is recognized (models list accessible), but inference is denied for this key/model. "
            f"Visible models: {count}."
        )
    except Exception:
        return "NVIDIA API key is not authorized for models list or is invalid."


def _candidate_models() -> List[str]:
    """Return ordered model fallback list for NVIDIA chat completions."""
    primary = _clean_env("NVIDIA_MODEL", "z-ai/glm5")
    fallback_env = _clean_env("NVIDIA_MODEL_FALLBACKS", "")
    fallbacks = [item.strip() for item in fallback_env.split(",") if item.strip()]

    ordered = [primary, *fallbacks, "openai/gpt-oss-120b"]
    unique: List[str] = []
    for model in ordered:
        if model not in unique:
            unique.append(model)
    return unique


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a clinical pharmacogenomics system. Analyze the provided VCF variants, RAG guidelines, and Deep Learning risk scores. You MUST output your final analysis adhering strictly to the provided JSON schema. Do not output markdown, introductory text, or anything outside the JSON.

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
    strict_llm: bool = False,
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

    global _nvidia_llm_auth_disabled

    if not _should_use_nvidia_llm():
        if strict_llm:
            raise LLMUnavailableError(
                "NVIDIA LLM unavailable: missing/invalid API configuration. " + _diagnose_nvidia_access()
            )
        return _fallback_explanation(patient_profile, context_chunks)

    try:
        return _call_nvidia_glm5(system_prompt, user_prompt)
    except Exception as exc:
        if _is_auth_error(exc):
            if not _nvidia_llm_auth_disabled:
                logger.warning(
                    "NVIDIA LLM authorization failed (%s). Disabling NVIDIA LLM for this process and using fallback explanation.",
                    exc,
                )
            _nvidia_llm_auth_disabled = True
            if strict_llm:
                raise LLMUnavailableError(
                    f"NVIDIA LLM authorization failed: {exc}. {_diagnose_nvidia_access()}"
                ) from exc
            return _fallback_explanation(patient_profile, context_chunks)

        if strict_llm:
            raise LLMUnavailableError(
                f"NVIDIA LLM request failed: {exc}. {_diagnose_nvidia_access()}"
            ) from exc
        logger.error("LLM call failed: %s", exc, exc_info=True)
        return _fallback_explanation(patient_profile, context_chunks)


# ---------------------------------------------------------------------------
# NVIDIA GLM5 streaming caller
# ---------------------------------------------------------------------------

def _call_nvidia_glm5(system_prompt: str, user_prompt: str) -> str:
    if _nvidia_llm_auth_disabled:
        raise RuntimeError("NVIDIA LLM disabled due to prior authorization failure.")

    from openai import OpenAI  # type: ignore

    client = OpenAI(
        base_url = _clean_env("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key  = _clean_env("NVIDIA_API_KEY", ""),
        timeout  = 120.0,
    )

    model_errors: List[str] = []
    for model in _candidate_models():
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.60,
                top_p=0.95,
                max_tokens=2048,
                stream=True,
            )

            output_parts: List[str] = []
            reasoning_parts: List[str] = []

            for chunk in completion:
                if not getattr(chunk, "choices", None):
                    continue
                if not chunk.choices or chunk.choices[0].delta is None:
                    continue
                delta = chunk.choices[0].delta

                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    logger.debug("%s<thinking>%s%s", _REASONING_COLOR, reasoning, _RESET_COLOR)
                    reasoning_parts.append(str(reasoning))

                content = getattr(delta, "content", None)
                if content is not None:
                    output_parts.append(str(content))

            output_text = "".join(output_parts).strip()
            if output_text:
                logger.info("NVIDIA LLM model selected: %s", model)
                return output_text

            reasoning_text = "".join(reasoning_parts).strip()
            if reasoning_text:
                logger.info("NVIDIA LLM model selected (reasoning fallback): %s", model)
                return reasoning_text

            model_errors.append(f"{model}: empty response")
            logger.warning("NVIDIA LLM model returned empty response: %s", model)
        except Exception as exc:
            model_errors.append(f"{model}: {exc}")
            logger.warning("NVIDIA LLM model failed (%s): %s", model, exc)
            continue

    raise RuntimeError("All candidate NVIDIA models failed. " + " | ".join(model_errors))


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
        guideline_chunk = next(
            (
                chunk for chunk in context_chunks
                if "cpic guideline" in chunk.lower() or "pharmgkb" in chunk.lower()
            ),
            context_chunks[0],
        )

        # Append a retrieved guideline-like reference snippet
        summary += (
            f"\n\nRelevant CPIC guideline excerpt:\n{guideline_chunk[:600]}…"
        )

    return summary
