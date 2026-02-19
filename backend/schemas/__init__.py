# backend/schemas/__init__.py
try:
    from backend.schemas.response import (
        PredictResponse,
        ErrorResponse,
        DetectedVariant,
        RiskAssessment,
        PharmacoGenomicProfile,
        ClinicalRecommendation,
        LLMExplanation,
        QualityMetrics,
    )
except ModuleNotFoundError:
    from schemas.response import (
        PredictResponse,
        ErrorResponse,
        DetectedVariant,
        RiskAssessment,
        PharmacoGenomicProfile,
        ClinicalRecommendation,
        LLMExplanation,
        QualityMetrics,
    )

__all__ = [
    "PredictResponse", "ErrorResponse", "DetectedVariant",
    "RiskAssessment", "PharmacoGenomicProfile", "ClinicalRecommendation",
    "LLMExplanation", "QualityMetrics",
]
