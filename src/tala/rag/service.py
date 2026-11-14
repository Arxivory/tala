"""Assessment-aware interface for TALA's RAG explanation component."""

from __future__ import annotations

from typing import Any

from tala.rag.generation import generate_response
from tala.rag.schemas import RAGRequest


def _build_rag_context(
    assessment: dict[str, Any],
) -> dict[str, Any]:
    """Select relevant assessment fields without passing full reports."""

    keys = (
        "daily_checkin",
        "movement_observations",
        "movement_analysis",
        "baseline_comparison",
        "gravity_context",
        "reconstruction_quality",
        "wellness_summary",
        "wellness_flags",
        "context_flags",
        "contextual_insights",
    )

    return {
        key: assessment[key]
        for key in keys
        if key in assessment
    }


def explain_assessment(
    question: str,
    assessment: dict[str, Any],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Generate an evidence-grounded explanation of assessment findings."""

    request = RAGRequest(
        question=question,
        assessment_context=_build_rag_context(assessment),
        top_k=top_k,
    )

    try:
        response = generate_response(request)

    except Exception as exc:
        return {
            "status": "generation_failed",
            "answer": None,
            "abstained": True,
            "abstention_reason": (
                "The evidence explanation could not be generated. "
                "The assessment results remain available."
            ),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "limitations": [
                "RAG generation failed; no generated explanation is available.",
                "Assessment findings were computed independently and were not modified.",
            ],
            "evidence": [],
        }

    evidence = []

    for item in response.evidence:
        metadata = item.metadata

        evidence.append({
            "source_id": metadata.source_id,
            "title": metadata.title,
            "page_start": metadata.page_start,
            "page_end": metadata.page_end,
            "population": metadata.population,
            "domain": metadata.domain,
            "distance": item.distance,
        })

    return {
        "status": "abstained" if response.abstained else "available",
        "answer": response.answer,
        "abstained": response.abstained,
        "abstention_reason": response.abstention_reason,
        "limitations": list(response.limitations),
        "evidence": evidence,
    }