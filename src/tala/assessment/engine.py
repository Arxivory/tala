
"""Entry point for TALA's contextual assessment."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .rules import (
    build_movement_wellness_insights,
    build_quality_insights,
    build_wellness_insights,
    build_wellness_flags,
    build_gravity_insights,
    extract_gravity_context,
    extract_movement_observations,
    extract_reconstruction_quality,
)
from .schemas import validate_daily_checkin


ASSESSMENT_VERSION = "0.2.0"


def generate_assessment(
    daily_checkin: dict[str, Any] | None,
    mapped_coordinates: pd.DataFrame,
    ik_report: dict[str, Any] | None,
    gravity_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate a transparent, non-diagnostic contextual assessment.

    Parameters
    ----------
    daily_checkin:
        User-reported wellness information. Partial check-ins are accepted.
    mapped_coordinates:
        Mapped OpenSim coordinates loaded as a DataFrame.
        Angular coordinates are expected in radians.
    ik_report:
        Parsed inverse-kinematics quality report.
    gravity_report:
        Optional parsed gravity-analysis report.

    Returns
    -------
    dict
        Structured movement observations, gravity context, wellness
        context, reconstruction quality, and limitations.
    """
    checkin = validate_daily_checkin(daily_checkin)

    movement_observations = extract_movement_observations(
        mapped_coordinates
    )

    quality = extract_reconstruction_quality(ik_report)

    gravity_context = extract_gravity_context(gravity_report)

    wellness_summary, wellness_flags = build_wellness_flags(checkin)

    wellness_insights = build_wellness_insights(wellness_flags)

    movement_context_insights = build_movement_wellness_insights(
        movement_observations,
        wellness_flags,
    )

    quality_insights = build_quality_insights(quality)

    gravity_insights = build_gravity_insights(gravity_context)

    contextual_insights = (
        wellness_insights
        + movement_context_insights
        + quality_insights
        + gravity_insights
    )

    context_flags = list(wellness_flags)

    if not wellness_summary["is_complete"]:
        context_flags.append({
            "code": "incomplete_daily_checkin",
            "category": "data_completeness",
            "severity": "information",
            "evidence": {
                "missing_fields": wellness_summary["missing_fields"]
            },
            "message": (
                "Some check-in fields were not reported. "
                "Unanswered fields were not interpreted as normal."
            ),
        })

    if quality["status"] != "reported":
        context_flags.append({
            "code": "reconstruction_quality_limited",
            "category": "reconstruction_quality",
            "severity": "information",
            "evidence": {
                "quality_status": quality["status"]
            },
            "message": (
                "The reconstruction has limitations that should be "
                "considered when interpreting movement observations."
            ),
        })

    limitations = [
        "This assessment is informational and is not a medical diagnosis.",
        "User-reported wellness information is not independently verified.",
        "Movement estimates are derived from monocular video and approximate "
        "OpenSim inverse kinematics.",
        "Coordinate ranges are not clinically validated range-of-motion "
        "measurements.",
        "The current pipeline does not calculate video-derived joint loads "
        "or establish exercise safety.",
        "No readiness score, medical clearance, or automatic exercise "
        "prescription is generated.",
        "Gravity outputs describe modeled gravitational potential-energy "
        "changes and do not measure muscle work or metabolic expenditure.",
        "The gravity comparison holds the reconstructed movement fixed "
        "across gravitational environments and does not predict movement "
        "adaptation.",
    ]

    limitations.extend(quality.get("limitations", []))
    limitations.extend(gravity_context.get("limitations", []))

    return {
        "assessment_version": ASSESSMENT_VERSION,
        "assessment_type": "contextual_wellness_and_movement_assessment",
        "daily_checkin": checkin,
        "movement_observations": movement_observations,
        "gravity_context": gravity_context,
        "reconstruction_quality": quality,
        "wellness_summary": wellness_summary,
        "wellness_flags": wellness_flags,
        "context_flags": context_flags,
        "contextual_insights": contextual_insights,
        "limitations": list(dict.fromkeys(limitations)),
        "evidence_references": [],
    }
