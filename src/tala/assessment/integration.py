"""Integration of movement analysis and baselines into assessment output."""

from __future__ import annotations

from typing import Any


def summarize_movement(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Create a compact movement-analysis summary."""

    segmentation = report.get("segmentation") or {}
    repetitions = report.get("repetitions") or {}
    completeness = report.get("completeness") or {}

    segments = segmentation.get("segments") or []

    phase_summary: dict[str, dict[str, float | int]] = {}

    for segment in segments:
        phase = str(segment.get("phase", "unknown"))

        entry = phase_summary.setdefault(
            phase,
            {"count": 0, "total_duration_seconds": 0.0},
        )

        entry["count"] += 1
        entry["total_duration_seconds"] += float(
            segment.get("duration_seconds", 0.0)
        )

    return {
        "exercise": report.get("exercise"),
        "sample_count": report.get("sample_count"),
        "signal": {
            "name": segmentation.get("signal_name"),
            "minimum": segmentation.get("signal_min"),
            "maximum": segmentation.get("signal_max"),
            "range": segmentation.get("signal_range"),
        },
        "phases": phase_summary,
        "repetitions": {
            "count": repetitions.get("repetition_count"),
            "status": repetitions.get("status", "unavailable"),
            "cycle_start_extreme": repetitions.get(
                "cycle_start_extreme"
            ),
            "durations_seconds": [
                item.get("duration_seconds")
                for item in repetitions.get("repetitions", [])
            ],
        },
        "completeness": {
            "status": completeness.get("status", "unavailable"),
            "is_complete": completeness.get("is_complete"),
            "reason": completeness.get("reason"),
        },
        "limitations": report.get("limitations", []),
    }


def integrate_movement_assessment(
    assessment: dict[str, Any],
    movement_report: dict[str, Any],
    baseline_comparison: dict[str, Any],
) -> dict[str, Any]:
    """Attach movement and baseline observations to an assessment."""

    result = dict(assessment)

    result["movement_analysis"] = summarize_movement(
        movement_report
    )

    result["baseline_comparison"] = baseline_comparison

    result.setdefault("limitations", [])

    additional_limitations = [
        "Movement analysis is based on the supplied exercise signal and does not independently validate pose estimation or OpenSim reconstruction.",
        "Baseline comparisons are descriptive comparisons with prior recorded sessions, not clinical reference ranges.",
    ]

    for limitation in additional_limitations:
        if limitation not in result["limitations"]:
            result["limitations"].append(limitation)

    return result