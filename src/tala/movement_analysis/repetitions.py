
"""Repetition counting from segmented movement signals."""

from __future__ import annotations

from tala.movement_analysis.profiles import MovementProfile
from tala.movement_analysis.segmentation import SegmentationResult


def _classify_turning_points(
    result: SegmentationResult,
    tolerance: float,
) -> list[dict]:
    """Classify endpoints as high/low using the observed signal range."""

    points = result.turning_points
    signal_range = result.signal_range

    if signal_range <= 0:
        return []

    high_threshold = result.signal_max - tolerance * signal_range
    low_threshold = result.signal_min + tolerance * signal_range

    classified = []

    for index, point in enumerate(points):
        value = point["value"]
        kind = point["kind"]

        if kind == "endpoint":
            if value >= high_threshold:
                kind = "high"
            elif value <= low_threshold:
                kind = "low"
            else:
                kind = "intermediate"

        classified.append({
            **point,
            "kind": kind,
        })

    return classified


def count_repetitions(
    result: SegmentationResult,
    profile: MovementProfile,
    tolerance: float | None = None,
) -> dict:
    """Count complete repetitions in a segmented movement sequence.

    A complete repetition requires three consecutive qualifying
    turning points: expected start extreme, opposite extreme,
    expected start extreme.

    This is a simple signal-based counter, not a clinical or
    production-grade exercise recognition system.
    """

    if not profile.is_cyclic:
        return {
            "exercise": profile.name,
            "repetition_count": None,
            "status": "not_applicable",
            "reason": "This movement profile is not cyclic.",
            "repetitions": [],
        }

    if tolerance is None:
        tolerance = profile.completion_tolerance

    if not 0 <= tolerance < 0.5:
        raise ValueError("tolerance must be between 0 and 0.5.")

    points = _classify_turning_points(result, tolerance)

    expected_start = profile.cycle_start_extreme
    expected_opposite = "low" if expected_start == "high" else "high"

    repetitions = []
    index = 0

    while index + 2 < len(points):
        first, middle, last = points[index:index + 3]

        matches_cycle = (
            first["kind"] == expected_start
            and middle["kind"] == expected_opposite
            and last["kind"] == expected_start
        )

        if matches_cycle:
            repetitions.append({
                "repetition_number": len(repetitions) + 1,
                "start_frame": first["frame_index"],
                "end_frame": last["frame_index"],
                "start_time": first["time"],
                "end_time": last["time"],
                "duration_seconds": (
                    last["time"] - first["time"]
                ),
                "start_extreme": first["kind"],
                "middle_extreme": middle["kind"],
                "end_extreme": last["kind"],
            })

            # Move to the final point so the next cycle can begin there.
            index += 2
        else:
            index += 1

    return {
        "exercise": profile.name,
        "repetition_count": len(repetitions),
        "status": "available",
        "cycle_start_extreme": expected_start,
        "tolerance": tolerance,
        "repetitions": repetitions,
    }
