
"""Movement-sequence completeness evaluation."""

from __future__ import annotations

import numpy as np

from tala.movement_analysis.profiles import MovementProfile
from tala.movement_analysis.segmentation import SegmentationResult


def evaluate_completeness(
    signal,
    result: SegmentationResult,
    profile: MovementProfile,
    tolerance: float | None = None,
    minimum_signal_range: float = 1e-6,
) -> dict:
    """Evaluate whether a recorded movement sequence appears complete.

    For cyclic profiles, the first and final signal values are compared
    with the expected cycle-start extreme.

    The function does not infer missing frames or extrapolate motion.
    """

    values = np.asarray(signal, dtype=float)

    if values.ndim != 1:
        raise ValueError("Movement signal must be one-dimensional.")

    if len(values) != result.sample_count:
        raise ValueError(
            "Signal length does not match the segmentation result."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("Movement signal contains non-finite values.")

    if tolerance is None:
        tolerance = profile.completion_tolerance

    if not 0 <= tolerance < 0.5:
        raise ValueError("tolerance must be between 0 and 0.5.")

    if not profile.is_cyclic:
        return {
            "exercise": profile.name,
            "status": "not_applicable",
            "is_complete": None,
            "reason": (
                "This profile represents a non-cyclic movement. "
                "Use a dedicated hold-duration or posture-stability "
                "evaluation instead."
            ),
        }

    signal_min = float(np.min(values))
    signal_max = float(np.max(values))
    signal_range = signal_max - signal_min

    if signal_range < minimum_signal_range:
        return {
            "exercise": profile.name,
            "status": "unknown",
            "is_complete": None,
            "reason": "Insufficient movement variation to evaluate completeness.",
        }

    if profile.cycle_start_extreme == "high":
        expected_value = signal_max
        start_distance = abs(values[0] - signal_max)
        end_distance = abs(values[-1] - signal_max)
    else:
        expected_value = signal_min
        start_distance = abs(values[0] - signal_min)
        end_distance = abs(values[-1] - signal_min)

    allowed_distance = tolerance * signal_range

    starts_at_expected_extreme = bool(
        start_distance <= allowed_distance
    )

    ends_at_expected_extreme = bool(
        end_distance <= allowed_distance
    )

    is_complete = bool(
        starts_at_expected_extreme
        and ends_at_expected_extreme
    )

    if is_complete:
        status = "complete"
        reason = (
            "The recording begins and ends sufficiently close to "
            "the expected cycle-start extreme."
        )
    else:
        status = "incomplete"

        missing = []

        if not starts_at_expected_extreme:
            missing.append("the expected starting posture")

        if not ends_at_expected_extreme:
            missing.append("the expected ending posture")

        reason = (
            "The recording does not sufficiently capture "
            + " and ".join(missing)
            + "."
        )

    return {
        "exercise": profile.name,
        "status": status,
        "is_complete": is_complete,
        "cycle_start_extreme": profile.cycle_start_extreme,
        "expected_extreme_value": expected_value,
        "observed_start_value": float(values[0]),
        "observed_end_value": float(values[-1]),
        "signal_min": signal_min,
        "signal_max": signal_max,
        "signal_range": signal_range,
        "tolerance": tolerance,
        "allowed_distance": allowed_distance,
        "starts_at_expected_extreme": starts_at_expected_extreme,
        "ends_at_expected_extreme": ends_at_expected_extreme,
        "reason": reason,
    }
