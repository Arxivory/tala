
"""Signal-based movement phase segmentation."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from tala.movement_analysis.profiles import MovementProfile


@dataclass
class MovementSegment:
    """A movement interval between two turning points."""

    phase: str
    direction: str

    start_index: int
    end_index: int

    start_frame: int
    end_frame: int

    start_time: float
    end_time: float

    duration_seconds: float

    start_value: float
    end_value: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SegmentationResult:
    """Output of movement phase segmentation."""

    exercise: str
    signal_name: str

    segments: list[MovementSegment]
    turning_points: list[dict]

    signal_min: float
    signal_max: float
    signal_range: float

    sample_count: int

    def to_dict(self) -> dict:
        return {
            "exercise": self.exercise,
            "signal_name": self.signal_name,
            "segments": [segment.to_dict() for segment in self.segments],
            "turning_points": self.turning_points,
            "signal_min": self.signal_min,
            "signal_max": self.signal_max,
            "signal_range": self.signal_range,
            "sample_count": self.sample_count,
        }


def _smooth_signal(
    values: np.ndarray,
    window: int,
) -> np.ndarray:
    """Apply a centered rolling median to reduce small fluctuations."""

    if window <= 1:
        return values.copy()

    if window % 2 == 0:
        window += 1

    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )




def _find_turning_points(
    values: np.ndarray,
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    min_phase_frames: int,
    min_prominence: float,
) -> list[dict]:
    """Detect plateau-aware local maxima and minima."""

    if len(values) < 3:
        return []

    differences = np.diff(values)
    signs = np.sign(differences)

    # Ignore zero differences when identifying direction changes.
    nonzero_indices = np.flatnonzero(signs)

    extrema = []

    for position in range(1, len(nonzero_indices)):
        left_diff = nonzero_indices[position - 1]
        right_diff = nonzero_indices[position]

        left_direction = signs[left_diff]
        right_direction = signs[right_diff]

        if left_direction == right_direction:
            continue

        # Determine the middle of a plateau, if one exists.
        plateau_start = left_diff + 1
        plateau_end = right_diff

        index = (plateau_start + plateau_end) // 2

        kind = (
            "high"
            if left_direction > 0 and right_direction < 0
            else "low"
        )

        # Retain the detected local extremum.
        extrema.append({
            "index": int(index),
            "kind": kind,
        })

    # Explicitly retain global extrema when they are interior points.
    global_max_index = int(np.argmax(values))
    global_min_index = int(np.argmin(values))

    for index, kind in [
        (global_max_index, "high"),
        (global_min_index, "low"),
    ]:
        if index == 0 or index == len(values) - 1:
            continue

        already_detected = any(
            point["kind"] == kind
            and abs(point["index"] - index) < min_phase_frames
            for point in extrema
        )

        if not already_detected:
            extrema.append({
                "index": index,
                "kind": kind,
            })

    extrema.sort(key=lambda point: point["index"])

    if not extrema:
        return [
            {
                "index": 0,
                "frame_index": int(frame_indices[0]),
                "time": float(timestamps[0]),
                "value": float(values[0]),
                "kind": "endpoint",
            },
            {
                "index": len(values) - 1,
                "frame_index": int(frame_indices[-1]),
                "time": float(timestamps[-1]),
                "value": float(values[-1]),
                "kind": "endpoint",
            },
        ]

    # Calculate prominence relative to neighboring extrema
    # or recording endpoints.
    for position, point in enumerate(extrema):
        previous_index = (
            extrema[position - 1]["index"]
            if position > 0
            else 0
        )

        next_index = (
            extrema[position + 1]["index"]
            if position < len(extrema) - 1
            else len(values) - 1
        )

        index = point["index"]

        prominence = min(
            abs(values[index] - values[previous_index]),
            abs(values[index] - values[next_index]),
        )

        point["prominence"] = float(prominence)

    # Remove extrema caused by small fluctuations.
    filtered = [
        point
        for point in extrema
        if point["prominence"] >= min_prominence
    ]

    # Always retain interior global extrema.
    # These represent the largest observed movement excursion and
    # should not be discarded by local prominence filtering.

    for index, kind in [
        (int(np.argmax(values)), "high"),
        (int(np.argmin(values)), "low"),
    ]:
        if index == 0 or index == len(values) - 1:
            continue

        existing = next(
            (
                point for point in filtered
                if point["index"] == index
            ),
            None,
        )

        if existing is None:
            filtered.append({
                "index": index,
                "kind": kind,
                "prominence": float(signal_range if "signal_range" in locals() else np.ptp(values)),
            })

    filtered.sort(key=lambda point: point["index"])

    # Enforce minimum separation.
    separated = []

    for point in filtered:
        if not separated:
            separated.append(point)
            continue

        previous = separated[-1]

        if point["index"] - previous["index"] < min_phase_frames:
            if point["prominence"] > previous["prominence"]:
                separated[-1] = point
        else:
            separated.append(point)

    # Ensure alternating high/low extrema.
    alternating = []

    for point in separated:
        if not alternating:
            alternating.append(point)
            continue

        previous = alternating[-1]

        if point["kind"] == previous["kind"]:
            if point["kind"] == "high":
                if values[point["index"]] > values[previous["index"]]:
                    alternating[-1] = point
            else:
                if values[point["index"]] < values[previous["index"]]:
                    alternating[-1] = point
        else:
            alternating.append(point)

    points = [
        {
            "index": 0,
            "frame_index": int(frame_indices[0]),
            "time": float(timestamps[0]),
            "value": float(values[0]),
            "kind": "endpoint",
        }
    ]

    for point in alternating:
        index = point["index"]

        points.append({
            "index": int(index),
            "frame_index": int(frame_indices[index]),
            "time": float(timestamps[index]),
            "value": float(values[index]),
            "kind": point["kind"],
        })

    points.append({
        "index": len(values) - 1,
        "frame_index": int(frame_indices[-1]),
        "time": float(timestamps[-1]),
        "value": float(values[-1]),
        "kind": "endpoint",
    })

    return points


def segment_movement(
    signal,
    profile: MovementProfile,
    frame_indices=None,
    timestamps=None,
    smoothing_window: int = 5,
    min_phase_frames: int = 3,
    min_prominence: float | None = None,
) -> SegmentationResult:
    """Segment a movement signal into increasing/decreasing phases.

    Parameters
    ----------
    signal:
        One-dimensional numerical movement signal.

    profile:
        Exercise profile defining the phase labels.

    frame_indices:
        Optional original video frame indices.

    timestamps:
        Optional timestamps in seconds.

    smoothing_window:
        Rolling median window. Use 1 to disable smoothing.

    min_phase_frames:
        Minimum frame separation between detected extrema.

    min_prominence:
        Minimum signal change required to retain an interior extremum.
        If omitted, 5% of the observed signal range is used.

    Returns
    -------
    SegmentationResult
    """

    values = np.asarray(signal, dtype=float)

    if values.ndim != 1:
        raise ValueError("Movement signal must be one-dimensional.")

    if len(values) < 3:
        raise ValueError("At least three samples are required.")

    if not np.all(np.isfinite(values)):
        raise ValueError("Movement signal contains non-finite values.")

    if smoothing_window < 1:
        raise ValueError("smoothing_window must be at least 1.")

    if min_phase_frames < 1:
        raise ValueError("min_phase_frames must be at least 1.")

    count = len(values)

    if frame_indices is None:
        frame_indices = np.arange(count)

    frame_indices = np.asarray(frame_indices)

    if len(frame_indices) != count:
        raise ValueError("frame_indices length must match signal length.")

    if timestamps is None:
        timestamps = np.arange(count, dtype=float)

    timestamps = np.asarray(timestamps, dtype=float)

    if len(timestamps) != count:
        raise ValueError("timestamps length must match signal length.")

    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps contain non-finite values.")

    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing.")

    smoothed = _smooth_signal(values, smoothing_window)

    signal_min = float(np.min(smoothed))
    signal_max = float(np.max(smoothed))
    signal_range = signal_max - signal_min

    if min_prominence is None:
        min_prominence = signal_range * 0.05

    if min_prominence < 0:
        raise ValueError("min_prominence cannot be negative.")

    turning_points = _find_turning_points(
        values=smoothed,
        frame_indices=frame_indices,
        timestamps=timestamps,
        min_phase_frames=min_phase_frames,
        min_prominence=min_prominence,
    )

    segments = []

    for start, end in zip(turning_points[:-1], turning_points[1:]):
        start_index = start["index"]
        end_index = end["index"]

        start_value = float(smoothed[start_index])
        end_value = float(smoothed[end_index])

        difference = end_value - start_value

        if difference > min_prominence:
            direction = "increasing"
            phase = profile.increasing_phase

        elif difference < -min_prominence:
            direction = "decreasing"
            phase = profile.decreasing_phase

        else:
            direction = "stable"
            phase = "transition_or_hold"

        segments.append(
            MovementSegment(
                phase=phase,
                direction=direction,
                start_index=start_index,
                end_index=end_index,
                start_frame=int(frame_indices[start_index]),
                end_frame=int(frame_indices[end_index]),
                start_time=float(timestamps[start_index]),
                end_time=float(timestamps[end_index]),
                duration_seconds=float(
                    timestamps[end_index] - timestamps[start_index]
                ),
                start_value=start_value,
                end_value=end_value,
            )
        )

    return SegmentationResult(
        exercise=profile.name,
        signal_name=profile.primary_signal,
        segments=segments,
        turning_points=turning_points,
        signal_min=signal_min,
        signal_max=signal_max,
        signal_range=signal_range,
        sample_count=count,
    )
