"""Video timestamp reconstruction and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_timestamps(
    landmarks: pd.DataFrame,
    frame_index: np.ndarray,
    fps: float,
):
    """Build a strictly increasing timestamp for every video frame."""
    fallback = frame_index / fps

    if "timestamp_seconds" not in landmarks.columns:
        return fallback, "metadata_fps"

    timestamp_rows = pd.DataFrame({
        "frame_index": landmarks["frame_index"].to_numpy(),
        "timestamp_seconds": pd.to_numeric(
            landmarks["timestamp_seconds"],
            errors="coerce",
        ).to_numpy(dtype=float),
    }).groupby("frame_index")["timestamp_seconds"].agg(
        ["min", "max", "first"]
    )

    has_timestamps = (
        timestamp_rows["min"].notna()
        & timestamp_rows["max"].notna()
    )

    inconsistent = has_timestamps & (
        timestamp_rows["max"] - timestamp_rows["min"] > 1e-6
    )

    if inconsistent.any():
        raise ValueError(
            "Inconsistent timestamp_seconds values within a video frame."
        )

    timestamp_values = timestamp_rows["first"]

    valid_timestamp_rows = timestamp_values[
        np.isfinite(timestamp_values.to_numpy(dtype=float))
    ]

    if len(valid_timestamp_rows) < 2:
        return (
            fallback,
            "metadata_fps_fallback_insufficient_csv_timestamps",
        )

    known_frames = valid_timestamp_rows.index.to_numpy(dtype=int)
    known_times = valid_timestamp_rows.to_numpy(dtype=float)

    if np.any(np.diff(known_times) <= 0):
        raise ValueError("Video timestamps must be strictly increasing.")

    timestamps = np.interp(
        frame_index,
        known_frames,
        known_times,
    )

    before_first = frame_index < known_frames[0]
    after_last = frame_index > known_frames[-1]

    timestamps[before_first] = (
        known_times[0]
        + (frame_index[before_first] - known_frames[0]) / fps
    )

    timestamps[after_last] = (
        known_times[-1]
        + (frame_index[after_last] - known_frames[-1]) / fps
    )

    if (
        not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) <= 0)
    ):
        raise ValueError(
            "Video timestamps must be finite and strictly increasing."
        )

    return timestamps, "pose_landmarks.csv.timestamp_seconds"