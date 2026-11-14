"""Input loading and validation for full-body kinematic reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tala.kinematic_reconstruction.preprocessing import (
    prepare_landmark_samples,
)
from tala.kinematic_reconstruction.timestamps import build_timestamps


VISIBILITY_THRESHOLD = 0.50

DIRECT_MARKERS = {
    "RSJC": ("right_shoulder", 2.0),
    "LSJC": ("left_shoulder", 2.0),
    "REJC": ("right_elbow", 2.0),
    "LEJC": ("left_elbow", 2.0),
    "IK_WRIST_R": ("right_wrist", 1.5),
    "IK_WRIST_L": ("left_wrist", 1.5),
    "RHJC": ("right_hip", 2.0),
    "LHJC": ("left_hip", 2.0),
    "RKJC": ("right_knee", 2.0),
    "LKJC": ("left_knee", 2.0),
    "RAJC": ("right_ankle", 1.5),
    "LAJC": ("left_ankle", 1.5),
    "RTOE": ("right_foot_index", 0.75),
    "LTOE": ("left_foot_index", 0.75),
}


def load_inputs(input_dir: str | Path):
    """Load and validate Phase 3.3 outputs for OpenSim IK."""
    input_dir = Path(input_dir)

    landmarks_path = input_dir / "pose_landmarks.csv"
    metadata_path = input_dir / "video_metadata.json"

    for path in (landmarks_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(path)

    landmarks = pd.read_csv(landmarks_path)

    if landmarks.empty:
        raise ValueError(
            f"Pose landmark CSV contains no rows: {landmarks_path}"
        )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Video metadata must be a JSON object: {metadata_path}"
        )

    required_columns = {
        "frame_index",
        "pose_id",
        "landmark_name",
        "visibility",
        "presence",
        "x_normalized",
        "y_normalized",
        "world_x_m",
        "world_y_m",
        "world_z_m",
    }

    missing = required_columns - set(landmarks.columns)

    if missing:
        raise ValueError(
            f"{landmarks_path} is missing required Phase 3.3 columns: "
            f"{sorted(missing)}"
        )

    required_metadata = {"fps", "width", "height"}
    missing_metadata = required_metadata - set(metadata)

    if missing_metadata:
        raise ValueError(
            f"{metadata_path} is missing required Phase 3.3 metadata: "
            f"{sorted(missing_metadata)}"
        )

    try:
        fps = float(metadata["fps"])
        image_width = int(metadata["width"])
        image_height = int(metadata["height"])

        frame_count_value = metadata.get("processed_frame_count")

        if frame_count_value is None:
            frame_count_value = int(
                landmarks["frame_index"].max()
            ) + 1

        frame_count = int(frame_count_value)

    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{metadata_path} contains invalid FPS, dimensions, "
            "or processed frame count."
        ) from error

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video FPS: {fps}")

    if frame_count < 2 or image_width <= 0 or image_height <= 0:
        raise ValueError("Invalid video dimensions or frame count.")

    if landmarks["pose_id"].nunique() > 1:
        raise ValueError(
            "Multiple poses are present. Select a single tracked "
            "person before running full-body IK."
        )

    landmarks = landmarks[
        landmarks["frame_index"].between(0, frame_count - 1)
    ].copy()

    duplicate_rows = landmarks.duplicated(
        ["frame_index", "landmark_name"],
        keep=False,
    )

    if duplicate_rows.any():
        raise ValueError(
            "Pose landmarks contain duplicate frame/landmark rows."
        )

    frame_index = np.arange(frame_count, dtype=int)

    timestamps, timestamp_source = build_timestamps(
        landmarks,
        frame_index,
        fps,
    )

    landmark_names = {
        point
        for point, _ in DIRECT_MARKERS.values()
    }

    points = {}
    image_points = {}
    sample_status = {}
    valid_counts = {}

    for name in sorted(landmark_names):
        rows = (
            landmarks[landmarks["landmark_name"] == name]
            .set_index("frame_index")
            .reindex(frame_index)
        )

        values = rows[
            [
                "world_x_m",
                "world_y_m",
                "world_z_m",
                "x_normalized",
                "y_normalized",
                "visibility",
                "presence",
            ]
        ].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(dtype=float)

        world = values[:, :3].copy()
        image_xy = values[:, 3:5].copy()
        confidence = np.minimum(values[:, 5], values[:, 6])

        valid_world = (
            np.isfinite(world).all(axis=1)
            & np.isfinite(confidence)
            & (confidence >= VISIBILITY_THRESHOLD)
        )

        valid_image = (
            np.isfinite(image_xy).all(axis=1)
            & np.isfinite(confidence)
            & (confidence >= VISIBILITY_THRESHOLD)
        )

        world[~valid_world] = np.nan
        image_xy[~valid_image] = np.nan

        interpolated_world = prepare_landmark_samples(
            world,
            f"{name} world coordinates",
            allow_all_missing=True,
        )

        interpolated_image = prepare_landmark_samples(
            image_xy,
            f"{name} image coordinates",
            allow_all_missing=True,
        )

        points[name] = interpolated_world
        image_points[name] = interpolated_image

        sample_status[name] = np.full(
            frame_count,
            "unavailable",
            dtype=object,
        )

        sample_status[name][valid_world] = "observed"

        filled = np.isfinite(interpolated_world).all(axis=1)

        sample_status[name][
            ~valid_world & filled
        ] = "interpolated"

        valid_counts[name] = int(valid_world.sum())

    return (
        points,
        image_points,
        sample_status,
        valid_counts,
        frame_index,
        timestamps,
        fps,
        image_width,
        image_height,
        timestamp_source,
        landmarks_path,
        metadata_path,
    )