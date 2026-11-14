"""Pose-extraction coverage and quality reporting."""

from __future__ import annotations

import pandas as pd


EXTRACTION_LIMITATIONS = [
    "Single-camera pose estimation.",
    "Estimated world coordinates are not calibrated motion capture.",
    "No independent ground-truth pose measurements.",
    "Front-oblique perspective may introduce depth ambiguity.",
    "Pose detection does not establish biomechanical validity.",
]


def calculate_detection_rate(
    detected_frames: int,
    processed_frames: int,
) -> float:
    """Calculate the proportion of processed frames with a detected pose."""
    if processed_frames <= 0:
        return 0.0

    return detected_frames / processed_frames


def build_extraction_report(
    landmarks_df: pd.DataFrame,
    processed_frames: int,
    detected_frames: int,
) -> dict:
    """Build the extraction summary and limitations report."""
    detection_rate = calculate_detection_rate(
        detected_frames,
        processed_frames,
    )

    unique_landmarks = (
        int(landmarks_df["landmark_id"].nunique())
        if not landmarks_df.empty
        else 0
    )

    return {
        "stage": "phase3_3_pose_extraction",
        "status": "completed",
        "processed_frames": processed_frames,
        "detected_frames": detected_frames,
        "detection_rate": detection_rate,
        "landmark_rows": len(landmarks_df),
        "unique_landmarks": unique_landmarks,
        "frames_without_detection": (
            processed_frames - detected_frames
        ),
        "limitations": EXTRACTION_LIMITATIONS.copy(),
    }