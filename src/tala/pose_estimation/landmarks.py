"""MediaPipe pose-landmark initialization and extraction."""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarker


LANDMARK_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


# Body and limb connections using MediaPipe's standard 33-landmark indices.
POSE_CONNECTIONS = {
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 29), (29, 31), (27, 31),
    (28, 30), (30, 32), (28, 32),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
}


def create_pose_landmarker(
    model_path: str | Path,
    pose_config: dict,
) -> PoseLandmarker:
    """Initialize MediaPipe PoseLandmarker in VIDEO mode."""
    model_options = python.BaseOptions(
        model_asset_path=str(Path(model_path).resolve())
    )

    options = vision.PoseLandmarkerOptions(
        base_options=model_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=pose_config["num_poses"],
        min_pose_detection_confidence=pose_config[
            "min_pose_detection_confidence"
        ],
        min_pose_presence_confidence=pose_config[
            "min_pose_presence_confidence"
        ],
        min_tracking_confidence=pose_config[
            "min_tracking_confidence"
        ],
        output_segmentation_masks=False,
    )

    return vision.PoseLandmarker.create_from_options(options)


def create_mp_image(frame):
    """Convert an OpenCV BGR frame into a MediaPipe RGB image."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )


def extract_landmark_rows(
    result,
    frame_index: int,
    fps: float,
    timestamp_ms: int,
) -> list[dict]:
    """Convert a MediaPipe result into CSV-compatible landmark rows."""
    rows = []

    if not result.pose_landmarks:
        return rows

    for pose_id, landmarks in enumerate(result.pose_landmarks):
        world_landmarks = result.pose_world_landmarks[pose_id]

        for landmark_id, landmark in enumerate(landmarks):
            world = world_landmarks[landmark_id]

            rows.append({
                "frame_index": frame_index,
                "timestamp_seconds": frame_index / fps,
                "timestamp_ms": timestamp_ms,
                "pose_id": pose_id,
                "landmark_id": landmark_id,
                "landmark_name": LANDMARK_NAMES[landmark_id],
                "x_normalized": landmark.x,
                "y_normalized": landmark.y,
                "z_relative": landmark.z,
                "visibility": landmark.visibility,
                "presence": landmark.presence,
                "world_x_m": world.x,
                "world_y_m": world.y,
                "world_z_m": world.z,
            })

    return rows