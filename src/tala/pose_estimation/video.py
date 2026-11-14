"""OpenCV video input, output, and pose-overlay rendering."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tala.pose_estimation.landmarks import POSE_CONNECTIONS


def open_video(video_path: str | Path) -> cv2.VideoCapture:
    """Open a video and validate its basic metadata."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("Video has invalid FPS metadata.")

    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Video has invalid dimensions.")

    return cap


def read_video_metadata(
    cap: cv2.VideoCapture,
) -> dict:
    """Read video properties from an already-open capture."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = (
        reported_frame_count / fps
        if reported_frame_count > 0
        else None
    )

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "reported_frame_count": reported_frame_count,
        "duration_seconds": duration,
    }


def create_video_writer(
    output_path: str | Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    """Create an MP4 video writer for annotated frames."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"Could not initialize output video writer: {output_path}"
        )

    return writer


def draw_pose_overlay(
    frame: np.ndarray,
    pose_landmarks: list,
) -> np.ndarray:
    """Draw MediaPipe pose landmarks and connections onto a frame."""
    height, width = frame.shape[:2]
    annotated = frame.copy()

    for landmarks in pose_landmarks:
        points = []

        for landmark in landmarks:
            px = int(landmark.x * width)
            py = int(landmark.y * height)
            points.append((px, py))

        for start, end in POSE_CONNECTIONS:
            x1, y1 = points[start]
            x2, y2 = points[end]

            if (
                0 <= x1 < width
                and 0 <= y1 < height
                and 0 <= x2 < width
                and 0 <= y2 < height
            ):
                cv2.line(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

        for px, py in points:
            if 0 <= px < width and 0 <= py < height:
                cv2.circle(
                    annotated,
                    (px, py),
                    3,
                    (0, 0, 255),
                    -1,
                )

    return annotated